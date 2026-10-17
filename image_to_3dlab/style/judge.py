"""A vision-language judge: Qwen3-VL looks at approved pictures and a concept and answers yes or no.

    with LlamaServer() as server:
        found = Judge(server.post).judge([refs_a, refs_b], concept, bible["style"]["criteria"])
    found["score"]       # mean probability of "yes", 0..1
    found["answers"]     # per question, so the reason can be shown

The answer is not read as a word but as the probability the model gives to "yes" against
"no" on its one-token reply, which makes a continuous score that can be calibrated. The
2026 work on vision-language judges says they rank well but give unreliable absolute
grades (arXiv 2604.25235); yes/no probabilities, two sets of references and thresholds
fitted to the user's own labels are the counter-measures.

The model is the Qwen3-VL-8B already used as Qwen-Image's text encoder, plus its vision
file; it runs in llama.cpp's server on the GPU. Starting the server takes the time of
loading 6 GB, so a caller keeps one open for a whole batch.
"""

from __future__ import annotations

import base64
import io
import json
import math
import socket
import subprocess
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from image_to_3dlab.style import models
from image_to_3dlab.style.embed import open_rgb

SIDE = 448  # longest side sent to the model: about 200 vision tokens a picture
YES = {"yes", "y"}
NO = {"no", "n"}
PROMPT = ("The first {count} pictures are approved concept art that defines a game's art style. "
          "The last picture is a new concept for the same game. Judge only the art style, not the subject. "
          "{question} Answer with yes or no only.")


def yes_probability(top_logprobs: list[dict]) -> float:
    """P(yes) / (P(yes) + P(no)) over the reply's candidate first tokens; 0.5 when neither appears."""
    yes = no = 0.0
    for candidate in top_logprobs:
        word = str(candidate.get("token", "")).strip().strip(".").lower()
        probability = math.exp(float(candidate.get("logprob", -math.inf)))
        if word in YES:
            yes += probability
        elif word in NO:
            no += probability
    return 0.5 if yes + no == 0 else yes / (yes + no)


def _data_url(path: Path) -> str:
    image = open_rgb(path)
    image.thumbnail((SIDE, SIDE), Image.LANCZOS)
    if max(image.size) < SIDE:  # small pictures are scaled up, so every picture costs the same
        scale = SIDE / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_request(references: list[Path], concept: Path, question: str) -> dict:
    """An OpenAI-style chat request: the references, the concept last, the question, one token back."""
    content = [{"type": "image_url", "image_url": {"url": _data_url(p)}} for p in [*references, concept]]
    content.append({"type": "text", "text": PROMPT.format(count=len(references), question=question)})
    return {"messages": [{"role": "user", "content": content}], "max_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": 20}


class Judge:
    def __init__(self, post: Callable[[dict], dict]):
        self.post = post

    def ask(self, references: list[Path], concept: Path, question: str) -> float:
        reply = self.post(build_request(references, concept, question))
        tokens = reply["choices"][0]["logprobs"]["content"]
        return yes_probability(tokens[0]["top_logprobs"] if tokens else [])

    def judge(self, reference_sets: list[list[Path]], concept: Path, criteria: list[str]) -> dict:
        """Every question against every reference set; the score is the mean probability of yes."""
        if not criteria:
            raise ValueError("the bible has no questions for the judge (style.criteria)")
        answers = []
        for question in criteria:
            values = [self.ask(refs, concept, question) for refs in reference_sets]
            answers.append({"question": question, "yes": sum(values) / len(values)})
        return {"score": sum(a["yes"] for a in answers) / len(answers), "answers": answers}


def find_parts(hub: Path | None = None) -> dict[str, Path]:
    """The server binary, the model and its vision file, or MissingModel saying how to get each."""
    server = models.llama_server()
    if server is None:
        raise models.MissingModel(f"llama.cpp is not installed; run: {models.BOOTSTRAP} --only llama")
    found = sorted((models.QWEN3VL.cache_dir(hub) / "snapshots").glob(f"*/{models.QWEN3VL.files[0]}"))
    if not found:
        raise models.MissingModel("Qwen3-VL-8B is not downloaded; it comes with Qwen-Image's setup")
    mmproj = models.mmproj_path(hub)
    if mmproj is None:
        raise models.MissingModel(f"Qwen3-VL's vision file ({models.MMPROJ}) is not downloaded; "
                                  "it comes with Qwen-Image's setup")
    return {"server": server, "model": found[0], "mmproj": mmproj}


def server_command(server: Path, model: Path, mmproj: Path, port: int) -> list[str]:
    return [str(server), "-m", str(model), "--mmproj", str(mmproj), "--host", "127.0.0.1",
            "--port", str(port), "-ngl", "99", "--ctx-size", "8192", "--no-webui"]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class LlamaServer:
    """llama.cpp's server with Qwen3-VL, for as long as the `with` block lasts."""

    def __init__(self, hub: Path | None = None, startup_seconds: float = 300.0, log: Path | None = None,
                 lock_path: Path | None = None):
        self.parts = find_parts(hub)
        self.port = _free_port()
        self.startup_seconds = startup_seconds
        self.log = log
        self.process: subprocess.Popen | None = None
        self.lock_path = lock_path
        self._lock = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> LlamaServer:
        # the model sits on the GPU for as long as the server runs: the shared lock is held as long
        from image_to_3dlab import gpu_lock

        self._lock = gpu_lock.gpu("qwen3-vl judge", path=self.lock_path or gpu_lock.LOCK)
        self._lock.__enter__()
        try:
            return self._start()
        except BaseException:
            # llama-server may be up and loading the model: stopped before the lock goes
            try:
                self._stop()
            finally:
                self._release()
            raise

    def _start(self) -> LlamaServer:
        out = open(self.log, "wb") if self.log else subprocess.DEVNULL  # noqa: SIM115 - closed with the process
        self.process = subprocess.Popen(server_command(self.parts["server"], self.parts["model"],
                                                       self.parts["mmproj"], self.port),
                                        stdout=out, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + self.startup_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server stopped while loading (exit {self.process.returncode})"
                                   + (f"; see {self.log}" if self.log else ""))
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=5) as reply:
                    if json.loads(reply.read()).get("status") == "ok":
                        return self
            except OSError:
                pass
            time.sleep(1)
        self._stop()
        raise RuntimeError(f"llama-server did not become ready in {self.startup_seconds:.0f} s")

    def _stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def _release(self) -> None:
        lock, self._lock = self._lock, None
        if lock is not None:
            lock.__exit__(None, None, None)

    def __exit__(self, *exc) -> None:
        try:
            self._stop()
        finally:
            self._release()

    def post(self, body: dict) -> dict:
        request = urllib.request.Request(f"{self.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as reply:
            return json.loads(reply.read())
