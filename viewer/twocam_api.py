"""Animate from video: the Animate step for a creature the preset moves do not fit.

People do it here, in the browser; agents run `scripts/twocam_mocap.py` themselves (see
docs/twocam-mocap.md). Each stage is that script, run as a subprocess with this Python, so
both do exactly the same thing:

    prepare  turn, size and paint the rig, render the frame videos start from   ~10 s
    teach    render 2,200 practice poses, then teach the joint reader on them   ~1 h, once
    film     MiniMax H3 films the creature front-on and side-on, on this machine ~6.5 min a take
    capture  read the joints in one take, fit the rig, export the GLB            ~1 min

Everything stays in the rig's own folder, so the Library lists a captured move like a preset:

    output/animate/<run>/
        <stem>_rigged.glb
        <stem>_<move>.glb (+ .provenance.json)   a captured move
        twocam/
            creature/          prepare's files, the practice poses (dataset/), reader.pt
            clips/<move>/      the takes filmed for one move, each with its verdict
            captures/<move>/   the joints read from the take used, and the fitted motion
            steps/run.log      what every stage printed
            pid                the stage running now; a restarted server cleans up after it

Nothing downloads without a click that says what: the Teach button names DINOv2 (346 MB),
and the video model (about 30 GB) has a Download button of its own.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import usage_api
from autorig_api import rig_run_of

from image_to_3dlab import host, processes
from image_to_3dlab.twocam import video

OUTPUT = REPO / "output"
SCRIPT = REPO / "scripts" / "twocam_mocap.py"
SDCPP = REPO / "vendor" / "sdcpp"
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
TAKE = re.compile(r"^take(\d+)_s(\d+)\.webm$")
TERMINAL = {"done", "error", "cancelled"}
KINDS = ("prepare", "teach", "film", "capture")
POSES = 2200            # the script's default `dataset --count`
EPOCHS = 20             # and its default `train --epochs`
MAX_TAKES = 3
LENGTH_RANGE = (0.05, 100.0)
# Measured on an RTX 5070 Ti; only the progress bar's guesses use them.
PREPARE_SECONDS = 15.0
TAKE_SECONDS = 390.0    # one 3 s take at 1024 x 576
CAPTURE_SECONDS = 60.0
TEACH_BANDS = {"dataset": (0.0, 35.0), "train": (35.0, 100.0)}
DINO = {"name": "DINOv2", "id": "facebook/dinov2-with-registers-base", "license": "Apache-2.0", "megabytes": 346}
USAGE = {"prepare": "creature", "teach": "reader", "film": "video", "capture": "animation"}
STARTING = {"prepare": "Preparing the rig", "teach": "Rendering practice poses",
            "film": "Starting the video model", "capture": "Reading the take"}
FILLER = {"the", "it", "its", "an", "and", "then", "of", "to", "on", "in", "with", "as", "at"}
BAD_CLIP = re.compile(r"this clip is not worth fitting: (.*?)\. Generate it again", re.DOTALL)


class Creature:
    """One rig's two-camera folders, all inside the rig's run folder."""

    def __init__(self, rigged: Path):
        self.rigged = rigged
        self.run = rigged.parent
        self.stem = rigged.name.removesuffix("_rigged.glb")
        self.root = self.run / "twocam"
        self.folder = self.root / "creature"
        self.clips = self.root / "clips"
        self.captures = self.root / "captures"
        self.log = self.root / "steps" / "run.log"
        self.pid = self.root / "pid"

    def file(self, name: str) -> Path:
        return self.folder / name

    def prepared(self) -> bool:
        return all(self.file(n).is_file() for n in ("creature.json", "rig_info.json", "rigged.blend", "ref.png"))


def resolve_rig(relative: str, output: Path = OUTPUT) -> Path:
    """A rigged model the Animate tab made, by its path under output/, and nothing else."""
    text = str(relative or "").strip()
    path = (output / text).resolve()
    if not text or path.parent.parent != (output / "animate").resolve() or rig_run_of(path) is None \
            or not path.is_file():
        raise ValueError(f"not a rigged model from the Animate tab: {text!r}")
    return path


def rel(path: Path, output: Path = OUTPUT) -> str:
    return Path(path).resolve().relative_to(Path(output).resolve()).as_posix()


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ------------------------------------------------------------------------------ input


def length_of(value: Any) -> float:
    try:
        length = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("say how long it is, in metres (1.6 for a wolf)") from exc
    low, high = LENGTH_RANGE
    if not low <= length <= high:
        raise ValueError(f"the length must be between {low:g} and {high:g} metres")
    return length


def words_of(value: Any, what: str, limit: int, required: bool = True) -> str:
    """One line of plain words. It starts with a letter or a number, so the script never
    takes it for one of its own options."""
    text = " ".join(str(value or "").split())
    if not text:
        if required:
            raise ValueError(f"say {what}")
        return ""
    if len(text) > limit:
        raise ValueError(f"keep {what} under {limit} characters")
    if not text[0].isalnum():
        raise ValueError(f"start {what} with a word")
    return text


def takes_of(value: Any) -> int:
    try:
        takes = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"takes must be 1 to {MAX_TAKES}") from exc
    if not 1 <= takes <= MAX_TAKES:
        raise ValueError(f"takes must be 1 to {MAX_TAKES}")
    return takes


def move_slug(action: str) -> str:
    """'The wolf rears up and howls' -> 'wolf_rears_howls': the move's name in the Library.
    One-letter words are dropped, so a name never ends like a take's `_b`."""
    words = [w for w in re.findall(r"[a-z0-9]+", action.lower()) if len(w) > 1 and w not in FILLER]
    name = ""
    for word in words[:4]:
        if len(name) + len(word) + 1 > 40:
            break
        name = f"{name}_{word}" if name else word
    name = name or (words[0][:40] if words else "move")
    # a move called "rigged" would be saved over the rig itself (<stem>_rigged.glb)
    return f"move_{name}" if name == "rigged" or name.startswith("rigged_") else name


def move_folder(clips: Path, action: str, end: str) -> Path:
    """Where a move's takes go: the folder already filmed with these exact words (more takes
    of the same move), else a new one."""
    base = move_slug(action)
    for n in range(1, 100):
        folder = clips / (base if n == 1 else f"{base}_{n}")
        said = read_json(folder / "move.json")
        if not isinstance(said, dict) or (said.get("action"), said.get("end") or "") == (action, end):
            return folder
    raise ValueError("too many moves with these words")


def next_seed(folder: Path) -> int:
    """More takes of one move never film the same seed again."""
    seeds = [int(m[2]) for clip in folder.glob("take*_s*.webm") if (m := TAKE.fullmatch(clip.name))]
    return max(seeds, default=0) + 1


def dino_cached(root: Path | None = None) -> bool:
    """Whether DINOv2's weights are here already: the same check `train` makes."""
    return video.dino_cached(root)


def video_plan() -> dict[str, Any]:
    return video.weights_plan()


def engine_ready() -> bool:
    return host.executable(SDCPP, "sd-cli").is_file()


# ----------------------------------------------------------------------------- commands


def script(*args: Any) -> list[str]:
    return [sys.executable, str(SCRIPT), *map(str, args)]


def prepare_command(c: Creature, length_m: float, noun: str, yaw_deg: float | None = None) -> list[str]:
    cmd = script("prepare", c.rigged, c.folder, "--length-m", f"{length_m:g}", f"--noun={noun}",
                 "--source-model", c.rigged)
    if yaw_deg is not None:
        cmd.append(f"--yaw-deg={yaw_deg:g}")
    return cmd


def teach_commands(c: Creature, fetch: bool = False) -> list[list[str]]:
    """Render the practice poses, then teach. `--yes` only when the button that started this
    named DINOv2's download (`fetch`): that click is the person's yes, and nothing else is."""
    rig_info, dataset = c.file("rig_info.json"), c.file("dataset")
    train = script("train", dataset, rig_info, c.file("reader.pt"))
    if fetch is True:
        train.append("--yes")
    return [script("dataset", c.file("rigged.blend"), rig_info, dataset), train]


def film_command(c: Creature, folder: Path, action: str, end: str, takes: int, seed: int,
                 download: bool = False) -> list[str]:
    cmd = script("film", c.folder, action, folder, "--takes", takes, "--seed", seed)
    if end:
        cmd.append(f"--end={end}")
    if c.file("reader.pt").is_file():
        cmd += ["--reader", str(c.file("reader.pt"))]
    if download is True:   # only from the Download button
        cmd += ["--download", "--yes"]
    return cmd


def capture_command(c: Creature, take: Path, slug: str, force: bool = False,
                    record: dict[str, Any] | None = None) -> list[str]:
    cmd = script("capture", take, c.file("reader.pt"), c.file("rig_info.json"), c.file("rigged.blend"),
                 c.captures / slug, "--ref", c.file("ref.png"), f"--name={c.stem}_{slug}")
    record = record or {}
    if record.get("made_with"):
        cmd.append(f"--made-with={record['made_with']}")
    if record.get("license"):
        cmd.append(f"--video-license={record['license']}")
    if force is True:      # only from "Fit anyway"
        cmd.append("--force")
    return cmd


# ----------------------------------------------------------------------------- progress


def eta(elapsed: float, start: float, fraction: float) -> float | None:
    """Seconds to go, from how long the part done so far took; None until worth showing."""
    gained = fraction - start
    if gained <= 0.02 or elapsed <= 1.0 or fraction >= 1.0:
        return None
    return round(elapsed * (1.0 - fraction) / gained, 1)


class Progress:
    """A stage that prints nothing while it works: the bar is a guess from the clock, capped
    short of done."""

    ticks = True

    def __init__(self, expected: float, message: str, clock=time.monotonic):
        self.expected, self.message, self.clock = expected, message, clock
        self.started = clock()

    def begin(self, index: int) -> dict[str, Any] | None:
        return None

    def feed(self, line: str) -> dict[str, Any] | None:
        return None

    def tick(self) -> dict[str, Any] | None:
        elapsed = self.clock() - self.started
        return {"phase": "running", "message": self.message,
                "overall_pct": min(95, round(100 * elapsed / max(self.expected, 1.0)))}


POSES_LINE = re.compile(r"^(\d+)/(\d+) poses rendered, (\d+) s$")
STEP_LINE = re.compile(r"^epoch (\d+) step (\d+) loss ")
EPOCH_LINE = re.compile(r"^epoch (\d+): held-out error front ([\d.]+) px, side ([\d.]+) px$")


class TeachProgress(Progress):
    """Two phases, read from the script's own lines: practice poses ("812/2200 poses rendered,
    300 s"), then teaching ("epoch 3 step 650 loss ..." every 50 steps, and after each round
    "epoch 3: held-out error front 4.10 px, side 3.80 px"). No clock guess: a bar that only
    moves on real news lets the page say when it has heard nothing for minutes."""

    ticks = False

    def __init__(self, epochs: int = EPOCHS, fetches_dino: bool = False, clock=time.monotonic):
        super().__init__(3600.0, STARTING["teach"], clock)
        self.epochs = epochs
        self.fetches_dino = fetches_dino
        self.phase = "dataset"
        self.fraction = 0.0
        self.phase_start: float | None = None   # where this phase's bar stood at its first news
        self.phase_started = self.started
        self.per_round: float | None = None   # steps in one round, known once the first ends
        self.last_step = 0
        self.error: dict[str, float] | None = None

    def begin(self, index: int) -> dict[str, Any] | None:
        if index == 1:
            return self._at("train", 0.0, f"Downloading {DINO['name']} ({DINO['megabytes']} MB), then teaching"
                            if self.fetches_dino else "Getting the reader ready")
        return None

    def feed(self, line: str) -> dict[str, Any] | None:
        line = line.strip()
        if m := POSES_LINE.match(line):
            done, count = int(m[1]), max(int(m[2]), 1)
            return self._at("dataset", done / count, f"Rendering practice poses: {done:,} of {count:,}")
        if m := STEP_LINE.match(line):
            round_, step = int(m[1]), int(m[2])
            self.last_step = step
            within = (step - round_ * self.per_round) / self.per_round if self.per_round else 0.0
            return self._at("train", (round_ + min(max(within, 0.0), 0.99)) / self.epochs, self._words(round_ + 1))
        if m := EPOCH_LINE.match(line):
            round_ = int(m[1])
            if self.per_round is None and self.last_step:
                self.per_round = self.last_step / (round_ + 1)
            self.error = {"front": float(m[2]), "side": float(m[3])}
            return self._at("train", (round_ + 1) / self.epochs, self._words(min(round_ + 2, self.epochs)))
        return None

    def _words(self, round_: int) -> str:
        words = f"Teaching the reader: round {round_} of {self.epochs}"
        if self.error:
            words += f", off by {(self.error['front'] + self.error['side']) / 2:.1f} pixels on poses it has not seen"
        return words

    def _at(self, phase: str, fraction: float, message: str) -> dict[str, Any]:
        if phase != self.phase:
            self.phase, self.fraction, self.phase_start = phase, 0.0, None
        if self.phase_start is None:      # a resumed render starts part-way
            self.phase_start, self.phase_started = min(fraction, 1.0), self.clock()
        self.fraction = max(self.fraction, min(fraction, 1.0))    # never backwards
        low, high = TEACH_BANDS[phase]
        return {"phase": phase, "message": message, "overall_pct": round(low + (high - low) * self.fraction, 1),
                "stage_pct": round(100 * self.fraction),
                "stage_eta_seconds": eta(self.clock() - self.phase_started, self.phase_start, self.fraction)}


DOWNLOAD_LINE = re.compile(r"^downloading (\S+) \(([\d.]+) GB\)$")
DOWNLOAD_PCT = re.compile(r"(\d{1,3})%\|")


class FilmProgress(Progress):
    """Takes filmed so far, from sd-cli's own "generate_video completed in 371.2s" (one per
    take); inside a take the clock guesses, as sd-cli prints nothing while it films. A first
    run may download the video model first: "downloading <file> (11.4 GB)", then its bar."""

    QUIET_DOWNLOAD = 10.0   # seconds without a download line: filming has begun

    def __init__(self, takes: int, clock=time.monotonic):
        super().__init__(TAKE_SECONDS, STARTING["film"], clock)
        self.takes = takes
        self.filmed = 0
        self.take_started = self.started
        self.file: str | None = None
        self.size = ""
        self.heard = self.started

    def feed(self, line: str) -> dict[str, Any] | None:
        line = line.strip()
        if m := DOWNLOAD_LINE.match(line):
            self.file, self.size = m[1].rsplit("/", 1)[-1], m[2]
            return self._downloading(None)
        if self.file and (m := DOWNLOAD_PCT.search(line)):
            return self._downloading(int(m[1]))
        if "generate_video completed" in line:
            self.file = None
            self.filmed += 1
            self.take_started = self.clock()
            return self._event(0.0)
        return None

    def _downloading(self, pct: int | None) -> dict[str, Any]:
        self.heard = self.take_started = self.clock()
        return {"phase": "download", "message": f"Downloading the video model: {self.file} ({self.size} GB)",
                **({"overall_pct": pct} if pct is not None else {}), "takes_done": self.filmed}

    def tick(self) -> dict[str, Any] | None:
        if self.file and self.clock() - self.heard < self.QUIET_DOWNLOAD:
            return None
        self.file = None
        if self.filmed >= self.takes:
            return None
        return self._event(min(0.95, (self.clock() - self.take_started) / TAKE_SECONDS))

    def _event(self, inside: float) -> dict[str, Any]:
        message = (f"Filming take {self.filmed + 1} of {self.takes}" if self.filmed < self.takes
                   else "Judging the takes")
        return {"phase": "film", "message": message, "takes_done": self.filmed,
                "overall_pct": round(100 * (self.filmed + inside) / self.takes, 1)}


def failure(lines: list[str]) -> str | None:
    """The script's own `error: ...`, with any lines after it (Blender's last words)."""
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("error: "):
            return "\n".join([lines[i][len("error: "):], *lines[i + 1:]]).strip()
    return None


def bad_clip_words(message: str) -> str | None:
    """The fit's refusal of a bad take, said for the page (which offers "Fit anyway")."""
    m = BAD_CLIP.search(message)
    if m is None:
        return None
    return (f"The two views in this take disagree too much to fit: {m[1]}. Film another take, "
            "or fit this one anyway.")


def summary_of(path: Path) -> dict[str, Any]:
    """The JSON object the script printed. Its stdout is a file of its own, so it is the whole
    file; the last `{` at the start of a line is tried too, should anything print before it."""
    text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    try:
        return json.loads(text)
    except ValueError:
        start = text.rfind("\n{")
        return json.loads(text[start + 1:]) if start >= 0 else {}


# ---------------------------------------------------------------------------------- jobs


class TwocamJob:
    def __init__(self, job_id: str, kind: str, creature: Creature, commands: list[list[str]],
                 progress: Progress, move: Path | None = None, take: Path | None = None):
        self.id = job_id
        self.kind = kind
        self.creature = creature
        self.commands = commands      # run in order; each prints one JSON summary
        self.progress = progress
        self.move = move              # film and capture: the move's folder under clips/
        self.take = take              # capture: the take used
        self.status = "queued"
        self.started = time.monotonic()
        self.events: list[dict[str, Any]] = []
        self.condition = threading.Condition()
        self.process: subprocess.Popen[str] | None = None
        self.cancel_requested = False
        self.log_lines: deque[str] = deque(maxlen=300)
        self.outputs: list[dict[str, Any]] = []

    def emit(self, event: dict[str, Any]) -> None:
        payload = {"elapsed_seconds": round(time.monotonic() - self.started, 1), **event}
        with self.condition:
            self.events.append(payload)
            self.condition.notify_all()

    def append_log(self, line: str) -> None:
        self.log_lines.append(line)
        if DOWNLOAD_PCT.search(line) and "100%|" not in line:
            return    # a download bar redraws itself many times a second: the file keeps the end
        self.creature.log.parent.mkdir(parents=True, exist_ok=True)
        with self.creature.log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class TwocamJobManager:
    """One stage at a time, for every rig: each one holds the graphics card."""

    def __init__(self, output: Path = OUTPUT):
        self.output = output
        self.jobs: dict[str, TwocamJob] = {}
        self.active: str | None = None
        self.newest: dict[Path, str] = {}   # run folder -> its newest job
        self.lock = threading.Lock()

    def busy(self) -> bool:
        job = self.jobs.get(self.active) if self.active else None
        return job is not None and job.status not in TERMINAL

    def get(self, job_id: str) -> TwocamJob | None:
        return self.jobs.get(job_id) if JOB_ID.fullmatch(job_id) else None

    def newest_for(self, rigged: Path) -> TwocamJob | None:
        job_id = self.newest.get(rigged.parent)
        return self.jobs.get(job_id) if job_id else None

    def finish(self, job: TwocamJob) -> None:
        with self.lock:
            if self.active == job.id:
                self.active = None

    def start(self, kind: str, body: dict[str, Any]) -> TwocamJob:
        """A stage from the page's JSON. ValueError: the request is wrong; RuntimeError: busy."""
        if kind not in KINDS:
            raise ValueError(f"no such stage: {kind!r}")
        rigged = resolve_rig(str(body.get("rig", "")), self.output)
        if kind == "prepare":
            return self.create_prepare(rigged, body.get("length_m"), body.get("noun"), turn=body.get("turn") is True)
        if kind == "teach":
            return self.create_teach(rigged)
        if kind == "film":
            return self.create_film(rigged, body.get("action"), body.get("end"), body.get("takes", 1),
                                    download=body.get("download") is True)
        return self.create_capture(rigged, str(body.get("take", "")), force=body.get("force") is True)

    def _check_free(self) -> None:
        if self.busy():
            raise RuntimeError("another Animate from video step is running; wait for it to finish")

    def _claim(self, job: TwocamJob) -> TwocamJob:
        self.jobs[job.id] = job
        self.active = job.id
        self.newest[job.creature.run] = job.id
        return job

    def create_prepare(self, rigged: Path, length_m: Any, noun: Any, turn: bool = False) -> TwocamJob:
        length, words = length_of(length_m), words_of(noun, "what it is", 60)
        c = Creature(rigged)
        yaw = None
        if turn is True:  # "It faces the wrong way": the turn it was given, plus half a turn
            done = read_json(c.file("prepared.json")) or {}
            if not isinstance(done.get("turned_deg"), (int, float)):
                raise ValueError("prepare the creature first")
            yaw = (float(done["turned_deg"]) + 180.0) % 360.0
        with self.lock:
            self._check_free()
            set_aside(c)
            c.folder.mkdir(parents=True, exist_ok=True)
            job = TwocamJob(uuid.uuid4().hex, "prepare", c, [prepare_command(c, length, words, yaw)],
                            Progress(PREPARE_SECONDS, "Turning, sizing and painting the rig"))
            return self._claim(job)

    def create_teach(self, rigged: Path) -> TwocamJob:
        c = Creature(rigged)
        if not c.prepared():
            raise ValueError("prepare the creature first")
        with self.lock:
            self._check_free()
            c.file("reader.json").unlink(missing_ok=True)   # taught again: not done until it is
            fetch = not dino_cached()                         # what the button said, checked the same way
            return self._claim(TwocamJob(uuid.uuid4().hex, "teach", c, teach_commands(c, fetch),
                                         TeachProgress(fetches_dino=fetch)))

    def create_film(self, rigged: Path, action: Any, end: Any, takes: Any, download: bool = False) -> TwocamJob:
        c = Creature(rigged)
        if not c.prepared():
            raise ValueError("prepare the creature first")
        said, ending, count = words_of(action, "what it does", 300), words_of(end, "how it ends", 200, False), takes_of(takes)
        if not engine_ready():
            raise ValueError("filming runs on stable-diffusion.cpp, which is not installed: run "
                             "python scripts/bootstrap_qwen_image.py --build-only (the engine, no weights), or "
                             "install Qwen-Image 2.1 on the Setup page")
        plan = video_plan()
        if not plan["ready"] and download is not True:
            raise ValueError(f"the video model is not downloaded yet ({plan['download_gb']} GB): "
                             "use Download the video model")
        with self.lock:
            self._check_free()
            folder = move_folder(c.clips, said, ending)
            folder.mkdir(parents=True, exist_ok=True)
            if not (folder / "move.json").is_file():
                write_json(folder / "move.json", {"action": said, "end": ending, "created": now()})
            job = TwocamJob(uuid.uuid4().hex, "film", c,
                            [film_command(c, folder, said, ending, count, next_seed(folder), download)],
                            FilmProgress(count), move=folder)
            return self._claim(job)

    def create_capture(self, rigged: Path, take: str, force: bool = False) -> TwocamJob:
        c = Creature(rigged)
        if not c.prepared():
            raise ValueError("prepare the creature first")
        if not c.file("reader.pt").is_file():
            raise ValueError("teach the reader first")
        clip = (self.output / take).resolve()
        if clip.parent.parent != c.clips.resolve() or not TAKE.fullmatch(clip.name) or not clip.is_file():
            raise ValueError(f"not a take of this creature: {take!r}")
        slug = clip.parent.name
        with self.lock:
            self._check_free()
            record = read_json(clip.with_suffix(".provenance.json"))
            job = TwocamJob(uuid.uuid4().hex, "capture", c,
                            [capture_command(c, clip, slug, force, record if isinstance(record, dict) else None)],
                            Progress(CAPTURE_SECONDS, "Reading the joints, fitting the rig"), move=clip.parent, take=clip)
            return self._claim(job)


TWOCAM_JOBS = TwocamJobManager()


def set_aside(c: Creature) -> Path | None:
    """Prepare again starts the creature over. If it had learned or filmed anything, that moves
    to twocam/earlier/<time>/ (the old reader and takes fit the old preparation only); nothing
    is deleted."""
    worth = [c.file("dataset"), c.file("reader.pt"), c.clips, c.captures]
    if not any(p.exists() for p in worth):
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest, n = c.root / "earlier" / stamp, 2
    while dest.exists():
        dest, n = c.root / "earlier" / f"{stamp}-{n}", n + 1
    dest.mkdir(parents=True)
    for part in (c.folder, c.clips, c.captures):
        if part.exists():
            part.rename(dest / part.name)
    return dest


def _tick(job: TwocamJob, stop: threading.Event) -> None:
    while not stop.wait(2.0):
        event = job.progress.tick()
        if event:
            job.emit(event)


def finish_prepare(job: TwocamJob, output: Path) -> dict[str, Any]:
    c = job.creature
    write_json(c.file("prepared.json"), job.outputs[-1])
    preview = c.file("preview.png")
    return {"message": "Prepared", "preview": rel(preview, output) if preview.is_file() else None}


def finish_teach(job: TwocamJob, output: Path) -> dict[str, Any]:
    dataset, train = job.outputs
    write_json(job.creature.file("reader.json"), {"taught_at": now(), "dataset": dataset, **train})
    return {"message": "The reader is taught", "held_out_error_px": train.get("held_out_error_px")}


def finish_film(job: TwocamJob, output: Path) -> dict[str, Any]:
    result = job.outputs[-1]
    for take in result.get("takes", []):
        write_json(Path(take["clip"]).with_suffix(".check.json"), {"seed": take.get("seed"), "check": take.get("check")})
    return {"message": f"Filmed {len(result.get('takes', []))} take(s)", "move": job.move.name if job.move else None}


def finish_capture(job: TwocamJob, output: Path) -> dict[str, Any]:
    """The GLB and its record go straight into the rig's folder, where the Library lists moves."""
    c, result = job.creature, job.outputs[-1]
    made = Path(result["apply"]["glb"])
    glb = c.run / made.name
    made.replace(glb)
    sidecar = made.with_suffix(".provenance.json")
    if sidecar.is_file():
        sidecar.replace(glb.with_suffix(".provenance.json"))
    slug = glb.stem.removeprefix(f"{c.stem}_")
    fit = result.get("fit") or {}
    warnings = (result.get("apply") or {}).get("warnings") or []
    write_json(c.captures / job.move.name / "capture.json",
               {"take": rel(job.take, output), "file": rel(glb, output), "created": now(),
                "verdict": fit.get("verdict"), "reproj_px": fit.get("reproj_px"), "warnings": warnings})
    return {"message": "Added the move", "clip": slug, "path": rel(glb, output), "warnings": warnings,
            "result_url": "/output/" + rel(glb, output), "size_bytes": glb.stat().st_size}


FINISHERS = {"prepare": finish_prepare, "teach": finish_teach, "film": finish_film, "capture": finish_capture}


def run_job(job: TwocamJob, manager: TwocamJobManager | None = None) -> None:
    manager = manager or TWOCAM_JOBS     # looked up now, so tests can swap the manager
    stop = threading.Event()
    c = job.creature
    try:
        if not job.cancel_requested:  # a cancel can land before this thread starts
            job.status = "running"
        job.emit({"phase": "queued", "overall_pct": 0, "message": STARTING[job.kind]})
        job.append_log(f"--- {job.kind}, {now()} ---")
        if job.progress.ticks:
            threading.Thread(target=_tick, args=(job, stop), daemon=True).start()
        code = 0
        for index, command in enumerate(job.commands):
            if job.cancel_requested:  # before each process: Teach's 12-minute dataset, or its training
                break
            if event := job.progress.begin(index):
                job.emit(event)
            summary = c.root / "steps" / f"{job.kind}-{index + 1}.json"
            summary.parent.mkdir(parents=True, exist_ok=True)
            # stdout is the script's JSON summary, kept whole in a file; stderr is its progress
            with summary.open("w", encoding="utf-8") as stdout:
                job.process = subprocess.Popen(
                    command, cwd=str(REPO), env=processes.child_env(), stdout=stdout, stderr=subprocess.PIPE,
                    bufsize=1, **processes.TEXT_PIPE, **processes.group_popen_kwargs())
                c.pid.write_text(processes.pid_record(job.process.pid), encoding="utf-8")
                assert job.process.stderr is not None
                for raw in job.process.stderr:
                    line = raw.rstrip("\n")
                    if not line.strip():
                        continue
                    job.append_log(line)
                    if event := job.progress.feed(line):
                        job.emit(event)
                code = job.process.wait()
            if code != 0 or job.cancel_requested:
                break
            job.outputs.append(summary_of(summary))
        stop.set()
        if job.cancel_requested:
            job.status = "cancelled"
            job.emit({"phase": "error", "message": "Cancelled"})
        elif code != 0:
            said = failure(list(job.log_lines)) or f"{job.kind} stopped with code {code}"
            bad = bad_clip_words(said) if job.kind == "capture" else None
            job.status = "error"
            job.emit({"phase": "error", "message": bad or said, "bad_clip": bad is not None,
                      "take": rel(job.take, manager.output) if job.take else None,
                      "log_tail": "\n".join(job.log_lines)[-6000:]})
        else:
            # the done event (with what was made) goes out BEFORE the status flips, as in Animate
            job.emit({"phase": "done", "overall_pct": 100, "kind": job.kind,
                      **FINISHERS[job.kind](job, manager.output)})
            job.status = "done"
            usage_api.count_done(USAGE[job.kind])
    except Exception as exc:
        job.status = "error"
        job.emit({"phase": "error", "message": str(exc)})
    finally:
        stop.set()
        if job.process is not None and job.process.poll() is None:
            try:
                processes.terminate_group(job.process.pid)
                job.process.wait(timeout=30)
            except Exception as cleanup_exc:
                job.append_log(f"cleanup warning: {cleanup_exc}")
        c.pid.unlink(missing_ok=True)
        manager.finish(job)


def cancel_job(job: TwocamJob) -> None:
    if job.status in TERMINAL:
        raise RuntimeError(f"job is already {job.status}")
    job.cancel_requested = True
    job.status = "cancelling"
    if job.process is not None and job.process.poll() is None:
        processes.terminate_group(job.process.pid)


def status_payload(job: TwocamJob) -> dict[str, Any]:
    return {"job_id": job.id, "status": job.status, "kind": job.kind,
            "log_tail": "\n".join(job.log_lines)[-4000:],
            "last_event": job.events[-1] if job.events else None}


# ---------------------------------------------------------------------------------- state


def takes_in(folder: Path, output: Path) -> list[dict[str, Any]]:
    found = []
    for clip in folder.glob("take*_s*.webm"):
        if m := TAKE.fullmatch(clip.name):
            said = read_json(clip.with_suffix(".check.json"))
            found.append({"clip": rel(clip, output), "seed": int(m[2]),
                          "check": said.get("check") if isinstance(said, dict) else None})
    found.sort(key=lambda t: t["seed"])
    for number, take in enumerate(found, 1):
        take["number"] = number
    return found


def creature_state(rigged: Path, manager: TwocamJobManager | None = None) -> dict[str, Any]:
    """Everything the page shows, read from disk (so a reopened page, or a restarted server,
    sees what is there), plus the newest job for this rig."""
    manager = manager or TWOCAM_JOBS
    output = manager.output
    c = Creature(rigged)
    prepared = None
    if c.prepared():
        creature = read_json(c.file("creature.json")) or {}
        summary = read_json(c.file("prepared.json")) or {}
        preview = c.file("preview.png")
        prepared = {"noun": creature.get("noun"), "length_m": (creature.get("prep") or {}).get("length_m"),
                    "preview": rel(preview, output) if preview.is_file() else None,
                    "updated": int(preview.stat().st_mtime) if preview.is_file() else 0,
                    "limbs": summary.get("limbs"),
                    "warning": "; ".join(summary.get("warnings") or ([summary["warning"]] if summary.get("warning") else []))
                    or None, "licence": summary.get("licence")}
    taught = read_json(c.file("reader.json"))
    reader = ("taught" if taught and c.file("reader.pt").is_file()
              else "unfinished" if c.file("reader.pt").is_file() else "none")
    moves = []
    for folder in (sorted(p for p in c.clips.iterdir() if p.is_dir()) if c.clips.is_dir() else []):
        said = read_json(folder / "move.json") or {}
        moves.append({"slug": folder.name, "action": said.get("action") or folder.name.replace("_", " "),
                      "end": said.get("end") or "", "created": said.get("created") or "",
                      "takes": takes_in(folder, output)})
    moves.sort(key=lambda m: m["created"], reverse=True)
    captured = []
    for record in (sorted(c.captures.glob("*/capture.json")) if c.captures.is_dir() else []):
        said = read_json(record)
        if isinstance(said, dict) and (output / str(said.get("file", ""))).is_file():
            captured.append({"move": record.parent.name, **said})
    plan = video_plan()
    job = manager.newest_for(rigged)
    return {
        "rig": rel(rigged, output), "stem": c.stem,
        "prepared": prepared,
        "reader": reader, "reader_info": taught if reader == "taught" else None,
        "poses": {"done": len(list(c.file("dataset").glob("*_meta.json"))) if c.file("dataset").is_dir() else 0,
                  "count": POSES},
        "dino": {**DINO, "cached": dino_cached()},
        "video_model": {k: plan.get(k) for k in ("ready", "engine", "license", "license_url", "license_note",
                                                 "download_gb", "missing")},
        "engine_ready": engine_ready(),
        "moves": moves, "captured": captured,
        "job": None if job is None else {**status_payload(job), "take": rel(job.take, output) if job.take else None,
                                         "move": job.move.name if job.move else None},
    }
