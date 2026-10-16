#!/usr/bin/env python3
"""Run a material model on images, inside the materials environment (``vendor/materials/.venv``).

    <materials python> scripts/materials_worker.py job.json

The viewer's environment calls this through ``image_to_3dlab.delight.run_intrinsics``, which
holds the shared GPU lock around it. The job names a Marigold-IID snapshot folder, the images
and where each answer goes::

    {"task": "intrinsics", "model": "<snapshot folder>", "images": [{"path": ..., "out": ...npz}],
     "steps": null, "ensemble": 1, "resolution": 768, "seed": 0, "background": 0.5,
     "result": "<where to write the run record>"}

``yes_no`` and ``name_materials`` ask Qwen3.5-4B a ``"question"`` about each image, reasoning
switched off: the first gives P(yes) from the first token's logits, the second the short free
answer. For ``intrinsics``, each ``out`` gets one float16 HxWx3 array per target the model predicts (``albedo``,
``shading``, ``residual`` for Lighting), in [0, 1], in the space the model declares. A
transparent image is laid on a flat grey first: the model was trained on full pictures.
Nothing here downloads: the snapshot must already be on disk.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def composite(image: Image.Image, background: float) -> Image.Image:
    """The picture on a flat grey where it is transparent, as RGB."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + background * (1.0 - alpha)
    return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8))


def intrinsics(job: dict) -> dict:
    import diffusers
    import torch
    from diffusers import MarigoldIntrinsicsPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    started = time.perf_counter()
    pipe = MarigoldIntrinsicsPipeline.from_pretrained(job["model"], variant="fp16", torch_dtype=dtype,
                                                      local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    names = list(pipe.target_properties["target_names"])
    for item in job["images"]:
        picture = composite(Image.open(item["path"]), float(job.get("background", 0.5)))
        generator = torch.Generator(device=device).manual_seed(int(job.get("seed", 0)))
        out = pipe(picture, num_inference_steps=job.get("steps"), ensemble_size=int(job.get("ensemble", 1)),
                   processing_resolution=int(job.get("resolution", 768)), match_input_resolution=True,
                   generator=generator, output_type="np")
        prediction = np.asarray(out.prediction)                      # (targets, H, W, 3)
        np.savez_compressed(item["out"], **{name: prediction[i].astype(np.float16)
                                            for i, name in enumerate(names)})
    return {"target_properties": pipe.target_properties, "device": device,
            "dtype": str(dtype).removeprefix("torch."), "seconds": round(time.perf_counter() - started, 1),
            "images": len(job["images"]), "diffusers": diffusers.__version__, "torch": torch.__version__,
            "steps": job.get("steps") or pipe.default_denoising_steps,
            "resolution": int(job.get("resolution", 768)), "ensemble": int(job.get("ensemble", 1))}


def _qwen(folder: str):
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(folder, local_files_only=True,
                                                            dtype=torch.bfloat16).to("cuda").eval()
    return processor, model


def _ask(processor, model, image: Image.Image, question: str):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                         return_tensors="pt", enable_thinking=False).to(model.device)


def yes_no(job: dict) -> dict:
    """P(yes) for each image, from the first answer token's logits over "Yes" and "No"."""
    import torch

    started = time.perf_counter()
    processor, model = _qwen(job["model"])
    yes = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no = processor.tokenizer.encode("No", add_special_tokens=False)[0]
    answers = []
    for item in job["images"]:
        picture = composite(Image.open(item["path"]), float(job.get("background", 0.84)))
        inputs = _ask(processor, model, picture, job["question"] + " Answer Yes or No.")
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1, [yes, no]].float()
        answers.append(float(torch.softmax(logits, dim=0)[0]))
    return {"answers": answers, "seconds": round(time.perf_counter() - started, 1), "images": len(answers),
            "torch": torch.__version__}


def name_materials(job: dict) -> dict:
    """The short free answer to `question` for each image (parsed by the caller)."""
    import torch

    started = time.perf_counter()
    processor, model = _qwen(job["model"])
    answers = []
    for item in job["images"]:
        picture = composite(Image.open(item["path"]), float(job.get("background", 0.84)))
        inputs = _ask(processor, model, picture, job["question"])
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        answers.append(processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
    return {"answers": answers, "seconds": round(time.perf_counter() - started, 1), "images": len(answers),
            "torch": torch.__version__}


TASKS = {"intrinsics": intrinsics, "yes_no": yes_no, "name_materials": name_materials}


def main(argv: list[str]) -> int:
    job = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    record = TASKS[job["task"]](job)
    Path(job["result"]).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
