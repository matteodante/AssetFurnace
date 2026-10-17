#!/usr/bin/env python3
"""Run a material model on images, inside the materials environment (``vendor/materials/.venv``).

    <materials python> scripts/materials_worker.py job.json

The viewer's environment calls this through ``image_to_3dlab.delight.run_intrinsics``, which
holds the shared GPU lock around it. The job names a Marigold-IID snapshot folder, the images
and where each answer goes::

    {"task": "intrinsics", "model": "<snapshot folder>", "images": [{"path": ..., "out": ...npz}],
     "steps": null, "ensemble": 1, "resolution": 768, "seed": 0, "background": 0.5,
     "result": "<where to write the run record>"}

``segment_sam3`` scores every pixel against each phrase with SAM 3 or SAM 3.1 (once
``convert_sam31`` has turned SAM 3.1's checkpoint into transformers' files); ``regions_sam2`` gives SAM
2.1's automatic regions. ``yes_no`` and ``name_materials`` ask Qwen3.5-4B a ``"question"`` about each image, reasoning
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


def segment_sam3(job: dict) -> dict:
    """For each image and each phrase, how likely every pixel is to be that phrase (SAM 3).

    Each ``out`` gets ``scores``, P x H x W float16: the highest score of the instances that
    cover the pixel, 0 where none does.
    """
    import torch
    from transformers import Sam3Model, Sam3Processor

    started = time.perf_counter()
    processor = Sam3Processor.from_pretrained(job["model"], local_files_only=True)
    model = Sam3Model.from_pretrained(job["model"], local_files_only=True).to("cuda").eval()
    texts = {p: processor(text=p, return_tensors="pt").to("cuda") for p in job["phrases"]}
    for item in job["images"]:
        picture = composite(Image.open(item["path"]), float(job.get("background", 0.84)))
        inputs = processor(images=picture, return_tensors="pt").to("cuda")
        size = [(picture.height, picture.width)]
        scores = np.zeros((len(job["phrases"]), picture.height, picture.width), np.float32)
        with torch.inference_mode():
            vision = model.get_vision_features(pixel_values=inputs.pixel_values)
            for p, phrase in enumerate(job["phrases"]):
                outputs = model(vision_embeds=vision, **texts[phrase])
                found = processor.post_process_instance_segmentation(outputs, threshold=0.3, mask_threshold=0.5,
                                                                     target_sizes=size)[0]
                for mask, score in zip(found["masks"], found["scores"]):
                    scores[p] = np.maximum(scores[p], mask.float().cpu().numpy() * float(score))
        np.savez_compressed(item["out"], scores=scores.astype(np.float16))
    return {"images": len(job["images"]), "phrases": job["phrases"],
            "seconds": round(time.perf_counter() - started, 1), "torch": torch.__version__}


def regions_sam2(job: dict) -> dict:
    """SAM 2.1's automatic regions for each image: ``masks``, R x H x W, at most 64, largest first."""
    import torch
    from transformers import pipeline

    started = time.perf_counter()
    generator = pipeline("mask-generation", model=job["model"], device=0, dtype=torch.float32)
    for item in job["images"]:
        picture = composite(Image.open(item["path"]), float(job.get("background", 0.84)))
        found = generator(picture, points_per_batch=64, pred_iou_thresh=0.86, stability_score_thresh=0.9)
        masks = sorted((np.asarray(m, bool) for m in found["masks"]), key=lambda m: -int(m.sum()))[:64]
        stack = np.stack(masks) if masks else np.zeros((0, picture.height, picture.width), bool)
        np.savez_compressed(item["out"], masks=stack)
    return {"images": len(job["images"]), "seconds": round(time.perf_counter() - started, 1),
            "torch": torch.__version__}


# Keys SAM 3.1's checkpoint has and Sam3Model does not: the video tracker's rotary buffers and
# convolutions. The one level Sam3Model has and SAM 3.1 does not, the neck's fourth FPN level,
# is never read by the image detector (checked: zeroing it changes no score).
SAM31_UNUSED = ("vision_encoder.backbone.layers.", "backbone.vision_backbone.interactive_convs.",
                "backbone.vision_backbone.propagation_convs.", "geometry_encoder.points_")
SAM31_ABSENT = "vision_encoder.neck.fpn_layers.3."


def convert_sam31(job: dict) -> dict:
    """SAM 3.1's image detector, from Meta's checkpoint, saved as transformers' Sam3Model files."""
    import shutil

    import torch
    from transformers import Sam3Config, Sam3Model

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sam3_keys

    started = time.perf_counter()
    source, out = Path(job["model"]), Path(job["out"])
    checkpoint = torch.load(source / "sam3.1_multiplex.pt", map_location="cpu", weights_only=True)
    checkpoint = checkpoint.get("model", checkpoint)
    detector = {k.replace("detector.", "", 1): v for k, v in checkpoint.items() if k.startswith("detector.")}
    mapping = sam3_keys.convert_old_keys_to_new_keys(list(detector))
    state = {}
    for old, value in detector.items():
        key = mapping.get(old, old)
        if key == "vision_encoder.backbone.embeddings.position_embeddings":
            value = value[:, 1:, :]                         # no cls token in transformers' layout
        state[key] = value
    state = sam3_keys.split_qkv(state)
    state["text_encoder.text_projection.weight"] = state["text_encoder.text_projection.weight"].T
    model = Sam3Model(Sam3Config.from_pretrained(source, local_files_only=True))
    expected = model.state_dict()
    unused = sorted(k for k in state if k not in expected)
    usable = {k: v for k, v in state.items() if k in expected}
    missing, _ = model.load_state_dict(usable, strict=False)
    wrong_missing = [k for k in missing if not k.startswith(SAM31_ABSENT)]
    wrong_unused = [k for k in unused if not k.startswith(SAM31_UNUSED)]
    if wrong_missing or wrong_unused:
        raise RuntimeError(f"SAM 3.1 does not map onto Sam3Model: missing {wrong_missing[:5]}, "
                           f"unexpected {wrong_unused[:5]}")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    for name in ("processor_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "vocab.json", "merges.txt", "LICENSE"):
        shutil.copyfile(source / name, out / name)
    return {"loaded": len(usable), "absent": sorted(missing), "unused": len(unused),
            "seconds": round(time.perf_counter() - started, 1), "torch": torch.__version__}


TASKS = {"intrinsics": intrinsics, "yes_no": yes_no, "name_materials": name_materials,
         "segment_sam3": segment_sam3, "regions_sam2": regions_sam2, "convert_sam31": convert_sam31}


def main(argv: list[str]) -> int:
    job = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    record = TASKS[job["task"]](job)
    Path(job["result"]).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
