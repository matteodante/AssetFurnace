#!/usr/bin/env python3
"""Fair A/B of two Qwen-Image 2.1 recipes through the same sd-cli. Judged by your eyes only.

Recipe U (upstream 0.5.2): base checkpoint qwen_image_2.1-Q8_0.gguf, --steps 8, cfg 1.0, euler.
Recipe O (ours):           Turbo checkpoint qwen-image-2.1-turbo-Q8_0.gguf,
                           --sigmas qwen-image-2.1-turbo (8 steps), cfg 1.0, euler.
Optional reference R:      base checkpoint at its card recipe, --steps 40, cfg 6.0 (slow).

Same sd-cli binary, same text encoder, same VAE, same prompts, same seeds, same size, same
flags otherwise (--diffusion-fa -v, no --offload-to-cpu: what both repos pass). No prompt
rewriting, no transparency template: both get the prompt as written.

It NEVER downloads anything. If a file is missing it names it, its size and where it comes
from, and stops. The base checkpoint is not installed by our setup (we fetch the Turbo), so
you will likely need to fetch it yourself first (leejet/Qwen-Image-2.1-GGUF,
qwen_image_2.1-Q8_0.gguf, about 7.7 GB, Qwen Research License).

Each picture takes the shared GPU lock (%TEMP%/assetfurnace-gpu.lock) only while it runs.
The recipe order alternates per picture so neither always gets the warm disk cache.

Output in --out:
  pictures/            every PNG, named <prompt#>_<seed>_<recipe>.png
  runs.csv             wall time and sd-cli's own "generate_image completed in" time per picture
  sheet_blind.jpg      one row per (prompt, seed); columns shuffled, labelled 1/2(/3) only
  key.json             which column is which recipe, per row -- do not open before judging
  votes.csv            fill in "pick" per row: 1, 2, 3 or "same"
Then:  python qwen_ab.py tally --out <dir>    counts your picks per recipe and shows times.

Usage:
  python qwen_ab.py run --repo C:/path/AssetFurnace --base-model <qwen_image_2.1-Q8_0.gguf> \
      [--turbo-model <...turbo-Q8_0.gguf>] [--size 1024] [--seeds 1 2 3] [--with-card-recipe]
  python qwen_ab.py tally --out ab_out
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

PROMPTS = [
    ("a stylised 3D game character render of a dwarf blacksmith in leather apron, full body, A-pose, "
     "soft even lighting, plain light grey background"),
    ("a wooden treasure chest with iron bands and brass studs, game prop, three-quarter view, "
     "soft even studio lighting, plain white background"),
    "a fantasy knight's helmet with a plume, hand-painted game asset style, plain white background",
    "a small green goblin with big ears holding a lantern, full body, stylised 3D render, plain grey background",
    "a sci-fi crate with glowing blue panels and scratched metal, game prop, plain white background",
    "a grey wolf standing side-on, realistic fur, full body, plain light background",
]
COMPLETED = re.compile(r"generate_image completed in ([\d.]+)s")
HF_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
NEEDED = {
    "turbo": ("models--unsloth--Qwen-Image-2.1-Turbo-GGUF", "qwen-image-2.1-turbo-Q8_0.gguf", "7.6 GB"),
    "base": ("models--leejet--Qwen-Image-2.1-GGUF", "qwen_image_2.1-Q8_0.gguf", "7.7 GB"),
    "llm": ("models--Qwen--Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf", "5.0 GB"),
    "vae": ("models--Comfy-Org--Qwen-Image-2.1", "qwen_image_2.1_vae_bf16.safetensors", "0.7 GB"),
}


def find(key: str, explicit: Path | None) -> Path:
    if explicit:
        if explicit.is_file():
            return explicit
        sys.exit(f"not a file: {explicit}")
    directory, name, size = NEEDED[key]
    hits = sorted((HF_HUB / directory).glob(f"snapshots/*/**/{name}"))
    if hits:
        return hits[0]
    sys.exit(f"missing {name} ({size}, Hugging Face repo {directory.split('--', 1)[1].replace('--', '/')}).\n"
             f"This script does not download. Fetch it yourself if you want this test, then rerun "
             f"(or pass its path with --{key}-model / --{key}).")


def recipes(paths: dict[str, Path], card: bool) -> dict[str, dict]:
    out = {
        "U_base_8step": {"model": paths["base"], "extra": ["--steps", "8", "--cfg-scale", "1.0"]},
        "O_turbo_sigmas": {"model": paths["turbo"],
                           "extra": ["--sigmas", "qwen-image-2.1-turbo", "--cfg-scale", "1.0"]},
    }
    if card:
        out["R_base_40step_cfg6"] = {"model": paths["base"], "extra": ["--steps", "40", "--cfg-scale", "6.0"]}
    return out


def command(binary: Path, recipe: dict, paths: dict, prompt: str, seed: int, size: int, out: Path) -> list[str]:
    return [str(binary), "--diffusion-model", str(recipe["model"]), "--vae", str(paths["vae"]),
            "--llm", str(paths["llm"]), "-p", prompt, "--sampling-method", "euler", *recipe["extra"],
            "-W", str(size), "-H", str(size), "--seed", str(seed), "--diffusion-fa", "-v", "-o", str(out)]


def sdcli_version(binary: Path) -> str:
    try:
        text = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=30, check=False)
        return (text.stdout + text.stderr).strip().splitlines()[0][:120]
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({exc})"


def check_turbo_preset(binary: Path) -> None:
    """Seconds, not minutes: refuse an sd-cli without the Turbo sigma preset (needs master-959+)."""
    text = subprocess.run([str(binary), "--help"], capture_output=True, text=True, timeout=30, check=False)
    if "--sigmas" not in (text.stdout + text.stderr):
        sys.exit("this sd-cli has no --sigmas option; the Turbo recipe needs stable-diffusion.cpp master-959 or newer")


def run(args) -> None:
    sys.path.insert(0, str(args.repo))
    from image_to_3dlab.gpu_lock import gpu  # the repo's shared-GPU lock

    binary = args.sdcli or next((p for p in (args.repo / "vendor/sdcpp/sd-cli.exe", args.repo / "vendor/sdcpp/sd-cli")
                                 if p.is_file()), None)
    if not binary or not Path(binary).is_file():
        sys.exit("no sd-cli found; pass --sdcli")
    binary = Path(binary)
    check_turbo_preset(binary)
    paths = {"turbo": find("turbo", args.turbo_model), "base": find("base", args.base_model),
             "llm": find("llm", args.llm), "vae": find("vae", args.vae)}
    plan = recipes(paths, args.with_card_recipe)
    prompts = [line.strip() for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if args.prompts else PROMPTS
    out = args.out
    (out / "pictures").mkdir(parents=True, exist_ok=True)
    meta = {"sd_cli": str(binary), "sd_cli_version": sdcli_version(binary), "size": args.size,
            "seeds": args.seeds, "prompts": prompts, "recipes": {k: [str(v["model"].name), *v["extra"]]
                                                                for k, v in plan.items()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    rows, turn = [], 0
    names = list(plan)
    for p_index, prompt in enumerate(prompts):
        for seed in args.seeds:
            order = names[turn % len(names):] + names[:turn % len(names)]
            turn += 1
            for name in order:
                png = out / "pictures" / f"p{p_index:02d}_s{seed}_{name}.png"
                if png.exists():
                    continue
                cmd = command(binary, plan[name], paths, prompt, seed, args.size, png)
                start = time.time()
                with gpu("qwen-ab"):
                    done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
                wall = time.time() - start
                found = COMPLETED.findall(done.stdout + done.stderr)
                rows.append({"prompt": p_index, "seed": seed, "recipe": name, "wall_s": round(wall, 1),
                             "sdcli_generate_s": found[-1] if found else "", "ok": done.returncode == 0
                             and png.exists()})
                print(rows[-1], flush=True)
                if not rows[-1]["ok"]:
                    (out / f"{png.stem}.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    with open(out / "runs.csv", "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["prompt", "seed", "recipe", "wall_s", "sdcli_generate_s", "ok"])
        if fh.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
    sheet(out, names, prompts, args.seeds, args.size)


def sheet(out: Path, names: list[str], prompts: list[str], seeds: list[int], size: int) -> None:
    """Blind side-by-side sheet: columns shuffled per row, labelled by number only. No scoring."""
    from PIL import Image, ImageDraw

    cell = 512  # shown at the same display size; open pictures/ for full resolution
    rng = random.Random(20261025)
    rows = [(p, s) for p in range(len(prompts)) for s in seeds]
    canvas = Image.new("RGB", (cell * len(names) + 60, cell * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    key = {}
    for r, (p, s) in enumerate(rows):
        shuffled = names[:]
        rng.shuffle(shuffled)
        key[f"p{p:02d}_s{s}"] = {str(i + 1): name for i, name in enumerate(shuffled)}
        draw.text((5, r * cell + 5), f"row {r + 1}\np{p:02d}\ns{s}", fill="black")
        for c, name in enumerate(shuffled):
            png = out / "pictures" / f"p{p:02d}_s{s}_{name}.png"
            if png.exists():
                image = Image.open(png).convert("RGB").resize((cell, cell), Image.LANCZOS)
                canvas.paste(image, (60 + c * cell, r * cell))
            draw.text((60 + c * cell + 6, r * cell + 6), str(c + 1), fill="red")
    canvas.save(out / "sheet_blind.jpg", quality=92)
    (out / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    votes = out / "votes.csv"
    if not votes.exists():
        with open(votes, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["row", "pick (1/2/3/same)", "note"])
            for r, (p, s) in enumerate(rows):
                writer.writerow([f"p{p:02d}_s{s}", "", ""])
    print(f"\nJudge {out / 'sheet_blind.jpg'} (and pictures/ at full size), fill {votes}, then run tally.")


def tally(args) -> None:
    key = json.loads((args.out / "key.json").read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    with open(args.out / "votes.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pick = (row.get("pick (1/2/3/same)") or "").strip()
            name = key[row["row"]].get(pick, pick or "blank")
            counts[name] = counts.get(name, 0) + 1
    print("picks:", counts)
    times: dict[str, list[float]] = {}
    with open(args.out / "runs.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            times.setdefault(row["recipe"], []).append(float(row["wall_s"]))
    for name, values in times.items():
        values.sort()
        print(f"{name}: median wall {values[len(values) // 2]:.1f} s over {len(values)} pictures "
              f"(includes model loading)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo", type=Path, default=Path.cwd())
    r.add_argument("--sdcli", type=Path)
    r.add_argument("--base-model", type=Path)
    r.add_argument("--turbo-model", type=Path)
    r.add_argument("--llm", type=Path)
    r.add_argument("--vae", type=Path)
    r.add_argument("--prompts", type=Path, help="one prompt per line (default: 6 asset prompts)")
    r.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    r.add_argument("--size", type=int, default=1024, help="same for both; 768 is upstream's default, "
                   "1024 ours for references")
    r.add_argument("--with-card-recipe", action="store_true", help="also base at 40 steps, cfg 6 (slow)")
    r.add_argument("--out", type=Path, default=Path("qwen_ab_out"))
    t = sub.add_parser("tally")
    t.add_argument("--out", type=Path, default=Path("qwen_ab_out"))
    args = parser.parse_args()
    run(args) if args.cmd == "run" else tally(args)


if __name__ == "__main__":
    main()
