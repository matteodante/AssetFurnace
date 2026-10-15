"""The models the style judges use: where they come from, how big they are, their licences.

Pure data plus lookups on disk. `scripts/bootstrap_style.py` downloads from here after
asking; everything else only *finds* files and says how to get them when they are not
there. Nothing in this module downloads.

Revisions and checksums were pinned on 2026-10-15, so a changed upstream file cannot
change what a judge measures without someone noticing.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HF_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
BOOTSTRAP = "python scripts/bootstrap_style.py"


@dataclass(frozen=True)
class HubModel:
    key: str
    label: str
    repo: str
    revision: str
    files: tuple[str, ...]
    size: int                 # bytes of all `files`
    license: str
    license_url: str
    note: str = ""

    def cache_dir(self, hub: Path | None = None) -> Path:
        return (hub or HF_HUB) / f"models--{self.repo.replace('/', '--')}"

    def path(self, name: str, hub: Path | None = None) -> Path | None:
        """The cached file at the pinned revision, or None when it is not downloaded."""
        candidate = self.cache_dir(hub) / "snapshots" / self.revision / name
        return candidate if candidate.is_file() else None


SIGLIP2 = HubModel(
    key="siglip2", label="SigLIP 2 base/16 224 (general image embedding)",
    repo="google/siglip2-base-patch16-224", revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    files=("config.json", "preprocessor_config.json", "model.safetensors"),
    size=1_500_800_904 + 253 + 394, license="Apache-2.0",
    license_url="https://huggingface.co/google/siglip2-base-patch16-224",
    note="One file with the image and text towers; only the image tower is used.")

CSD = HubModel(
    key="csd", label="CSD ViT-L (style embedding)",
    repo="tomg-group-umd/CSD-ViT-L", revision="5bc26a6fb0487f3f00a2a7313135103a005b1b67",
    files=("config.json", "pytorch_model.bin"),
    size=2_438_228_893 + 34, license="CC-BY-4.0 (from OpenAI CLIP ViT-L/14, MIT)",
    license_url="https://huggingface.co/tomg-group-umd/CSD-ViT-L",
    note="A pickle file: loaded with torch.load(weights_only=True), which runs no code from it. "
         "Attribution: Somepalli et al., 'Measuring Style Similarity in Diffusion Models', 2024.")

QWEN3VL = HubModel(
    key="qwen3vl", label="Qwen3-VL-8B-Instruct Q4_K_M (judge; Qwen-Image's text encoder)",
    repo="Qwen/Qwen3-VL-8B-Instruct-GGUF", revision="f982a07559d4a2f6c8744d840bf6fccab30eea96",
    files=("Qwen3VL-8B-Instruct-Q4_K_M.gguf",), size=5_027_784_800, license="Apache-2.0",
    license_url="https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF",
    note="Already there when Qwen-Image is set up.")

# The vision half of Qwen3-VL. Qwen-Image's setup owns it (its edit route needs it too);
# the judge only looks for it, in whatever snapshot it landed.
MMPROJ = "mmproj-Qwen3VL-8B-Instruct-F16.gguf"

HUB_MODELS = {m.key: m for m in (SIGLIP2, CSD)}


@dataclass(frozen=True)
class LlamaBuild:
    archive: str
    sha256: str
    size: int


LLAMA_TAG = "b11669"
LLAMA_RELEASES = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/"
LLAMA_LICENSE = "MIT"
# Vulkan where there is one: 30 MB against 550 MB for CUDA with its runtime, and the
# judge answers a handful of yes/no questions, not a long generation.
LLAMA_BUILDS = {
    ("windows", "amd64"): LlamaBuild(
        f"llama-{LLAMA_TAG}-bin-win-vulkan-x64.zip",
        "2c07574e7bde71321cb785282b20cc821e951132149eeceeea32ddebdbf0598e", 32_182_350),
    ("linux", "x86_64"): LlamaBuild(
        f"llama-{LLAMA_TAG}-bin-ubuntu-vulkan-x64.tar.gz",
        "e8709e5ba6c1b093dea7f3de57f4aba842f9a188fba6b465782d65d6fcfa4d9f", 30_497_889),
    ("darwin", "arm64"): LlamaBuild(
        f"llama-{LLAMA_TAG}-bin-macos-arm64.tar.gz",
        "dbed55493d72bc1ba2babc8770d669c2289b5bc4ee9fb2a4dfc158d45b7d13fd", 12_191_582),
}


def llama_build(system: str | None = None, machine: str | None = None) -> LlamaBuild | None:
    return LLAMA_BUILDS.get(((system or platform.system()).lower(), (machine or platform.machine()).lower()))


def llama_dir(repo: Path = REPO) -> Path:
    return repo / "vendor" / "llama-cpp"


def llama_server(repo: Path = REPO) -> Path | None:
    """The llama-server binary, wherever the archive put it under vendor/llama-cpp/."""
    root = llama_dir(repo)
    if not root.is_dir():
        return None
    for name in ("llama-server.exe", "llama-server"):
        found = sorted(root.rglob(name))
        if found:
            return found[0]
    return None


def mmproj_path(hub: Path | None = None) -> Path | None:
    found = sorted((QWEN3VL.cache_dir(hub) / "snapshots").glob(f"*/{MMPROJ}"))
    return found[0] if found else None


class MissingModel(RuntimeError):
    """A judge's files are not on disk; the message says how to get them."""


def require(model: HubModel, hub: Path | None = None) -> dict[str, Path]:
    """Every file of `model` at its pinned revision, or MissingModel naming the command."""
    paths = {name: model.path(name, hub) for name in model.files}
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        raise MissingModel(f"{model.label} is not downloaded ({', '.join(missing)}); "
                           f"run: {BOOTSTRAP} --only {model.key}")
    return paths  # type: ignore[return-value]
