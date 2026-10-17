"""The models the game-material steps use: where they come from, how big, under which licence.

Pure data plus lookups on disk. `scripts/bootstrap_materials.py` downloads from here after
asking; everything else only *finds* files and says how to get them when they are missing.

The models run in their own environment, ``vendor/materials/.venv``: diffusers (Marigold)
still requires huggingface-hub below 2, which the viewer's environment is past. Revisions,
file lists and sizes were pinned on 2026-10-16 from the Hugging Face API, so a changed
upstream file cannot change what a step does without someone noticing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from image_to_3dlab.style.models import HF_HUB, HubModel

REPO = Path(__file__).resolve().parents[1]
BOOTSTRAP = "python scripts/bootstrap_materials.py"

# Marigold's weights carry the CreativeML Open RAIL++-M licence (the one of Stable Diffusion 2,
# which they are fine-tuned from). Outputs are free to use; the use restrictions are not optional.
OPENRAIL_URL = "https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL"
OPENRAIL_CONDITIONS = (
    ("Use restrictions (paragraph 5, Attachment A): not to break the law; not to exploit or harm "
     "minors; not to spread verifiably false information or harmful personal data; not to defame or "
     "harass; not for fully automated decisions on someone's legal rights; not to discriminate or "
     "exploit vulnerable groups; not for medical advice; not for justice, law enforcement, "
     "immigration or asylum processes."),
    ("Anyone given the weights or a model derived from them must be bound by the same restrictions "
     "and given a copy of the licence (paragraph 4)."),
    ("The licensor claims no rights in the outputs; you are accountable for them and their use "
     "(paragraph 6)."),
)

_MARIGOLD_FILES = (
    "model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors", "tokenizer/merges.txt", "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors", "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)

MARIGOLD_LIGHTING = HubModel(
    key="marigold-lighting", label="Marigold-IID Lighting v1.1 (albedo, shading, residual)",
    repo="prs-eth/marigold-iid-lighting-v1-1", revision="08c3930bb641abf786ba44ce92547507ebefbc16",
    files=_MARIGOLD_FILES, size=2_581_766_286, license="CreativeML Open RAIL++-M",
    license_url=OPENRAIL_URL,
    note="fp16 files only. Released 2024-12-16, updated 2025-05-20; code Apache-2.0 (prs-eth/Marigold).")

MARIGOLD_APPEARANCE = HubModel(
    key="marigold-appearance", label="Marigold-IID Appearance v1.1 (albedo, roughness, metallic)",
    repo="prs-eth/marigold-iid-appearance-v1-1", revision="e7280a0a0fc5a0df0b36050882b3d8b77da22fd9",
    files=_MARIGOLD_FILES, size=2_581_720_163, license="CreativeML Open RAIL++-M",
    license_url=OPENRAIL_URL,
    note="fp16 files only. Released 2024-10-21, updated 2025-05-20; code Apache-2.0 (prs-eth/Marigold).")

QWEN35 = HubModel(
    key="qwen35", label="Qwen3.5-4B (names the materials it sees)",
    repo="Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    files=("LICENSE", "chat_template.jinja", "config.json", "merges.txt",
           "model.safetensors-00001-of-00002.safetensors", "model.safetensors-00002-of-00002.safetensors",
           "model.safetensors.index.json", "preprocessor_config.json", "tokenizer.json",
           "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json"),
    size=9_342_828_238, license="Apache-2.0", license_url="https://huggingface.co/Qwen/Qwen3.5-4B",
    note="bf16, reads images itself. Released 2026-02-27.")

SAM3 = HubModel(
    key="sam3", label="SAM 3 (segments every region of a named material)",
    repo="facebook/sam3", revision="3c879f39826c281e95690f02c7821c4de09afae7",
    files=("LICENSE", "config.json", "merges.txt", "model.safetensors", "processor_config.json",
           "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"),
    size=3_445_003_826, license="SAM License (custom; Meta may change it)",
    license_url="https://huggingface.co/facebook/sam3/blob/main/LICENSE",
    note="Gated: the Hugging Face account must be granted access first, through a form only its "
         "owner fills in. The transformers files only, not sam3.pt. Released 2025-11-20. The licence "
         "forbids uses under ITAR or barred by trade controls (military, nuclear, espionage, developing "
         "or using guns or illegal weapons; section 1.b.v) and Meta may change it (section 8).")

SAM21 = HubModel(
    key="sam21", label="SAM 2.1 large (automatic regions, the zones' fallback)",
    repo="facebook/sam2.1-hiera-large", revision="665f8e2ad61cf5f53d65644ff27c8ee525124610",
    files=("config.json", "model.safetensors", "preprocessor_config.json", "processor_config.json",
           "video_preprocessor_config.json"),
    size=897_904_604, license="Apache-2.0", license_url="https://huggingface.co/facebook/sam2.1-hiera-large",
    note="The transformers files only, not the .pt. Released 2024-09-24; revision of 2025-08-15.")

SAM31 = HubModel(
    key="sam31", label="SAM 3.1 (segments every region of a named material; the zones' default)",
    repo="facebook/sam3.1", revision="daa63191845a41281374e725f4c9e51c7a824460",
    files=("LICENSE", "config.json", "merges.txt", "processor_config.json", "special_tokens_map.json",
           "tokenizer.json", "tokenizer_config.json", "vocab.json", "sam3.1_multiplex.pt"),
    size=3_502_755_717 + 7352 + 25843 + 524619 + 1712 + 588 + 3642073 + 799 + 862328,
    license="SAM License (custom; Meta may change it)",
    license_url="https://huggingface.co/facebook/sam3.1/blob/main/LICENSE",
    note="Gated like SAM 3 (one form grants the whole SAM 3 collection). Released 2026-03-26. Meta's own "
         "format only: setup converts its image detector into transformers' Sam3Model files "
         "(vendor/materials/sam3.1-transformers), the checkpoint loaded with weights_only=True. "
         "Same licence text as SAM 3: no ITAR or trade-controlled end uses (1.b.v); Meta may change it (8).")

MODELS = {m.key: m for m in (MARIGOLD_LIGHTING, MARIGOLD_APPEARANCE, QWEN35, SAM3, SAM21, SAM31)}
GATED = {SAM3.key, SAM31.key}


def model_folder(model: HubModel, hub: Path | None = None) -> Path:
    """The snapshot folder a pipeline loads from (``from_pretrained`` takes a folder)."""
    return model.cache_dir(hub or HF_HUB) / "snapshots" / model.revision


def missing(model: HubModel, hub: Path | None = None) -> list[str]:
    return [name for name in model.files if model.path(name, hub or HF_HUB) is None]


class MissingModel(RuntimeError):
    """A step's model or environment is not installed; the message says how to get it."""


def require(model: HubModel, hub: Path | None = None) -> Path:
    """The model's snapshot folder at its pinned revision, or MissingModel naming the command."""
    absent = missing(model, hub)
    if absent:
        raise MissingModel(f"{model.label} is not downloaded ({len(absent)} files missing); "
                           f"run: {BOOTSTRAP} --only {model.key}")
    return model_folder(model, hub)


# --------------------------------------------------------------------------- the environment

# Pinned together: diffusers 0.40 is the newest that installs with this transformers, and it
# brings huggingface-hub 1.33 into this environment only. torch comes from PyTorch's own index.
ENV_PACKAGES = ("diffusers==0.40.0", "transformers==5.20.0", "numpy<2", "pillow", "safetensors")
TORCH = ("torch==2.14.1", "torchvision==0.29.1")      # torchvision: Qwen3.5's image processor
PYTHON_VERSION = "3.11"


def env_dir(repo: Path = REPO) -> Path:
    """Where the environment lives: ``I2L_MATERIALS_ENV``, else ``vendor/materials/.venv``."""
    override = os.environ.get("I2L_MATERIALS_ENV")
    return Path(override) if override else repo / "vendor" / "materials" / ".venv"


def env_python(repo: Path = REPO) -> Path:
    folder = env_dir(repo)
    return folder / "Scripts" / "python.exe" if os.name == "nt" else folder / "bin" / "python"


def sam31_folder(repo: Path = REPO) -> Path:
    """SAM 3.1's image detector as transformers' Sam3Model files, made by the setup from Meta's checkpoint."""
    return env_dir(repo).parent / "sam3.1-transformers"


def require_sam31(repo: Path = REPO) -> Path:
    folder = sam31_folder(repo)
    if not (folder / "model.safetensors").is_file():
        raise MissingModel(f"SAM 3.1 is not set up ({folder}); run: {BOOTSTRAP} --only env,sam31")
    return folder


def env_marker(repo: Path = REPO) -> Path:
    """Written after every package is installed: an interrupted setup leaves python.exe without it."""
    return env_dir(repo) / ".i2l-materials-complete"


def env_contents() -> str:
    return json.dumps({"python": PYTHON_VERSION, "packages": [*TORCH, *ENV_PACKAGES]}, sort_keys=True)


def env_ready(repo: Path = REPO) -> bool:
    """The environment exists and holds exactly the packages pinned here."""
    marker = env_marker(repo)
    return env_python(repo).is_file() and marker.is_file() and marker.read_text(encoding="utf-8") == env_contents()


def require_env(repo: Path = REPO) -> Path:
    python = env_python(repo)
    if not env_ready(repo):
        raise MissingModel(f"the materials environment is not set up, or its setup was interrupted "
                           f"({env_dir(repo)}); "
                           f"run: {BOOTSTRAP} --only env")
    return python
