"""A text-to-image command of sd-cli turned into an edit of Qwen-Image 2.1.

Two things change: `--llm_vision` gives the encoder its vision half (the mmproj file, in
the same cache folder as the encoder), and each reference picture goes in with `-r`.
Everything else (weights, sampler, size, seed) stays whatever the caller built, so an
edit follows the text-to-image recipe of the day.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

VISION_REPO = "models--Qwen--Qwen3-VL-8B-Instruct-GGUF"
VISION_FILE = "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
# Two shape pictures (front and the depth of another side) plus up to three style
# pictures from a style bible: the most anything here sends.
MAX_REFERENCES = 5


def find_vision(hub: Path) -> Path | None:
    """The vision file in a Hugging Face cache, whatever snapshot it landed under."""
    matches = sorted((Path(hub) / VISION_REPO).glob(f"snapshots/*/{VISION_FILE}"))
    return matches[0] if matches else None


def with_references(command: list[str], references: list[Path], vision: Path) -> list[str]:
    """`command` with the vision file and the pictures, in order, right after the binary."""
    return command[:1] + edit_args(references, vision) + command[1:]


def edit_args(references: list[Path], vision: Path) -> list[str]:
    """The vision file and one `-r` per picture, checked, in the order given."""
    if not references:
        raise ValueError("an edit needs at least one reference picture")
    if len(references) > MAX_REFERENCES:
        raise ValueError(f"at most {MAX_REFERENCES} reference pictures, got {len(references)}")
    extra = ["--llm_vision", str(vision)]
    for reference in references:
        path = Path(reference)
        if not path.is_file():
            raise FileNotFoundError(f"reference picture not found: {path.name}")
        try:
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError(f"{path.name} is not a picture sd-cli can read") from error
        extra += ["-r", str(path.resolve())]
    return extra
