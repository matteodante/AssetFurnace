"""Estimate a picture's camera with MoGe-2 and print its focal length (runs in vendor/moge/.venv).

    vendor/moge/.venv/Scripts/python scripts/moge_worker.py picture.png model.pt

Prints one JSON line, {"fx_normalized": ...}: the focal length as a share of the picture's
width, as the official Pixal3D recipe reads it (`intrinsics[0, 0]`). The caller,
image_to_3dlab/moge_fov.py, turns it into the field of view.
"""

from __future__ import annotations

import json
import sys


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    import torch
    from moge.model.v2 import MoGeModel
    from PIL import Image

    image_path, weights = (argv or sys.argv[1:])[:2]
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = MoGeModel.from_pretrained(weights).to(device).eval()
    # the caller (moge_fov.preprocessed) already cropped it and put it on black, as the recipe does
    picture = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    with torch.no_grad():
        # half precision only on CUDA: on the CPU it is what made one estimate take 8 minutes
        output = model.infer(torch.from_numpy(picture).permute(2, 0, 1).to(device), use_fp16=device == "cuda")
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    print(json.dumps({"fx_normalized": float(intrinsics[0, 0])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
