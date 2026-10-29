"""The joint reader: find every joint in both halves of a two-camera video.

Ported from AnyMove's `mocap/model.py` and `mocap/detect.py` (Patch, Apache-2.0). The
network is DINOv2 ViT-B/14 with registers (Meta, Apache-2.0) plus a small upsampling
head that draws one heatmap per joint. Each view is padded to a square with its own
backdrop colour and resized to 448 x 448; heatmaps come out at 128 x 128.

A trained reader file holds the whole network, backbone included, so reading builds the
backbone from its pinned configuration and downloads nothing. Only training starts from
Meta's pretrained weights (346 MB), and that download is the caller's to confirm.

Needs OpenCV and `transformers`, imported here only.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .layout import Layout

BACKBONE_ID = "facebook/dinov2-with-registers-base"
BACKBONE_LICENSE = "Apache-2.0"
BACKBONE_DOWNLOAD_MB = 346
# The backbone's config.json on the Hugging Face hub (checked 2026-10-29). transformers'
# own defaults differ (patch 16, image 224), so a reader built from them would load
# without error and read nonsense.
BACKBONE_CONFIG = {
    "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "mlp_ratio": 4,
    "hidden_act": "gelu", "layer_norm_eps": 1e-6, "layerscale_value": 1.0, "patch_size": 14,
    "image_size": 518, "num_channels": 3, "num_register_tokens": 4, "qkv_bias": True,
    "use_swiglu_ffn": False, "drop_path_rate": 0.0, "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0, "interpolate_antialias": True, "interpolate_offset": 0.0,
    "apply_layernorm": True,
}
SIZE, HM = 448, 128
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ReaderError(RuntimeError):
    pass


def _transformers():
    try:
        import transformers
    except ImportError as exc:
        raise ReaderError("reading a video needs transformers: uv pip install transformers") from exc
    return transformers


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ReaderError("reading a video needs OpenCV: uv pip install opencv-python-headless") from exc
    return cv2


def build_backbone(pretrained: bool = False):
    """DINOv2 with registers. `pretrained=True` downloads Meta's weights (346 MB) unless
    they are cached: only training needs that, and its caller must ask first."""
    tf = _transformers()
    if pretrained:
        return tf.AutoModel.from_pretrained(BACKBONE_ID)
    return tf.Dinov2WithRegistersModel(tf.Dinov2WithRegistersConfig(**BACKBONE_CONFIG))


class JointReader(nn.Module):
    def __init__(self, n_joints: int, pretrained: bool = False):
        super().__init__()
        self.backbone = build_backbone(pretrained)
        self.n_reg = getattr(self.backbone.config, "num_register_tokens", 0)
        c = self.backbone.config.hidden_size

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(16, o), nn.GELU())

        self.neck = block(c, 256)
        self.up1 = block(256, 128)
        self.up2 = block(128, 128)
        self.out = nn.Conv2d(128, n_joints, 1)

    def forward(self, x):
        b = x.shape[0]
        tokens = self.backbone(pixel_values=(x - MEAN.to(x)) / STD.to(x)).last_hidden_state
        g = SIZE // 14
        f = tokens[:, 1 + self.n_reg:].transpose(1, 2).reshape(b, -1, g, g)
        f = self.neck(f)
        f = self.up1(F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False))
        f = self.up2(F.interpolate(f, size=(HM, HM), mode="bilinear", align_corners=False))
        return self.out(f)


def square_geometry(h: int, w: int) -> tuple[int, int, int]:
    """Where an h x w view sits inside its centred square: (offset_x, offset_y, side)."""
    s = max(h, w)
    return (s - w) // 2, (s - h) // 2, s


def decode(hm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Heatmaps (B, J, H, W) to sub-pixel peaks in heatmap units, and the peak values."""
    b, j, h, w = hm.shape
    flat = hm.reshape(b, j, -1)
    conf, idx = flat.max(-1)
    y, x = (idx // w).float(), (idx % w).float()
    # quadratic refinement from the neighbours of the peak
    xi, yi = x.long().clamp(1, w - 2), y.long().clamp(1, h - 2)

    def at(dy, dx):
        col = hm.gather(-1, (xi + dx).view(b, j, 1, 1).expand(b, j, h, 1)).squeeze(-1)
        return col.gather(-1, (yi + dy).view(b, j, 1)).squeeze(-1)

    c, left, right, up, down = at(0, 0), at(0, -1), at(0, 1), at(-1, 0), at(1, 0)
    dx = 0.5 * (right - left) / (2 * c - left - right).clamp(min=1e-6)
    dy = 0.5 * (down - up) / (2 * c - up - down).clamp(min=1e-6)
    return torch.stack([x + dx.clamp(-0.5, 0.5), y + dy.clamp(-0.5, 0.5)], -1), conf


def heatmap_to_view(uv_hm: np.ndarray, geometry: tuple[int, int, int]) -> np.ndarray:
    """Peaks in heatmap units back to pixels of the original view."""
    ox, oy, side = geometry
    return uv_hm * (SIZE / HM) * side / SIZE - np.array([ox, oy])


def prep(view: np.ndarray):
    """A view padded to its centred square with its own corner colour, resized to SIZE."""
    cv2 = _cv2()
    ox, oy, side = square_geometry(*view.shape[:2])
    k = SIZE / side
    A = np.array([[k, 0, ox * k], [0, k, oy * k]], np.float32)
    fill = tuple(float(x) for x in view[2, 2])
    out = cv2.warpAffine(view, A, (SIZE, SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=fill)
    return out, (ox, oy, side)


def load_reader(path: Path, device: str) -> tuple[JointReader, list[str]]:
    # tensors, names and numbers only: safe for downloaded readers
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or "state_dict" not in ck or "names" not in ck:
        raise ReaderError(f"{path} is not a joint reader (needs 'state_dict' and 'names')")
    model = JointReader(len(ck["names"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), list(ck["names"])


@torch.no_grad()
def run(model: JointReader, views: Sequence[np.ndarray], device: str, batch: int = 32):
    uv_all, conf_all = [], []
    for i in range(0, len(views), batch):
        prepped = [prep(v) for v in views[i:i + batch]]
        x = torch.from_numpy(np.stack([p[0] for p in prepped])).permute(0, 3, 1, 2).float().to(device) / 255
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hm = model(x).float()
        else:
            hm = model(x).float()
        uv, conf = decode(hm)
        for (_, geometry), u in zip(prepped, uv.cpu().numpy()):
            uv_all.append(heatmap_to_view(u, geometry))
        conf_all.append(conf.cpu().numpy())
    return np.array(uv_all), np.concatenate(conf_all)


def read_frames(path: Path) -> tuple[np.ndarray, float]:
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ReaderError(f"cannot open video {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[..., ::-1])
    cap.release()
    if len(frames) < 2:
        raise ReaderError(f"{path} has fewer than two frames")
    return np.array(frames), fps


def read_image(path: Path) -> np.ndarray:
    img = _cv2().imread(str(path))
    if img is None:
        raise ReaderError(f"cannot read image {path}")
    return img[..., ::-1]


def calibrate(first: np.ndarray, ref: np.ndarray, layout: Layout):
    """Affine map from video pixels to layout pixels, found by aligning the first frame to
    the reference image (the video model may resize or re-frame the canvas)."""
    cv2 = _cv2()
    h, w = first.shape[:2]
    S = np.array([[layout.w / w, 0, 0], [0, layout.h / h, 0]], np.float32)  # plain stretch to start
    half = (layout.w // 2, layout.h // 2)
    a = cv2.cvtColor(cv2.resize(ref, half), cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(cv2.resize(cv2.warpAffine(first, S, (layout.w, layout.h)), half),
                     cv2.COLOR_RGB2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    converged = True
    try:
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        converged = False
    # warp maps reference (half-size) coords to frame coords; we want frame -> layout
    W2 = np.vstack([warp, [0, 0, 1]])
    W2[:2, 2] *= 2  # half-size -> full-size translation
    A = np.linalg.inv(W2) @ np.vstack([S, [0, 0, 1]])
    return A[:2].astype(np.float32), converged


def read_video(video: Path, reader_path: Path, ref_path: Path | None, layout: Layout,
               rig_joints: Sequence[str], device: str = "cpu", fps: float | None = None) -> dict[str, Any]:
    """Every joint in every frame and both views, in layout pixels, with peak confidences.

    `ref_path=None` skips calibration: the clip is already a layout frame (the exam).
    """
    cv2 = _cv2()
    frames, video_fps = read_frames(video)
    fps = float(fps or video_fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ReaderError(f"{video} reports no frame rate: pass it with --fps")
    model, names = load_reader(reader_path, device)
    if list(names) != list(rig_joints):
        raise ReaderError("the reader was trained on another rig: its joints are not this rig's keypoints, "
                          "in this order")
    converged = True
    if ref_path is None:
        A = np.array([[layout.w / frames.shape[2], 0, 0], [0, layout.h / frames.shape[1], 0]], np.float32)
    else:
        A, converged = calibrate(frames[0], read_image(ref_path), layout)
    frames = np.array([cv2.warpAffine(f, A, (layout.w, layout.h), flags=cv2.INTER_AREA) for f in frames])
    fuv, fconf = run(model, [f[:, :layout.w_front] for f in frames], device)
    suv, sconf = run(model, [f[:, layout.w_front:] for f in frames], device)
    return {"front_uv": fuv, "side_uv": suv, "front_conf": fconf, "side_conf": sconf, "fps": fps,
            "names": np.array(names), "calib": A, "calib_converged": converged, "frames": frames}


def overlay(frames: np.ndarray, det: dict[str, Any], layout: Layout, path: Path, fps: float) -> None:
    """The clip with every read joint drawn on (left pink, right blue, middle white)."""
    cv2 = _cv2()
    names = list(det["names"])
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (layout.w, layout.h))

    def colour(n):  # BGR
        return (209, 60, 245) if n.endswith("_L") else (255, 107, 47) if n.endswith("_R") else (255, 255, 255)

    for t, f in enumerate(frames):
        img = np.ascontiguousarray(f[..., ::-1])
        for j, n in enumerate(names):
            for (u, v), x0 in ((det["front_uv"][t, j], 0), (det["side_uv"][t, j], layout.w_front)):
                cv2.circle(img, (int(u + x0), int(v)), 4, colour(n), -1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
