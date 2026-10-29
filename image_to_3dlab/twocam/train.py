"""Teach the joint reader one creature, from its rendered poses.

Ported from AnyMove's `mocap/train.py` (Patch, Apache-2.0), changed to read the layout from
the rig and to run on CUDA or the CPU. Every sample gives two training images (front and
side). The renders are transparent, so each is laid on a fresh random backdrop, then
jittered in framing, colour, sharpness and compression, the ways a video model's frames
differ from a clean render. Every 25th sample is held out; its mean joint error in layout
pixels says whether the reader can be trusted (AnyMove's readers: 3-9 px).

Starts from Meta's pretrained DINOv2 weights (346 MB, Apache-2.0): the caller must have the
user's yes before calling `train` when they are not cached (`backbone_cached`).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .layout import BACKDROP, FLOOR, Layout
from .reader import BACKBONE_ID, HM, SIZE, JointReader, _cv2, decode, square_geometry


class TrainError(RuntimeError):
    pass


def backbone_cached() -> bool:
    """Whether DINOv2's pretrained weights are already on this machine."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return any(isinstance(try_to_load_from_cache(BACKBONE_ID, name), str)
               for name in ("model.safetensors", "pytorch_model.bin"))


def backdrop(h: int, w: int, ground: int, rng) -> np.ndarray:
    g = rng.uniform(95, 175)
    img = np.empty((h, w, 3), np.float32)
    img[:] = g + rng.uniform(-6, 6, 3)
    img += np.linspace(-1, 1, h)[:, None, None] * rng.uniform(-10, 10)
    img[ground:] += rng.uniform(5, 35) * (1 if rng.random() < 0.85 else -1)
    return img


def photometric(img: np.ndarray, rng) -> np.ndarray:
    cv2 = _cv2()
    img = img.astype(np.float32)
    if rng.random() < 0.8:  # brightness, contrast, saturation, hue
        img = img * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
        m = img.mean()
        img = (img - m) * rng.uniform(0.75, 1.25) + m
        hsv = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-5, 5)) % 180
        hsv[..., 1] *= rng.uniform(0.7, 1.3)
        img = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.5))
    if rng.random() < 0.3:
        f = rng.uniform(0.4, 0.8)
        small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.2:
        img = img + rng.normal(0, rng.uniform(2, 8), img.shape)
    img = img.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        _, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return img


class Views(Dataset):
    def __init__(self, metas: Sequence[Path], layout: Layout, train: bool):
        self.metas, self.layout, self.train = list(metas), layout, train

    def __len__(self) -> int:
        return 2 * len(self.metas)

    def __getitem__(self, i: int):
        cv2 = _cv2()
        rng = np.random.default_rng() if self.train else np.random.default_rng(i)
        m = self.metas[i // 2]
        view = ("front", "side")[i % 2]
        meta = json.loads(m.read_text(encoding="utf-8"))
        s = meta["scale"]
        path = str(m).replace("_meta.json", f"_{view}.png")
        rgba = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)[..., [2, 1, 0, 3]].astype(np.float32)
        h, w = rgba.shape[:2]
        kp = np.array(meta[f"{view}_uv"], np.float32) * s
        ground = round(self.layout.ground_v * s)
        if self.train:
            bg = backdrop(h, w, ground, rng)
        else:
            bg = np.empty((h, w, 3), np.float32)
            bg[:] = BACKDROP
            bg[ground:] = FLOOR
        a = rgba[..., 3:] / 255
        img = rgba[..., :3] * a + bg * (1 - a)
        if self.train and rng.random() < 0.7:  # the divider at the shared edge
            if view == "front":
                img[:, -2:] = 34
            else:
                img[:, :2] = 34
        ox, oy, side = square_geometry(h, w)
        k = SIZE / side
        A = np.array([[k, 0, ox * k], [0, k, oy * k]], np.float32)
        if self.train:  # framing jitter: scale, small turn, shift
            c = SIZE / 2
            R = cv2.getRotationMatrix2D((c, c), rng.uniform(-4, 4), rng.uniform(0.85, 1.15))
            R[:, 2] += rng.uniform(-0.05, 0.05, 2) * SIZE
            A = R @ np.vstack([A, [0, 0, 1]])
        out = cv2.warpAffine(img, A.astype(np.float32), (SIZE, SIZE), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(x) for x in bg[0, 0]))
        kp = kp @ A[:, :2].T + A[:, 2]
        if self.train:
            out = photometric(out, rng)
        x = torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1).float() / 255
        return x, torch.from_numpy(kp.astype(np.float32)), torch.tensor([ox, oy, side, s], dtype=torch.float32)


def heatmaps(kp: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """Gaussian targets at heatmap resolution for keypoints given in SIZE pixels."""
    b, j, _ = kp.shape
    g = (kp * HM / SIZE).view(b, j, 1, 1, 2)
    ys, xs = torch.meshgrid(torch.arange(HM, device=kp.device), torch.arange(HM, device=kp.device), indexing="ij")
    d2 = (xs - g[..., 0]) ** 2 + (ys - g[..., 1]) ** 2
    return torch.exp(-d2 / (2 * sigma ** 2))


def split(dataset: Path) -> tuple[list[Path], list[Path]]:
    metas = sorted(Path(dataset).glob("*_meta.json"))
    val = [m for m in metas if int(m.name[:5]) % 25 == 0]
    return [m for m in metas if int(m.name[:5]) % 25 != 0], val


def check_dataset(dataset: Path, info: dict[str, Any]) -> int:
    train, val = split(dataset)
    if len(train) < 50 or not val:
        raise TrainError(f"{dataset} has {len(train) + len(val)} samples; render the dataset first (2,000 or so)")
    meta = json.loads((train or val)[0].read_text(encoding="utf-8"))
    if len(meta["front_uv"]) != len(info["keypoints"]):
        raise TrainError("the dataset was rendered for another rig: its joints are not this rig's keypoints")
    return len(train) + len(val)


def train(dataset: Path, info: dict[str, Any], out: Path, *, epochs: int = 20, batch: int = 20,
          device: str = "cuda", workers: int = 4, log: Callable[[str], None] | None = None) -> dict[str, float]:
    """Train and save the reader after every epoch (so a stopped run keeps its best so far).
    Returns the last held-out error per view, in layout pixels."""
    say = log or (lambda _msg: None)
    layout = Layout.from_rig_info(info)
    names = [k["name"] for k in info["keypoints"]]
    check_dataset(dataset, info)
    _, val = split(dataset)
    vl = DataLoader(Views(val, layout, False), batch_size=batch, num_workers=workers)
    model = JointReader(len(names), pretrained=True).to(device)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": 2e-5},
                             {"params": head, "lr": 1e-3}], weight_decay=0.05)
    base = [g["lr"] for g in opt.param_groups]
    cuda = device == "cuda"
    step, t0, warm = 0, time.time(), 300
    result: dict[str, float] = {}
    for ep in range(epochs):
        tr, _ = split(dataset)  # samples still being rendered join as they land
        dl = DataLoader(Views(tr, layout, True), batch_size=batch, shuffle=True, num_workers=workers,
                        drop_last=True, persistent_workers=False)
        model.train()
        for bi, (x, kp, _) in enumerate(dl):
            f = 0.5 * (1 + math.cos(math.pi * (ep + bi / len(dl)) / epochs)) * min(1, (step + 1) / warm)
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * f
            x, kp = x.to(device, non_blocking=True), kp.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                pred = model(x)
            loss = F.mse_loss(pred.float(), heatmaps(kp)) * 1000
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                say(f"epoch {ep} step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s")
        model.eval()
        errs: dict[str, list] = {"front": [], "side": []}
        with torch.no_grad():
            for bi, (x, kp, geo) in enumerate(vl):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                    pred = model(x.to(device)).float()
                uv, _ = decode(pred)
                uv = uv.cpu() * SIZE / HM

                def to_layout(p, geo=geo):  # undo the resize, the padding and the render scale
                    return (p * geo[:, None, 2:3] / SIZE - geo[:, None, :2]) / geo[:, None, 3:4]

                e = (to_layout(uv) - to_layout(kp)).norm(dim=-1)
                for r in range(len(x)):
                    errs[("front", "side")[(bi * batch + r) % 2]].append(e[r].numpy())
        result = {v: float(np.mean(errs[v])) for v in errs}
        say(f"epoch {ep}: held-out error front {result['front']:.2f} px, side {result['side']:.2f} px")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "names": names, "epoch": ep, "val": result}, out)
    return result
