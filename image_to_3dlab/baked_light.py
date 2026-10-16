"""A learned check for light painted into the albedo (design section 3B, first candidate).

No pixel rule decides: a small logistic head on frozen SigLIP 2 embeddings (Apache-2.0, the one
the style judges use) does, trained on synthetic pairs and admitted only if it passes the test
against the user's labels.

**The pairs.** Clean albedos come from CC0 Poly Haven models, whose textures are scanned without
light. Each is rendered from an orbit twice: as it is (unlit), and on a neutral studio material
under a sun fixed in the world, with a grey sky (``multiview.render_views(lit=True, light=...)``).
The second over the first is the shading that light, its shadows and its occlusion would
leave; painting it into the albedo with a random strength makes a "baked" twin
(``bake_views``). The light is fixed in the world because baked light is: it stays on the
same side whichever way the asset is turned.

**The features.** Each view is embedded twice, as the albedo on grey and as its normals
drawn as colour, and the two means are put side by side, so the head can tell shading that
follows the shape from colour that was painted.

**The test** (design section 6): ROC AUC with a 95% lower bound of at least 0.90, and at the
threshold chosen on the calibration split, at least 80% of baked assets caught and at most 10%
of clean ones flagged. Splits are by asset, never by sample. On synthetic data this only says
the head learned the synthetic difference; ``admitted`` needs the human-labelled set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from image_to_3dlab import multiview as mv
from image_to_3dlab.materials import linear_to_srgb, luminance, srgb_to_linear

GREY = (214, 214, 214)              # the grey the style embeddings put cut-outs on
AUC_LOW = 0.90
CATCH = 0.80
FLAG = 0.10


# --------------------------------------------------------------------------- synthetic pairs

def bake_views(views: list[mv.View], strength: float) -> list[np.ndarray]:
    """Each view's albedo with the studio light's shading painted in, `strength` 0..1.

    The shading is the lit picture's luminance over the albedo's, relative to its median over
    every object pixel of every view, so the albedo keeps its overall brightness.
    """
    ratios, shown = [], []
    for view in views:
        mask = view.material >= 0
        base = luminance(srgb_to_linear(view.color / 255.0))
        lit = luminance(srgb_to_linear(view.lit / 255.0))
        ratio = np.where(mask & (base > 1e-4), lit / np.maximum(base, 1e-4), 1.0)
        ratios.append(ratio)
        shown.append(mask & (base > 1e-4))
    pooled = np.concatenate([r[s] for r, s in zip(ratios, shown)]) if any(s.any() for s in shown) else np.ones(1)
    middle = float(np.median(pooled)) or 1.0
    out = []
    for view, ratio, mask in zip(views, ratios, shown):
        gain = np.clip(ratio / middle, 0.02, 20.0) ** strength
        baked = srgb_to_linear(view.color / 255.0) * gain[..., None]
        image = view.color.copy()
        image[mask] = np.round(linear_to_srgb(baked) * 255.0).astype(np.uint8)[mask]
        out.append(image)
    return out


def picture(colour: np.ndarray, material: np.ndarray) -> Image.Image:
    """The object on the embeddings' grey, as SigLIP is shown it."""
    out = np.empty_like(colour)
    out[:] = GREY
    shown = material >= 0
    out[shown] = colour[shown]
    return Image.fromarray(out)


def normal_picture(view: mv.View) -> Image.Image:
    """The world normals drawn as colour (0.5 + n / 2), on the same grey."""
    rgb = np.round((view.normal * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    return picture(rgb, view.material)


def features(embed, pictures: list[Image.Image], normals: list[Image.Image]) -> np.ndarray:
    """Mean albedo embedding and mean normal embedding, side by side."""
    return np.concatenate([np.mean(embed(pictures), axis=0), np.mean(embed(normals), axis=0)])


# --------------------------------------------------------------------------- the head

@dataclass
class LogisticHead:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, l2: float = 1.0, steps: int = 50) -> LogisticHead:
        """L2-regularised logistic regression by Newton's method on standardised features;
        classes weighted to count equally."""
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        mean, scale = x.mean(axis=0), x.std(axis=0) + 1e-6
        z = np.c_[(x - mean) / scale, np.ones(len(x))]
        balance = np.where(y > 0.5, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9))
        w = np.zeros(z.shape[1])
        ridge = np.full(z.shape[1], l2)
        ridge[-1] = 0.0
        for _ in range(steps):
            p = 1.0 / (1.0 + np.exp(-z @ w))
            grad = z.T @ (balance * (p - y)) + ridge * w
            hess = (z * (balance * p * (1 - p))[:, None]).T @ z + np.diag(ridge + 1e-9)
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        return cls(w[:-1], float(w[-1]), mean, scale)

    def score(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-(z @ self.weights + self.bias)))

    def to_json(self) -> dict:
        return {"weights": self.weights.tolist(), "bias": self.bias, "mean": self.mean.tolist(),
                "scale": self.scale.tolist()}

    @classmethod
    def from_json(cls, data: dict) -> LogisticHead:
        return cls(np.asarray(data["weights"]), float(data["bias"]), np.asarray(data["mean"]),
                   np.asarray(data["scale"]))


# --------------------------------------------------------------------------- the test

def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Probability that a baked sample scores above a clean one; ties count half."""
    y, s = np.asarray(y) > 0.5, np.asarray(s, dtype=np.float64)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def auc_lower_bound(y, s, groups, rounds: int = 2000, seed: int = 0, level: float = 0.95) -> float:
    """The (1 - level) / 2 quantile of the AUC over bootstrap resamples of whole assets."""
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    names = np.unique(groups)
    members = {name: np.flatnonzero(groups == name) for name in names}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(rounds):
        pick = np.concatenate([members[n] for n in rng.choice(names, len(names))])
        value = roc_auc(y[pick], s[pick])
        if not np.isnan(value):
            values.append(value)
    return float(np.quantile(values, (1 - level) / 2)) if values else float("nan")


def split_assets(names: list[str], seed: str = "assetfurnace-baked-light") -> dict[str, list[str]]:
    """Half to train, a quarter to calibrate the threshold, a quarter to test; by asset."""
    order = sorted(set(names), key=lambda n: hashlib.sha1(f"{seed}:{n}".encode()).hexdigest())
    half, three = len(order) // 2, (3 * len(order)) // 4
    return {"train": order[:half], "calibration": order[half:three], "test": order[three:]}


def choose_threshold(y: np.ndarray, s: np.ndarray, catch: float = CATCH) -> float:
    """The highest score threshold that still catches `catch` of the baked samples."""
    bad = np.sort(np.asarray(s, dtype=np.float64)[np.asarray(y) > 0.5])[::-1]
    if not len(bad):
        return 0.5
    return float(bad[int(np.ceil(catch * len(bad))) - 1])


def rates(y: np.ndarray, s: np.ndarray, threshold: float) -> tuple[float, float]:
    """(share of baked caught, share of clean flagged) at `threshold`."""
    y, flagged = np.asarray(y) > 0.5, np.asarray(s) >= threshold
    return (float(flagged[y].mean()) if y.any() else float("nan"),
            float(flagged[~y].mean()) if (~y).any() else float("nan"))


def verdict(auc_low: float, caught: float, flagged: float, human: bool) -> dict:
    out = {"auc_lower_bound": auc_low >= AUC_LOW, "caught": caught >= CATCH, "flagged": flagged <= FLAG,
           "human_labels": human}
    out["numbers_pass"] = out["auc_lower_bound"] and out["caught"] and out["flagged"]
    out["admitted"] = out["numbers_pass"] and human
    return out


QUESTION = ("This is a 3D model rendered with its colour texture only, without any lighting. "
            "Are shadows or highlights painted into its colours?")


def run_qwen(task: str, images: list[Path], question: str, work: Path, *, python: Path | None = None,
             run=None, lock=None) -> list:
    """Qwen3.5-4B's answer for each image, in the materials environment, GPU lock held meanwhile."""
    import subprocess

    from image_to_3dlab import gpu_lock
    from image_to_3dlab import material_models as mm
    from image_to_3dlab.delight import WORKER

    folder = mm.require(mm.QWEN35)
    python = python or mm.require_env()
    work.mkdir(parents=True, exist_ok=True)
    job_path, result = work / f"{task}_job.json", work / f"{task}_result.json"
    job_path.write_text(json.dumps({"task": task, "model": str(folder), "question": question, "result": str(result),
                                    "images": [{"path": str(i)} for i in images]}, indent=2), encoding="utf-8")
    with (lock or gpu_lock.gpu)("materials"):
        done = (run or subprocess.run)([str(python), str(WORKER), str(job_path)], capture_output=True, text=True,
                                       check=False)
    if done.returncode:
        tail = (done.stderr or done.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError(f"Qwen3.5 failed: {' / '.join(tail)}")
    return json.loads(result.read_text(encoding="utf-8"))["answers"]


def save_head(head: LogisticHead, path: Path, record: dict) -> None:
    Path(path).write_text(json.dumps({"head": head.to_json(), **record}, indent=2), encoding="utf-8")


def load_head(path: Path) -> tuple[LogisticHead, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return LogisticHead.from_json(data.pop("head")), data
