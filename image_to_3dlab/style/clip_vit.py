"""OpenAI CLIP's image transformer, enough of it to run the CSD style checkpoint.

CSD (Somepalli et al., 2024) is CLIP ViT-L/14's image tower fine-tuned for style, plus a
1024x768 style projection. Its upstream code builds the tower with `clip.load`, which
downloads OpenAI's own weights first; the CSD checkpoint already holds every weight, so
the tower is written out here instead (the same layout, so the keys load as they are).
"""

from __future__ import annotations

import torch
from torch import nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential()
        self.mlp.add_module("c_fc", nn.Linear(width, width * 4))
        self.mlp.add_module("gelu", QuickGELU())
        self.mlp.add_module("c_proj", nn.Linear(width * 4, width))
        self.ln_2 = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (tokens, batch, width)
        y = self.ln_1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.Sequential(*[Block(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    """CLIP's `model.visual` without its final projection: returns the 1024-wide class token."""

    def __init__(self, resolution: int = 224, patch: int = 14, width: int = 1024, layers: int = 24,
                 heads: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch, stride=patch, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((resolution // patch) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x).flatten(2).permute(0, 2, 1)                      # (batch, patches, width)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x).permute(1, 0, 2)                                # (tokens, batch, width)
        x = self.transformer(x).permute(1, 0, 2)
        return self.ln_post(x[:, 0, :])


class CSDStyle(nn.Module):
    """The CSD backbone and its style head: one unit-length 768-wide style vector per image."""

    def __init__(self, **vit):
        super().__init__()
        self.backbone = VisionTransformer(**vit)
        width = self.backbone.ln_post.normalized_shape[0]
        self.last_layer_style = nn.Parameter(torch.zeros(width, 768))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.backbone(x) @ self.last_layer_style, dim=1)


def csd_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    """The weights CSDStyle needs from a CSD training checkpoint (`module.` prefix and content head dropped)."""
    state = checkpoint.get("model_state_dict", checkpoint)
    out = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key.startswith("backbone.") or key == "last_layer_style":
            out[key] = value
    return out
