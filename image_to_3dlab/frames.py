"""glTF's frame and Blender's, written once.

glTF is Y up with +Z the front; Blender is Z up with -Y the front, and its glTF importer puts a
glTF point (x, y, z) at Blender (x, -z, y). Everything that crosses between the two (the fit,
route B0's segmentation, the Unreal export's facing and grip) goes through here. Plain numpy,
so Blender's own Python imports it too.
"""

from __future__ import annotations

import numpy as np

GLTF_TO_BLENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def gltf_to_blender(points) -> np.ndarray:
    """glTF points or directions, (3,) or (N, 3), in Blender's frame."""
    return np.asarray(points, float) @ GLTF_TO_BLENDER.T


def blender_to_gltf(points) -> np.ndarray:
    """Blender points or directions, (3,) or (N, 3), in glTF's frame (the inverse turn)."""
    return np.asarray(points, float) @ GLTF_TO_BLENDER
