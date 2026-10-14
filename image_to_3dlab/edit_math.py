"""Pure deformation math for S0/RIG0; inputs share one explicit coordinate space.

No implicit weight normalization, pseudoinverse, bind recovery or unit conversion.
Callers must remove viewer transforms and certify the authoritative bind separately.
"""

from __future__ import annotations

import numpy as np


class EditMathError(ValueError):
    def __init__(self, code: str, message: str, vertices=()):
        super().__init__(message)
        self.code = code
        self.vertices = list(vertices)


def _finite(value, label):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise EditMathError("invalid_data", f"{label} must be numeric") from exc
    if not np.isfinite(array).all():
        raise EditMathError("invalid_data", f"{label} must be finite")
    return array


def affine_matrices(value, label="matrices"):
    array = _finite(value, label)
    if array.ndim != 3 or array.shape[1:] != (4, 4) or not len(array):
        raise EditMathError("invalid_data", f"{label} must have shape (joints, 4, 4)")
    if not np.allclose(array[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise EditMathError("invalid_data", f"{label} must be affine")
    return array


def certify_bind(bind, inverse_bind, *, atol=1e-5, rtol=1e-5):
    """Return the common mesh-to-reference transform, or refuse weight editing.

    This verifies a supplied candidate pose. It does NOT establish that the GLB's
    default node transforms are the authoritative bind pose when this check fails.
    """
    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise EditMathError("invalid_data", "invalid bind tolerance")
    bind = affine_matrices(bind, "bind")
    inverse_bind = affine_matrices(inverse_bind, "inverse bind")
    if bind.shape != inverse_bind.shape:
        raise EditMathError("invalid_data", "bind joint counts differ")
    contributions = bind @ inverse_bind
    common = contributions[0]
    if not np.allclose(contributions, common, atol=atol, rtol=rtol):
        raise EditMathError("incoherent_bind", "joint contributions do not share a common bind")
    if np.linalg.matrix_rank(common[:3, :3]) != 3:
        raise EditMathError("singular_bind", "common bind is singular")
    return common.copy()


def rebind_inverse(old_bind, new_bind, inverse_bind):
    """Compensate new joint centers/frames, retaining a certified common rest bind."""
    certify_bind(old_bind, inverse_bind)
    old = affine_matrices(old_bind, "old bind")
    new = affine_matrices(new_bind, "new bind")
    inverse = affine_matrices(inverse_bind, "inverse bind")
    if new.shape != old.shape:
        raise EditMathError("invalid_data", "new bind joint count differs")
    try:
        result = np.linalg.solve(new, old @ inverse)
    except np.linalg.LinAlgError as exc:
        raise EditMathError("singular_bind", "new bind is singular") from exc
    certify_bind(new, result)
    return result


def skin_matrices(joints, weights, matrices):
    """Blend complete joint matrices (including bind/space conversion), per vertex."""
    matrices = affine_matrices(matrices)
    joints = _finite(joints, "joint indices")
    weights = _finite(weights, "weights")
    if joints.ndim != 2 or joints.shape[1] != 4 or joints.shape != weights.shape:
        raise EditMathError("invalid_data", "joints and weights must have shape (vertices, 4)")
    if (joints < 0).any() or (joints >= len(matrices)).any() or (joints != np.floor(joints)).any():
        raise EditMathError("invalid_data", "joint indices must be integers within the skin")
    if (weights < 0).any() or not np.allclose(weights.sum(axis=1), 1, atol=1e-5, rtol=0):
        raise EditMathError("invalid_weights", "weights must be nonnegative and sum to one")
    return np.einsum("ni,nijk->njk", weights, matrices[joints.astype(np.int64)])


def posed_delta_to_rest(delta, joints, weights, matrices, *, max_condition=1e4,
                       max_rest_distance=None):
    """Invert the blended linear map; reject unstable/oversized gestures atomically."""
    delta = _finite(delta, "delta")
    blend = skin_matrices(joints, weights, matrices)[:, :3, :3]
    if delta.shape != (len(blend), 3):
        raise EditMathError("invalid_data", "delta must have shape (vertices, 3)")
    if not np.isfinite(max_condition) or max_condition < 1:
        raise EditMathError("invalid_data", "invalid condition limit")
    if max_rest_distance is not None and (not np.isfinite(max_rest_distance) or max_rest_distance <= 0):
        raise EditMathError("invalid_data", "invalid displacement limit")
    if not len(blend):
        return delta.copy()
    condition = np.linalg.cond(blend)
    bad = np.flatnonzero(~np.isfinite(condition) | (condition > max_condition))
    if len(bad):
        raise EditMathError("unstable_pose", "pose collapses the selected surface", bad.tolist())
    result = np.linalg.solve(blend, delta[..., None])[..., 0]
    bad = ~np.isfinite(result).all(axis=1)
    if max_rest_distance is not None:
        bad |= np.linalg.norm(result, axis=1) > max_rest_distance
    if bad.any():
        raise EditMathError("displacement_limit", "rest displacement exceeds the limit",
                            np.flatnonzero(bad).tolist())
    return result


def _rotation(value, label):
    array = _finite(value, label)
    if (array.shape != (3, 3) or not np.allclose(array.T @ array, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(array), 1, atol=1e-6)):
        raise EditMathError("invalid_rotation", f"{label} must be a proper rotation")
    return array


def calibrate_local_rotation(rest, posed, calibration):
    """Rotate the entire local delta's axis. Descendant FK belongs to the caller."""
    rest = _rotation(rest, "rest")
    posed = _rotation(posed, "posed")
    calibration = _rotation(calibration, "calibration")
    delta = rest.T @ posed
    return rest @ calibration @ delta @ calibration.T
