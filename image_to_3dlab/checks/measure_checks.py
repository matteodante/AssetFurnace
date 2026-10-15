"""Size against a reference: height and footprint, in metres (glTF units, Y up)."""

from __future__ import annotations

from image_to_3dlab.checks.registry import grade_max, register, result


def extents(bounds) -> dict[str, float]:
    size = bounds[1] - bounds[0]
    return {"width": float(size[0]), "height": float(size[1]), "depth": float(size[2])}


def reference_extents(asset) -> dict[str, float]:
    """Wanted sizes: measurements given in numbers win, the reference model fills the rest.

    A batch row that states a height means that height, even when its reference model
    (the original game's, say) was built at another scale.
    """
    wanted = extents(asset.reference_asset.bounds) if asset.reference_asset is not None else {}
    wanted.update({k: float(v) for k, v in (asset.reference_numbers or {}).items()
                   if k in ("width", "height", "depth") and v})
    return wanted


def error_percent(value: float, wanted: float) -> float:
    return 100.0 * abs(value - wanted) / wanted if wanted else float("inf")


def _limits(config):
    return {"warn": config.get("warn", 5.0), "fail": config.get("fail", 15.0)}


@register("measure.height", needs=("glb", "reference_height"))
def height(asset, config):
    wanted = reference_extents(asset).get("height")
    limits = _limits(config)
    if not wanted:
        return result("measure.height", "warn", None, limits, "the reference gives no height")
    got = extents(asset.bounds)["height"]
    error = error_percent(got, wanted)
    return result("measure.height", grade_max(error, **limits),
                  {"height": round(got, 4), "error_percent": round(error, 2)},
                  {**limits, "reference": round(wanted, 4)},
                  f"height {got:.3f} m against {wanted:.3f} m ({error:.1f}% off)")


@register("measure.footprint", needs=("glb", "reference_footprint"))
def footprint(asset, config):
    wanted = reference_extents(asset)
    got = extents(asset.bounds)
    limits = _limits(config)
    axes = [axis for axis in ("width", "depth") if wanted.get(axis)]
    if not axes:
        return result("measure.footprint", "warn", None, limits, "the reference gives no width or depth")
    errors = {axis: error_percent(got[axis], wanted[axis]) for axis in axes}
    worst_axis = max(errors, key=errors.get)
    error = errors[worst_axis]
    return result("measure.footprint", grade_max(error, **limits),
                  {**{a: round(got[a], 4) for a in axes}, "error_percent": round(error, 2)},
                  {**limits, "reference": {a: round(wanted[a], 4) for a in axes}},
                  f"{worst_axis} {got[worst_axis]:.3f} m against {wanted[worst_axis]:.3f} m "
                  f"({error:.1f}% off)")
