"""Automatic checks for generated assets, the same for every asset in a batch.

    from image_to_3dlab.checks import Asset, run_checks, write_report

    report = run_checks(Asset(glb=path, reference=ref, key="mob_0412"), profile="creature")
    write_report(report)                      # <name>.checks.json beside the GLB

    report, path = check_and_enqueue(Asset(glb=path, key="mob_0412"), profile="prop")  # and queue it

Every check returns `{"check", "status", "value", "threshold", "message"}` (plus optional
`details`); see `registry.py` for the contract and `profiles/` for the thresholds.
"""

from image_to_3dlab.checks import (  # noqa: F401 - importing registers the checks
    budget_checks,
    licenses,
    measure_checks,
    mesh_checks,
    silhouette,
    validator,
)
from image_to_3dlab.checks.asset import Asset, find_validator
from image_to_3dlab.checks.registry import (
    REGISTRY,
    STATUSES,
    load_profile,
    profile_names,
    register,
    report_path,
    result,
    run_checks,
    validate_result,
    write_report,
)
from image_to_3dlab.checks.review import check_and_enqueue

__all__ = [
    "REGISTRY", "STATUSES", "Asset", "check_and_enqueue", "find_validator", "load_profile", "profile_names", "register",
    "report_path", "result", "run_checks", "validate_result", "write_report",
]
