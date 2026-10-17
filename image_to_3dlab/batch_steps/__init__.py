"""The steps a batch row can run, one module each, looked up by name.

    image_to_3dlab/batch_steps/<name>.py

Names are fixed (`image_to_3dlab.batch.STEP_NAMES`): `generate`, `finish`, `materials`,
`export_unreal`, `checks`. A row lists the ones it wants, in order. A name whose module
does not exist yet is skipped with a warning, so a spec can ask for a step before it is
built; a module that exists but fails to import is an error, because a broken step must
not look like a missing one.

**Adding a step** is one file here, with:

    def run(ctx: StepContext) -> StepResult: ...
    def preflight(rows: list[BatchRow]) -> list[str]: ...   # optional

`preflight` gets every row that asks for the step, before the batch starts, and returns
what is missing (a tool not installed, an option it does not know). Any problem stops the
batch before its first row, so a missing install never fails row 7 of 400. It must never
download or install anything: name the command instead.

`run` gets the row's current asset (the GLB the previous step left) and returns a
`StepResult`: the new asset and its provenance sidecar if it made one, every file it
wrote (they are hashed, which is how a resumed batch knows the step really finished), and
`ok=False` with a `reason` to stop the row. A step that only checks returns no asset.

Steps write inside `ctx.step_dir`, which the runner empties before a step runs again.
Per-step settings come from the spec, under the step's name, in `ctx.options`:

    {"defaults": {"finish": {"atlas": 1024}}, "rows": [{"id": "x", "finish": {"paint": true}}]}
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from image_to_3dlab.batch import BatchRow


@dataclass
class StepContext:
    row: BatchRow
    row_dir: Path            # rows/<id>/
    step_dir: Path           # rows/<id>/<step>/, emptied before the step runs
    asset: Path | None       # the current GLB, from the previous step or the row's `asset`
    sidecar: Path | None     # its provenance record, when it has one
    intent: dict[str, Any]   # use_case, distribution, commercial_intent
    license_policy: dict[str, Any]
    options: dict[str, Any]  # the spec's settings for this step
    log: Callable[[str], None]
    picture: Path | None = None   # the row's current picture: its first image, or a drawn one
    clips: list[Path] = field(default_factory=list)   # animated GLBs a step before made


@dataclass
class StepResult:
    asset: Path | None = None       # the new current asset; None leaves it as it was
    sidecar: Path | None = None     # the new asset's provenance record
    files: list[Path] = field(default_factory=list)
    record: dict[str, Any] = field(default_factory=dict)   # kept in the batch state
    ok: bool = True
    reason: str | None = None
    picture: Path | None = None     # the new current picture (a step that draws one)
    clips: list[Path] = field(default_factory=list)   # animated GLBs made (animate, clips)
    # several results to pick from, and no pick yet: {"candidates": [{"choice", "file"}]};
    # the row waits until a pick is given
    waiting: dict[str, Any] | None = None
