# Batches: many assets, one command

    python pipeline.py --run-batch manifests/my-batch.json     # or my-batch.csv

A batch is a list of assets, one row each. Every row runs the same steps with the same
checks, gets its own `.provenance.json`, and a stopped batch picks up where it left off
when you run the same command again. In the studio, **Batches** in the top bar does the
same from a page; an agent in another repository drives it through the
[MCP server](#the-mcp-server). Finished rows join the Library and the review queue.

Why it works the way it does: [the design](plans/2026-10-15-batch-mcp-design.md).

## The spec

Start from [`manifests/example-batch.json`](../manifests/example-batch.json) or
[`manifests/example-batch.csv`](../manifests/example-batch.csv). **Paths are relative to
the spec file**, like in a run manifest.

```json
{
  "schema_version": 1,
  "name": "pilot-props",
  "use_case": "game",
  "distribution": "private",
  "commercial_intent": true,
  "license_policy": {"allow_conditional": true},
  "output": {"directory": "../output/batches"},
  "defaults": {"backend": "pixal3d", "kind": "prop", "faces": 5000,
               "steps": ["generate", "finish"], "parameters": {"seed": 42}},
  "rows": [
    {"id": "SM_Barrel_01", "images": ["refs/barrel.png"], "faces": 3000},
    {"id": "SK_Spider_01", "images": ["refs/spider.png", "refs/spider_side.png"],
     "kind": "creature", "faces": 20000},
    {"id": "SM_Crate_02", "asset": "../output/old/crate.glb", "steps": ["finish"]}
  ]
}
```

| Field | What it is |
|---|---|
| `id` | Your name for the asset, unique in the batch. Letters, digits, `_ . -`: it names the row's folder and files |
| `images` | The first is what the generator sees; the rest are kept as references for later steps |
| `asset` | An existing GLB, for a row that skips `generate` |
| `kind` | `prop` (default), `creature` or `equipment` |
| `faces` | Face budget for `finish`, 1,000 to 200,000 (default 40,000) |
| `profile` | Output profile, for `export_unreal` once it exists |
| `size_cm` | The asset's real size in centimetres, for `export_unreal` |
| `measure` | What `size_cm` measures: `height` (default) or `longest` side |
| `steps` | In order. Default `generate`, `finish` |
| `backend` | Default `pixal3d`, the only generator batches run today (see below) |
| `parameters`, `seed` | Generator settings: `seed`, `steps`, `gss`, `gsh`, `fov`, `matte` for Pixal3D |
| `finish`, ... | Settings for one step, under its name: `"finish": {"atlas": 1024}` |

`defaults` fills any field a row leaves out (`parameters` and step settings are merged).
`rows` can also name a CSV file: `"rows": "pilot-props.csv"`.

A CSV holds rows only, one per line, with `;` between several values in a cell and an
empty cell for the default:

```csv
id,images,kind,faces,profile,steps,backend,seed,size_cm,measure
SM_Barrel_01,refs/barrel.png,prop,3000,unreal,generate;finish,pixal3d,42,90,height
```

A CSV on its own runs as a private showcase, like a single run without a manifest; for a
game, say so: `--use-case game --distribution private --commercial`. Its outputs go to
`output/batches/<file name>/`.

**The whole spec is checked before the first row**: unknown fields or steps, missing
images, duplicate ids, face budgets out of range, the licence policy, and whatever each
step needs installed. A bad batch does not start, rather than stopping at row 7.

## The steps

| Step | What it does |
|---|---|
| `generate` | Writes the row's run manifest and runs it, exactly as `pipeline.py --run-manifest` would |
| `finish` | The studio's Finish (`scripts/retopo_repaint.py`) at the row's face budget, with Pixel Match when the model kept its camera |
| `materials`, `export_unreal`, `checks` | Reserved names: skipped with a warning until they exist |

`finish` does **not** repaint: the repaint needs Hunyuan 2.1 Paint weights, whose licence
excludes the EU, UK and South Korea, and runs on Apple Silicon only. A row that wants it
says `"finish": {"paint": true}`. Its other settings: `photo` (Pixel Match, on by
default), `atlas`, `texture_size`, `metallic`, `roughness`, `ior`, `voxel`, `skip_bake`,
`skip_compress`.

Adding a step is one file in `image_to_3dlab/batch_steps/`; its docstring has the
contract. A done row whose skipped step later exists runs that step (and the ones after
it) on the next run.

**Nothing downloads.** Before the first row, each step checks what it needs (Pixal3D's
build and weights, the background remover an uncut picture would need, Blender, the
repaint's files when asked for) and the batch stops with the command to install it. SF3D
and TRELLIS.2 fetch weights on first use, so they are not offered in batches yet. A resume
checks only the rows it will run.

**Licences.** Every model a row would run is checked against the batch's intent before
the first row: its generator, the repaint if asked, and for a row with only an asset, the
model its record names. A repainted asset's provenance lists the paint model and takes the
stricter licence class.

## What it writes

```
output/batches/<name>/
  batch-state.json     every row's status and every step's files, with their sha256
  outputs.json         the finished rows and their files (relative paths)
  spec.json            the spec as resolved for the last run
  run.log              the runner's output, when started from the MCP server
  rows/<id>/
    manifest.json      re-run this row alone: pipeline.py --run-manifest <this file>
    input/             copies of the row's images
    generate/<licence class>/<stem>.glb, .provenance.json, .json, .svviews/
    finish/<id>.glb, <id>.provenance.json, <id>.retopo-repaint.json, finish.log
```

`outputs.json` lists only finished rows, with their kind, profile, size, final asset,
its provenance and every file the steps wrote: the hand-off for anything that reads a
batch's results.

**The Library.** Finished rows under `output/batches/` are Library assets: the row's
picture, its 3D model and its finished model, linked the way the studio links its own
runs, so you can rig or hide a batch asset like any other.

**The review queue.** Each finished row is queued for review in `output/review-log.jsonl`
(the folder that holds the batches folder) through the checks' own
`image_to_3dlab.checks.enqueue_for_review`, or, until that package is installed, a line
in the same format. The key is the row's id and the version its GLB's sha256, with the
`checks` report when that step ran on this very file; the same version is never queued
twice, a new one is. `equipment` rows are queued with the checks' `equipment_part` kind.
Approving or sending back happens in the review queue, and the Batches page shows where
each row's current version stands.

## Stopping and resuming

Run the same command again. Then:

- a **done** row is skipped, as long as every file its steps wrote is still there at the
  size recorded; one whose files went missing runs again from its first broken step;
- an **interrupted** row starts again from its first step whose files are not all still
  there with the recorded hash, so a finish that crashed does not cost a generation;
- a **failed** row stays failed (its reason is in the state) unless you add
  `--retry-failed`: a row that always fails should not take the GPU on every resume;
- a row you **edited** in the spec runs again; its old files stay on disk;
- a changed **intent or licence policy** refuses to resume: rename the batch or remove its
  folder.

One batch runs at a time per output folder: the GPU is the bottleneck, and a second one
is refused while the first is alive. The lock is `<output folder>/.batch.lock`, taken
atomically, so two submissions at once cannot both start; one left by a dead process is
taken over. The studio does not check for running batches, so
do not generate there at the same time.

Exit codes: 0 every row done, 1 some rows failed, 2 the batch cannot start, 130
interrupted.

## The Batches page

**Batches** in the studio's top bar. Give a spec by its path on this computer, or choose
the file (an uploaded spec is kept in `output/batches/.uploads/<content hash>/`, so a later
upload with the same name never changes an earlier batch's spec; its relative paths resolve
from there, so use absolute ones, or give the path instead). For a CSV, pick the
use case, distribution and commercial intent. **Check** validates without running;
**Start** runs it in the background. The page follows batches in `output/batches/` only, so
it refuses a spec that writes elsewhere (run that one with `pipeline.py --run-batch`). Each batch shows how far it got; open one to see every
row with its status, why it failed, and its review state. **Resume** carries on a stopped
batch, **Retry failed** also re-runs the failed rows, **Stop** stops it, **Open** shows a
finished row in the viewer.

## The MCP server

For an agent working in another repository (Auroconts, say):

```bash
pip install "mcp>=2.3,<3"
```

Then, in that repository's `.mcp.json`:

```json
{"mcpServers": {"assetfurnace": {
  "command": "C:/path/to/AssetFurnace/.venv/Scripts/python.exe",
  "args": ["-m", "image_to_3dlab.mcp_server"],
  "cwd": "C:/path/to/AssetFurnace"}}}
```

| Tool | What it does |
|---|---|
| `validate_batch(spec)` | Checks a spec and its installs; runs nothing |
| `submit_batch(spec, retry_failed)` | Starts `pipeline.py --run-batch` in the background and answers at once |
| `resume_batch(batch_ref, retry_failed)` | Starts a stopped batch again with the spec and intent it ran with |
| `batch_status(batch_ref)` | Counts and per-row status, with the reason for failures. `batch_ref` is the batch folder or its spec |
| `list_outputs(batch_ref, row_id)` | Finished rows and their files, as absolute paths on this machine |
| `cancel_batch(batch_ref)` | Stops the runner; the row in progress goes back in the queue |
| `list_steps()` | Step names and which exist yet |

`validate_batch` and `submit_batch` also take `use_case`, `distribution` and `commercial`
for a CSV spec. It speaks stdio, uses the official MCP Python SDK (`MCPServer`, version 2.x) and never
downloads weights: a spec that needs a missing install comes back with the problems.
