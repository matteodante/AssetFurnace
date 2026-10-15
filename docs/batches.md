# Batches: many assets, one command

    python pipeline.py --run-batch manifests/my-batch.json     # or my-batch.csv

A batch is a list of assets, one row each. Every row runs the same steps with the same
checks, gets its own `.provenance.json`, and a stopped batch picks up where it left off
when you run the same command again. An agent in another repository can drive the same
thing through the [MCP server](#the-mcp-server).

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
| `steps` | In order. Default `generate`, `finish` |
| `backend` | Default `pixal3d`, the only generator batches run today (see below) |
| `parameters`, `seed` | Generator settings: `seed`, `steps`, `gss`, `gsh`, `fov`, `matte` for Pixal3D |
| `finish`, ... | Settings for one step, under its name: `"finish": {"atlas": 1024}` |

`defaults` fills any field a row leaves out (`parameters` and step settings are merged).
`rows` can also name a CSV file: `"rows": "pilot-props.csv"`.

A CSV holds rows only, one per line, with `;` between several values in a cell and an
empty cell for the default:

```csv
id,images,kind,faces,profile,steps,backend,seed
SM_Barrel_01,refs/barrel.png,prop,3000,unreal,generate;finish,pixal3d,42
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
build and weights, Blender) and the batch stops with the command to install it. SF3D and
TRELLIS.2 fetch weights on first use, so they are not offered in batches yet.

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

`outputs.json` is the hand-off for whatever reviews the assets next (the review queue of
function 7, the studio's library): it lists only finished rows, with their kind, profile,
final asset, its provenance and every file the steps wrote.

## Stopping and resuming

Run the same command again. Then:

- a **done** row is skipped;
- an **interrupted** row starts again from its first step whose files are not all still
  there with the recorded hash, so a finish that crashed does not cost a generation;
- a **failed** row stays failed (its reason is in the state) unless you add
  `--retry-failed`: a row that always fails should not take the GPU on every resume;
- a row you **edited** in the spec runs again; its old files stay on disk;
- a changed **intent or licence policy** refuses to resume: rename the batch or remove its
  folder.

One batch runs at a time per output folder: the GPU is the bottleneck, and a second one
is refused while the first is alive. The studio does not check for running batches, so
do not generate there at the same time.

Exit codes: 0 every row done, 1 some rows failed, 2 the batch cannot start, 130
interrupted.

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
| `batch_status(batch_ref)` | Counts and per-row status, with the reason for failures. `batch_ref` is the batch folder or its spec |
| `list_outputs(batch_ref, row_id)` | Finished rows and their files, as absolute paths on this machine |
| `cancel_batch(batch_ref)` | Stops the runner; the row in progress goes back in the queue |
| `list_steps()` | Step names and which exist yet |

It speaks stdio, uses the official MCP Python SDK (`MCPServer`, version 2.x) and never
downloads weights: a spec that needs a missing install comes back with the problems.
