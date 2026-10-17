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
| `prompt` | Words to draw the picture from, for a row whose first step is `picture` (instead of `images`) |
| `replace` | Steps whose output is a file of your own: `{"finish": "edits/knight.glb"}` (a list of GLBs for `animate`). The step does not run; the file goes on to the next one |
| `asset` | An existing GLB, for a row that skips `generate` |
| `kind` | `prop` (default), `creature` or `equipment` |
| `faces` | Face budget for `finish`, 1,000 to 200,000 (default 40,000) |
| `profile` | Output profile, kept with the row and its outputs |
| `size_cm` | The asset's real size in centimetres, for `export_unreal` and `checks` |
| `measure` | What `size_cm` measures: `height` (default) or `longest` side |
| `steps` | In order. Default `generate`, `finish`, `checks` |
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
| `picture` | Draws the row's picture from its `prompt` with Qwen-Image 2.1 Turbo, the studio's Create engine (`scripts/text_image.py`); comes first. With `"count": 4` it draws four (seeds `seed`, `seed`+1...) and the row **waits for a pick** |
| `concept` | A new concept drawn from the row's picture (feature 2: redraw, dress, add_piece, other_view, complete_piece), as the studio's "Draw from a reference" and `scripts/reference_concept.py make`. Before `generate`, which then draws from the concept, the row's images following as references. Candidates and picks as for `picture` |
| `generate` | Writes the row's run manifest and runs it, exactly as `pipeline.py --run-manifest` would |
| `finish` | The studio's Finish (`scripts/retopo_repaint.py`) at the row's face budget, with Pixel Match when the model kept its camera |
| `rig` | A skeleton and skin weights (SkinTokens and the weight clean-up, as the studio's Animate tab; `scripts/autorig.py rig`). The rigged GLB becomes the row's asset. Holds the GPU lock while SkinTokens runs |
| `clips` | A rigged creature's family and its test clips (feature 6, `scripts/creature_clips.py`): the CC0 library's carried by chains, procedural ones for the rest, longer GenMM takes unless `"variations": false`. After `rig`; the clips reach a creature's Unreal export |
| `animate` | Preset clips played on the rig (`"animate": {"presets": ["idle", "walk"]}`, `speed`, `arm_spread`; `scripts/autorig.py animate`). CPU only. The row's asset stays the rig; `export_unreal` takes the clips as a creature's animations |
| `export_unreal` | The row's asset as Unreal 5.8 files (`scripts/export_unreal.py`), into `rows/<id>/export_unreal/`. Needs `size_cm`. The row's asset stays the finished model |
| `checks` | The automatic checks of the review queue (`scripts/check_asset.py`): the row reaches the queue with its report and can be approved there |
| `materials` | Reserved name: skipped with a warning until it exists |

`finish` does **not** repaint: the repaint needs Hunyuan 2.1 Paint weights, whose licence
excludes the EU, UK and South Korea, and runs on Apple Silicon only. A row that wants it
says `"finish": {"paint": true}`. Its other settings: `photo` (Pixel Match, on by
default), `atlas`, `texture_size`, `metallic`, `roughness`, `ior`, `voxel`, `skip_bake`,
`skip_compress`.

`export_unreal` settings: `name` (default the row's id), `collision` (`box`, `convex`,
`unreal`; objects), `lods` (files, `[]` for none; default the `_LODn` files beside the
model), `tint` (`false` for none), `forward` (three numbers, Blender axes), `body` and `body_name`
(equipment, which needs its body), `animations` (creatures). File paths are relative to
the spec. `checks` settings: `profile` (default from `kind`), `set` (thresholds, as
`--set`: `{"budget.faces.max": 20000}`), `reference` (compare with the row's picture,
default true), `height`, `width`, `depth` in metres (default `size_cm` when it is a
height), `stop_on_fail` (a failed check fails the row; by default the reviewer decides).
Every step's settings are described once, with their types and ranges, in
`image_to_3dlab/step_catalog.py`; a misspelt or out-of-range one stops the batch before
its first row.

`picture` settings: `seed` (default 42), `count` (1 to 8), `pick` (the seed to keep),
`width`, `height`, `cfg_scale`, `sampler`, `negative_prompt`.

`concept` settings: `mode` (required), `shape` and `style_refs` (files; the shape default
is the row's picture), the words the mode needs (`subject` and `colours` for redraw,
`outfit` for dress, `piece`, `view`...), `structure`, `guide`, `depth`, `style`, `seed`,
`count` (1 to 4), `pick`.

**Your own file instead of a step's output.** `"replace": {"<step>": "file"}` in a row: the
step does not run, the file (checked when the spec loads: a whole GLB, a skeleton for
`rig`, a readable picture for `picture` and `concept`) is copied into the step's folder
under the name the step would give it, with its own `.provenance.json` (a Blender version
has one) or a record saying it is yours and its licence is not known, and the steps after
it run on it. Its records are held to the batch's licence policy like any source.
Changing the file, or the record beside it, reruns from that step; the steps before keep
their results. A file inside the batch's own `rows/<id>/` folder is refused (the batch
clears and rewrites it): copy an edit made there somewhere else first.
`generate`, `finish`, `rig`, `animate`, `picture` and `concept` can be replaced; the
others make nothing a file can stand in for.

**Picking.** A step that made several candidates leaves its row `waiting`; the rest of the
batch goes on and the run ends with exit code 3. Pick with `"pick": <seed>` in the step's
settings, or with the MCP tool `pick_candidate` (it writes `picks.json` in the batch
folder and resumes). The candidates are kept in `rows/<id>/picture.candidates/`, so a pick,
or another pick later, draws nothing again; it only reruns what follows. On the studio's
Batches page a waiting row shows its candidates with a Pick button, and every row has
"Run again…" to rerun one of its steps (`rerun_step`). A row whose steps only draw
(`["picture"]`, `["picture", "concept"]`) needs no model: its picture is listed in
`outputs.json` under `picture`.

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
    export_unreal/SM_<Name>.glb, .unreal.json, .provenance.json (T_<Name>_Tint.png)
    checks/<id>.checks.json
```

`outputs.json` lists only finished rows, with their kind, profile, size, final asset, the
Unreal export (`unreal`) and checks report (`checks`) when those steps ran,
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
twice, a new one is, and so is the same version when it now comes with its report (adding
`checks` to a finished batch makes its rows approvable). `equipment` rows are queued with the checks' `equipment_part` kind.
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
- a row you **edited** in the spec runs again **from its first step whose settings
  changed**: a new face budget reruns `finish` and what follows, not the generation; a
  new `size_cm` reruns `export_unreal` and `checks`; a step added at the end runs alone; a
  new seed or picture reruns everything. Each step keeps a fingerprint of what it ran
  with (its settings, the row fields it reads, and every step before it). Old files stay
  on disk;
- a changed **intent or licence policy** refuses to resume: rename the batch or remove its
  folder.

One batch runs at a time per output folder: the GPU is the bottleneck, and a second one
is refused while the first is alive. The lock is `<output folder>/.batch.lock`, taken
atomically, so two submissions at once cannot both start; one left by a dead process is
taken over. The studio does not check for running batches, so
do not generate there at the same time.

Exit codes: 0 every row done, 1 some rows failed, 2 the batch cannot start, 3 some rows
wait for a pick (none failed), 130 interrupted.

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
| `review_queue()`, `review_item(key)` | What waits for review, the latest approvals; one item with its notes, history and checks report |
| `check_asset(glb, key, profile, set, reference, height, width, depth)` | The checks on a GLB, as `scripts/check_asset.py`: report beside it, item in the queue. `set` takes thresholds: `{"budget.faces.max": 20000}` |
| `review_decide(key, decision, asset_sha256, note)` | `approved`, or `sent_back` with a note; the version you looked at, as the queue gives it. Also a verdict in `verdicts.jsonl` |
| `pieces_list()`, `pieces_add_body(...)` | Characters in pieces: base bodies, sets, slots; register a rigged humanoid as a body |
| `pieces_run(body, set, mode, slots, outfit, concept, weapon, shield, style, seeds, choose)` | A set on a body, in the background, as the studio's Pieces page starts it; `choose` stops after the pictures |
| `pieces_status()`, `pieces_pick(body, set, picks)` | The run's log and the pictures waiting (paths and scores); go on with your picks |
| `pieces_fit`, `pieces_refit(body, set, to)`, `pieces_unreal(body, set, size_cm)`, `pieces_cancel()` | Fit again, carry a set to another body, export it for Unreal, stop the run |
| `describe_steps()` | Every step with its settings (types, defaults, ranges), what can be skipped or replaced, what makes candidates, and whether it is installed |
| `run_steps(name, steps, prompt, images, asset, ..., options)` | One asset through some steps: a batch of one row, written to `output/batches/.agent/<name>.json` and started. Call it again with other settings to rerun from the first changed step |
| `pick_candidate(batch_ref, row_id, step, choice)` | Pick one of the candidates a waiting row offers (`batch_status` lists them under `waiting`), or change a pick; resumes the batch |
| `rerun_step(batch_ref, row_id, step)` | Run one step of a row again, and what follows, with the same settings |

`validate_batch` and `submit_batch` also take `use_case`, `distribution` and `commercial`
for a CSV spec. The review and pieces tools call the same code as the studio and the
command line (`checks/review.py`, `scripts/review_queue.py`, `slot_parts/runs.py`); one set
is made at a time on the machine, whether the studio or an agent started it
(`output/sets/.run.json`). It speaks stdio, uses the official MCP Python SDK (`MCPServer`, version 2.x) and never
downloads weights: a spec that needs a missing install comes back with the problems.
