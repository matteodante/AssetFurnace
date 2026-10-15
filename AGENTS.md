# Repository Guide

How to communicate in this repo: Keep your answers short. No jargon - if we have to use abbriviations or jargon
we must explain it. We simplify everything. We don't over think or over explain. Brevity is important. Humour and wit is acceptable. 

Local Apple Silicon **image → 3D** pipeline wrapping four backends (SF3D, TRELLIS.2, and
two Hunyuan3D-MLX paths — see `docs/hunyuan-mlx-recipes.md`, not the old ComfyUI
route) behind one CLI, with license provenance as a first-class concern.

1. Commits and PRs must not include any co-authorship trailer — Claude, Codex, whatever.
2. **Only commit a script if it generalises.** Anything written for one asset, one
   debugging session or one render stays local — a scratch directory, never `scripts/`.
   Before committing one, take the asset out of it: no hard-coded object, image or
   material names, no colours or measurements that suit only the creature it was written
   for. Turn those into arguments, and it earns its place. And if it is worth committing
   it is worth finding, so list it in `scripts/README.md`; `tests/test_scripts_registry.py`
   fails if you do not.


## Layout

| Path | What lives here |
|------|-----------------|
| `pipeline.py` | CLI entry point (sets MPS env, delegates to `image_to_3dlab.cli`) |
| `image_to_3dlab/` | The package: `cli.py`, `provenance.py`, and one `*_backend.py` per backend |
| `manifests/` | Versioned run manifests (schema v1) — the preferred, traceable way to run |
| `scripts/` | Every command-line tool: generation, repair, texture, measurement, Blender, vendor patches. **`scripts/README.md` indexes all of them** and a test fails if a new script is not listed there |
| `workflows/` | ComfyUI API-format workflow JSON for the Hunyuan `--quality` path |
| `tests/` | pytest suite — one `test_*.py` per script or module it covers; run it with `PYTHONPATH=. pytest -q` |
| `journal/` | Investigation logs and session history (git-ignored — local only, not part of the shipped repo) |
| `hunyuan_mlx/` | Xiong's Hunyuan3D-MLX shape+paint port (MIT) — **tracked in-repo**, moved out of `vendor/` 2026-08-19 so a clone alone has the code. `shape/` and `paint/` each need `uv sync`; `weights/` under each is git-ignored, fetched via `download_weights.py`. No patch-reapply dance needed here — fixes are just part of the tracked source |
| `vendor/` | Vendored backend checkouts — **git-ignored**, cloned by the bootstrap scripts (or manually, for `hunyuan-mlx`). `trellis-mac` is a clone of `shivampkumar/trellis-mac` (~1.1 GB of code, weights and compiled Metal kernels); `hunyuan-mlx` is dgrauet's shape port, kept vendored on purpose since it's Tencent-licensed code, not just weights (see `docs/info_and_credits.md`). Ignored because these are someone else's repos at multi-GB scale; the cost is that patches vanish on re-bootstrap, so they live in `scripts/patch_*.py` — except `hunyuan-mlx-paint`, retired 2026-08-19 once its code moved to `hunyuan_mlx/` |
| `characters/` | Per-creature rigs, animations and their tests, one folder each — **git-ignored**. The pipeline is the product; individual creatures are our own content. Scripts here add `scripts/` to `sys.path` explicitly, since shared helpers such as `blender_joint_markers.send` still live there. Run their tests with `pytest characters/` |
| `output/` | Generated assets + `.provenance.json` sidecars (git-ignored) |
| `videos/`, `assets_to_test/` | Promo-video project and backend-comparison meshes — **git-ignored** (2026-09-19). Both are local working material; together they were 223 MB of a 244 MB repo |

## Commit conventions

Use **[Conventional Commits](https://www.conventionalcommits.org)**, one logical change per commit:

```
<type>: <imperative summary>

<optional body: what and why, wrapped ~72 cols>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `build`, `chore`.

- Keep commits small and self-contained — a commit should build and pass tests.
- Separate refactors from behavior changes.
- **No LLM co-author or attribution trailers.** Plain, human-authored messages only.

## Changelog

Follow **[Keep a Changelog](https://keepachangelog.com)** in `CHANGELOG.md`.

- Every user-facing change adds a line under `## [Unreleased]`, grouped by
  `Added` / `Changed` / `Fixed` / `Removed` / `Security`.
- On release, rename `[Unreleased]` to the version + date and open a fresh `[Unreleased]`.
- Versioning is **SemVer**. Provenance schema and manifest schema bumps are breaking.

## Development

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=. pytest -q
ruff check .
python pipeline.py --help
```

- Target Python 3.10/3.11 (not 3.14 — SF3D's native deps don't fit yet).
- Backends that load real models are exercised manually; keep them out of the unit suite.
- Prefer running via a manifest (`--run-manifest manifests/...json`) so runs stay traceable.

## Testing (write the test first — one minute now saves fifteen)

**Every code addition gets a test.** Generation runs cost 15-20 minutes and Blender steps
are interactive, so a defect found by running the real thing is expensive; the same defect
found by a unit test is nearly free. A missing one-line import once cost a full 16-minute
generation run that crashed on its very last statement.

Three rules, each learned the hard way:

1. **Test the real artifact, never a re-derived copy.** A test that re-extracts code from
   a file, or re-implements the logic it is checking, tests something that is not what
   ships. One such test reported a failure that did not exist, costing more time than no
   test at all. Import the function; do not `exec` a copy of it.
2. **Extract logic so it can be imported.** Anything embedded in a string that is sent to
   Blender, or injected by a patch script, is unreachable by tests. Pull the pure parts
   (geometry maths, file writers, validation) into module-level functions and test those;
   leave only the thin `bpy` calls in the string.
3. **Verify the environment before the expensive step, not after.** Check that an operator
   exists, an import resolves, a flag is honoured — these take seconds. Discovering them
   after a 16-minute generation is a self-inflicted wound.

For patch scripts specifically: assert the anchor is present, assert re-running is
idempotent, and **never assume the host file's imports** — patched code must import what
it uses.

## Provenance & licensing (do not weaken)

- Every run emits a `.provenance.json` sidecar recording the licence of every model it used.
  That record is the guardrail and travels with the file; output folders are just
  organisation (3D runs still sort by licence class, generated images all go to
  `output/images/`). `output/` is git-ignored: it holds our own work, not the product.
- `validate_run_policy` gates generation on declared intent — keep it ahead of model work.
- **BRIA RMBG-2.0 must never be loaded by this repo's own generation pipeline** (the vendored
  `trellis-mac`/`trellis-space-mac` backends) — the TRELLIS backend **refuses to run** unless
  the BRIA-disable patch is present. This is a license guardrail; never bypass it there.
  It does **not** block unrelated work that happens to touch BRIA-adjacent code: a one-off
  CUDA control run against the pristine upstream TRELLIS.2 repo (never shipped, never
  redistributed) is fine as long as BRIA itself is never actually downloaded/loaded/used —
  stub the eager constructor instead of requesting gated access, rather than treating "BRIA"
  as a word that halts all work near it.

### Weights are never downloaded without an explicit choice

**No script, bootstrap, viewer button or agent tool may download model weights until the
user has confirmed which pipeline and which route they want.** Nobody should discover
50 GB on their disk because they ran a setup step that sounded generic.

This is a licensing rule as much as a disk-space one: the Hunyuan weights are not licensed
in the EU, UK or South Korea, so "fetch everything, sort it out later" can put a user in
breach on our behalf. It also matters for agent use; an agent driving this repo must be
able to state what it is about to fetch and how large it is, and stop there.

Concretely, a download path must: name the backend, name the route, state the size, and
require an affirmative answer. `--yes` for non-interactive use is fine; defaulting to yes
is not.

## Quality checks on images are learned, never pixel heuristics

Decided by the user on 2026-10-15: deterministic checks on brightness, edges (Canny),
saturation, palette distances or colour clustering "do not work well" and are not used for
style, copying, material zones or baked light. Image-level judgements come from learned
models (embeddings such as SigLIP 2 / CSD / DINOv2, vision-language judges, segmenters).
A check ships only if it beats a test against the user's own labels (ROC AUC ≥ 0.90, at
least 80% of bad assets caught, at most 10% of good ones flagged); otherwise it is left out,
not kept as "informational". Geometric checks on meshes (faces, holes, UVs, stretch,
interpenetration, silhouette IoU on cut-out masks) are fine.

Before proposing a model or tool, check that it is the most recent approach that fits, and
record its release date and the licence of both code and weights (including weights the
code downloads by itself).

## Engines stay inside this program

AssetFurnace is standalone: engines live in `vendor/` (images: `vendor/sdcpp/sd-cli`),
their weights are `WeightSet`s in `viewer/backend_catalog.py`, downloaded only after
consent. Do not add ComfyUI or another external server as a dependency.

## Shared GPU

Agents share one GPU through `%TEMP%\assetfurnace-gpu.lock` (pid and session name inside).
Create it exclusively before GPU work and delete it afterwards, also on error. Hold it only
while a job actually computes on the GPU — never across installs, downloads, compiles or
waits. Release it after every job, even if your next job is queued; after a job longer than
20 minutes, let waiting sessions go first. If the lock belongs to a dead process, it may be
removed.

## Code review

Pull requests may get one round of review from the Codex bot (`@codex review`); it is
optional. Fix only real P1 findings, after checking the cited lines; record the rest in the
"Seguiti" section of the feature's design doc. Do not wait inside one long turn for a
review: request it, end the turn, and resume on the notification.
