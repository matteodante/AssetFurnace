# Draw from a reference: a new concept with a reference's shape

In the studio, **+ Create** → **Draw from a reference**. You get a new picture that keeps
the outline and pose of a model or a picture you already have, drawn from scratch. It
runs on this machine, with Qwen-Image 2.1 Turbo in edit mode (`sd-cli` with the vision
file, from Setup). Why it works the way it does: [the design](plans/2026-10-15-reference-concept-design.md).

## The four ways

| Way | Start from | What you write | What you get |
|---|---|---|---|
| Redraw a reference | A model (its depth is used) or a picture (its lines are used) | What it is, its colours, optionally the parts to redesign | A new design in the same outline and pose |
| Dress the base body | The base body model (its render, or its pose map) | The outfit | The body in its exact T-pose, dressed |
| Add a piece to the body | The base body model or picture | The piece | That piece on the body; the rest unchanged |
| Another view | The front picture of a creature, plus its model if you have one | Side or back | The same creature from that side |

Colours are written, not shown. Given the original picture, the model copies it, so a
redraw sends only the outline map. With the model, another view follows that side's
outline. Without it, the back view sometimes draws the front again; take 2-4 tries and
keep the best.

## Each try

The sheet prepares the map (the first time a model is used, Blender renders its views on
the CPU, about 20 s), then runs each try through the ordinary picture queue, about
25 s each on an RTX 5070 Ti. Every result shows the shape picture beside it and the
**outline IoU**, how well its silhouette matches (1.00 is the same outline). That number
is the only measurement; whether the picture is good is your call. **Keep** or
**Discard** saves your judgement.

Optional **style** words and up to three style pictures are painted from, never copied.

## What is written

- The picture, under `output/images/`, with its `.provenance.json` (licence, settings,
  the references by name and SHA-256).
- `<picture>.concept.json`: the way, the prompt, the seed, each reference with its role
  (shape or style), the inputs, the weight files, the IoU and your judgement.
- `output/concept/maps/`: the maps made from your models and pictures, kept for reuse.
- `output/concept/uploads/`: pictures you uploaded, as PNG.
- `output/concept/judgements.jsonl`: every Keep and Discard.

## From a script or another step

    python scripts/reference_concept.py prepare wolf/wolf.glb depth --view front
    python scripts/reference_concept.py make redraw --shape concept/maps/<hash>/front_depth_black.png \
        --subject werewolf --colours "grey fur, a brown mane and pale claws" --attempts 2

Paths are relative to `output/`. From Python, `viewer/concept_api.generate(prompt,
references, seed)` runs one edit and returns the picture's path: shape pictures first,
then style pictures, in the order the prompt names them.
