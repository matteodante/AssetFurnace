"""Automatic rig for a generated creature: four legs, two legs and wings, a centaur, a spider.

A learned rigger draws several skeletons of the model; each is cleaned and scored on the
model's own voxels, the best one is kept and repaired, and its bones are named by body
plan so the two-camera fit knows a knee from a wing:

* `shell`     the watertight shell as voxels at 1% of the height: inside or out, depth,
              the middle plane, where the creature stands on the floor;
* `skeleton`  joint trees, named bones, and the centre lines of the voxels (the fallback
              skeleton, and where a missed leg comes from);
* `draws`     clean a drawn skeleton (folded fingers, branches outside the skin, ends
              pulled in), score it, choose the best;
* `body`      feet, head, tail, spine and limb pairs; the wing test; names by plan;
* `repairs`   missed legs, exact twins, the middle plane, soles, the head, an open jaw;
* `manylegs`  six or more legs, traced from the body;
* `pipeline`  `joints_from_draws` and `joints_without_unirig`, which write `joints.json`.

Kept free of imports on purpose: Blender's Python, which has numpy but not scipy, imports
`limits` from this package, and a package-level import of the other modules would break it.
"""
