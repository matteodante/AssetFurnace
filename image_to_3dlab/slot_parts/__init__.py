"""Characters in pieces: armour slots worn on a rigged base body.

    from image_to_3dlab.slot_parts import body, cuts, masks, slots

    chosen = slots.choose({"helmet": {}, "shoulders": {"split": True}})
    record = body.load("output/bodies/knight/body.json")
    labels = masks.slot_labels(foreground, body.segments_px(record, concept_size), chosen=set(chosen))
    piece, box = cuts.cut_slot(concept_rgb, labels, "helmet")
    canvas, place = cuts.flatten(piece, box)        # the picture route C completes

The geometry here is plain numpy, with no Blender: Blender runs in the scripts it writes jobs
for. `candidates`, `orient` and `grip` ask the picture judge (on the GPU, when it is there) and
`runs` starts the set's processes. The design is
`docs/plans/2026-10-16-slot-parts-design.md`; the trials behind it are in
`docs/plans/2026-10-15-slot-parts-spike.md`.
"""
