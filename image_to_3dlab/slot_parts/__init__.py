"""Characters in pieces: armour slots worn on a rigged base body (feature 4).

    from image_to_3dlab.slot_parts import body, cuts, masks, slots

    chosen = slots.choose({"helmet": {}, "shoulders": {"split": True}})
    record = body.load("output/bodies/knight/body.json")
    labels = masks.slot_labels(foreground, body.segments_px(record, concept_size), chosen=set(chosen))
    piece, box = cuts.cut_slot(concept_rgb, labels, "helmet")
    canvas, place = cuts.flatten(piece, box)        # the picture route C completes

Everything here is plain numpy: no Blender, no GPU. The design is
`docs/plans/2026-10-16-slot-parts-design.md`; the trials behind it are in
`docs/plans/2026-10-15-slot-parts-spike.md`.
"""
