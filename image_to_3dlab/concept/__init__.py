"""From a reference to a concept: a new picture that keeps a reference's shape, drawn from scratch.

    from image_to_3dlab.concept import edit, prompts, sidecar

    prompt = prompts.redraw("werewolf", "grey fur and pale claws", structure="depth")
    command = edit.with_references(text_to_image_command, [depth_map], edit.find_vision(hub))
    sidecar.write(image, mode="redraw", prompt=prompt, seed=11, command=command,
                  references=[(depth_map, "shape")])

Qwen-Image 2.1 in edit mode does the drawing: the text-to-image command of
`viewer/image_api.py` plus the vision half of its encoder (`--llm_vision`) and one `-r`
per picture. Shape pictures come first, then any style pictures, and the prompt names
them in that order. Measured on 16/10/2026 (docs/plans/2026-10-15-reference-concept-design.md):
the model keeps a silhouette almost to the letter, so a redraw gets its outline from a
structure map (depth or line drawing) and its colours in words; given the original
picture too, it copies it.
"""
