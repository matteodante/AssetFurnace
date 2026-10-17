"""Style bibles: approved pictures, prompt templates and style references for a project.

    from image_to_3dlab.style import bible, prompts, refs

    root = bible.create(path, "my-game", wording="bold saturated colours")
    bible.add(root, "concept.png", "Colours and proportions are right.", kind="creature")
    prompt = prompts.compose("a forest troll", "creature", bible=root)["prompt"]

The prompt alone carries the style. Style references (`refs.pick`, `references=` in
`compose`) are off unless asked for: in an image edit they also pass on their content
(every creature took the first reference's head in the 2026-10-16 trial), so they are kept
for redrawing one given object (function 2), not for new concepts.

A bible is a self-contained folder (see `bible.py`), so it can live in the game's own
repository. AssetFurnace only keeps a list of the bibles it knows (`registry.py`).
"""
