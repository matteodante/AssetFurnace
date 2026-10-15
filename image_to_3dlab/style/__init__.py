"""Style bibles: approved pictures, prompt templates and style references for a project.

    from image_to_3dlab.style import bible, prompts, refs

    root = bible.create(path, "my-game", wording="bold saturated colours")
    bible.add(root, "concept.png", "Colours and proportions are right.", kind="creature")
    prompt = prompts.compose("a forest troll", "creature", bible=root, references=2)["prompt"]
    references = refs.pick(root, "creature", k=2)

A bible is a self-contained folder (see `bible.py`), so it can live in the game's own
repository. AssetFurnace only keeps a list of the bibles it knows (`registry.py`).
"""
