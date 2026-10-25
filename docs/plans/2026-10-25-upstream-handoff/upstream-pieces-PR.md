Title: feat: In pieces — one subject drawn taken apart, split into pieces

Adds an **In pieces** goal (`kit`) to Create and to MCP (`create(want="kit")`). It draws one character, monster or object as an exploded view, builds it in one Pixal3D run and splits it with the prop-sheet pipeline in a new `--pieces` mode of `scripts/blender_split_props.py`:

- one shared tilt for every piece (the median of each piece's best tilt), no turns;
- parts join only where their meshes come within 1% of the sheet size, not when their boxes overlap; small parts (down to 0.3% of the faces) are kept;
- pieces keep their scale, named `piece_01`, `piece_02`... (`--names` on the command line);
- `props.json` records each piece's place in the straightened sheet and, for each neighbouring pair, the nearest points as assembly hints; `props.blend` keeps the exploded layout with `SOCKET_<piece>_<neighbour>` empties. They are hints, not joints;
- `--turn` is refused in pieces mode, since it would move a piece across the sheet.

Prop sheets are unchanged, apart from one small fix: a prop or piece with fewer triangles than a LOD level skips that level.

The pure maths (shared tilt, layout, sockets, neighbour tree, grouping) is unit-tested. **The Blender path and the prompt still need a real GPU run**: 3 subjects (humanoid, four-legged monster, object) × 3 seeds — <results here>.

Tests: 2 failed before and after (already on `main`), 3683 passed. Ruff is clean.
