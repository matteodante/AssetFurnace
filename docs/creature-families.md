# Creature families: rig, animate and export creatures that are not humanoids

The auto-rig (SkinTokens) gives any model a skeleton. What a skeleton is for depends on the
creature, so after rigging AssetFurnace works out which **family** it belongs to and gives
it clips that suit it: a spider walks on eight legs, a serpent slithers, a drake flaps.

| Family | For | Clips |
|---|---|---|
| Humanoid | people, orcs, robots | the 21 preset moves, as before |
| Two legs, digitigrade | werewolves, demons, raptors on two legs | idle, walk, attack, hit, death |
| Four legs | wolves, cats, horses | idle, walk, run, attack, death, and a trot test |
| Many legs | spiders, insects | idle, walk, attack, hit, death |
| Winged | dragons on four legs, birds on two | idle, walk, fly, glide, and hover and attack tests |
| Serpent | snakes, worms, basilisks | idle, slither, attack, hit, death |
| Plant | carnivorous plants, trees | sway and attack tests |

Every looping clip (idle, walk, run, fly, glide, slither) also comes as a `-long` take,
twice as long and made of the clip's own pieces in a new order, so a creature does not
idle in a two-second loop.

## In the studio

1. **Rig it.** In the Rig step, *What is it?* defaults to **Recognise it for me**: the
   family is read from the skeleton, from its legs, arms, wings, tail and how they meet the
   ground. Pick the family yourself when the guess is wrong; the Animate step says which
   one was used and whether you chose it.
2. **Play the clips.** They are listed under the model like any other move. Each is a
   separate GLB beside the rig, `<model>_<family>-<move>.glb`, for example
   `wolf_digitigrade-walk.glb`.
3. **Check them.** The review queue runs `rig.deformation` on every clip. It measures
   stretched edges and flipped triangles, and whether the feet cross, and warns above 0.5%.

## Where the clips come from

- **The clip library** (optional, recommended): hand-made animations from
  [Mesh2Motion](https://github.com/Mesh2Motion/mesh2motion-app), dedicated to the public
  domain (CC0). Install it from **Setup & Status**, or:

  ```sh
  python scripts/bootstrap_clip_library.py
  ```

  Six files, about 4 MB, from one pinned commit, each checked against its SHA-256 before it
  is kept. A fox drives four-legged creatures, a kaiju digitigrade ones, a spider the
  many-legged, a snake the serpents, and a dragon or a bird the winged. Each clip is
  carried onto your rig chain by chain (`scripts/chain_retarget.py`). Your creature keeps
  its own proportions and rest pose, and its feet stay planted where the source's do.
- **Procedural test clips** (`scripts/creature_motion.py`) for the motions the library does
  not have, and for every family when it is not installed. They exist to prove a rig, not
  to ship.
- **The `-long` takes** come from a numpy port of
  [GenMM](https://github.com/wyysf-98/GenMM) (Apache-2.0), with no model and no download.

Every clip's `.provenance.json` names where its motion came from: the CC0 source clip with
its file hash and commit, or "procedural test clip", and the variation method and seed for
a `-long` take.

From the command line, after rigging:

```sh
python scripts/creature_clips.py --rig out/wolf_rigged.glb            # recognise, then write the clips
python scripts/creature_clips.py --rig out/wolf_rigged.glb --family quadruped
python scripts/creature_clips.py --rig out/wolf_rigged.glb --no-variations          # no -long takes
python scripts/creature_clips.py --rig out/wolf_rigged.glb --no-library             # procedural only
python scripts/creature_clips.py --rig out/wolf_rigged.glb --no-clips               # the family file alone
python scripts/creature_clips.py --rig out/wolf_rigged.glb --variation-factor 3 --variation-seed 7
python scripts/chain_retarget.py --list                                             # the library's clips
python scripts/chain_retarget.py --rig out/wolf_rigged.glb --source fox --clip walk --longer
```

Run them with the Python that runs SkinTokens (`vendor/SkinTokens/.venv`), which has Blender's
`bpy` (`--list` needs none). `chain_retarget.py` carries one library clip onto the rig.

**Switching them off.** In the studio, the Rig step has "Family test clips" and "Longer
takes", both on by default: untick the first for no clips (the family file is still written,
for the Unreal export's facing), the second to leave out the GenMM takes. In a batch, the
clips are a step of their own, `clips`, after `rig` (`"clips": {"family": "quadruped",
"variations": false}`; settings `family`, `clips`, `library`, `variations`,
`variation_factor`, `variation_seed`), and leaving it out of the row's steps skips them; an
`animate` step after it adds its clips to the family's. Agents reach it through `run_steps`.

## Into Unreal

Export the rig as a **Creature** (studio: Export for Unreal; or
`scripts/export_unreal.py --kind creature`). Every clip beside the rig goes along, as
`AS_<Name>_<Family><Move>`, on the creature's own skeleton. A creature that is not a
humanoid faces where its family file says, so a drake generated facing half-way between
two axes is turned to face forward in Unreal, and it is sized after that turn. A humanoid,
digitigrade ones included, is renamed like the UE5 Mannequin, as before.

*Check in Unreal* (`scripts/unreal_import_check.py`) measures a humanoid's facing from
Manny's feet, and a creature that is not a humanoid from its hips to its head (a serpent:
tail to head), named in the export's `.unreal.json`.

## Known limits

- The recogniser can leave a chain without a role, such as one leg of a spider whose mesh
  fuses two legs. That chain stays still while the others move, and the skin around it
  stretches. Pick the family by hand, or check the clips in the review queue.
- Attack, hit and death clips are extreme poses. They can pass the warning threshold on the
  library's own models too: compare with the source before judging a rig by them.
- Plants have no library clips, only the procedural tests.
- A werewolf's thick legs can pass through each other when Manny's walk is retargeted onto
  it in Unreal's IK Retargeter: its joints never cross, its leg meshes do. In its own
  digitigrade clips the feet never cross (`rig.deformation` checks it). In the retargeter,
  widen the stance by hand; this is not automated yet.

The design, the measurements and every decision are in
[docs/plans/2026-10-15-creature-families-design.md](plans/2026-10-15-creature-families-design.md).
