# Two-camera motion capture

Animate a rigged creature (four legs, wings, a tail) from one AI video. The
humanoid moves in the studio come from Kimodo; this is for everything else.

**How it works.** A video model films the creature from two cameras in one frame: front
on the left, side on the right. A joint reader, trained on that one creature, finds every
joint in both halves. A joint sits at the same height in both, so the pair pins it in 3D.
The rig is fitted to those joints frame by frame and exported as a GLB.

The method is [@CanItCode](https://www.youtube.com/watch?v=AS92HYsEBqo)'s. The code is a
port of [Patch's AnyMove](https://github.com/Patch-gg/anymove) (Apache-2.0).

## What you need

| | Where it comes from |
|---|---|
| a rigged model | the studio's Animate step (SkinTokens), or any GLB with one armature |
| Blender 4.2+ | found the same way Finish finds it |
| an NVIDIA card with 16 GB for filming | MiniMax H3 runs here through stable-diffusion.cpp |

Downloads, each only with `--yes` (or a y typed at a terminal):

- DINOv2 with registers for `train` (Meta, Apache-2.0, 346 MB);
- MiniMax H3 for `film --download`: the pruned model, its text encoder and video VAE,
  about 30 GB, under the MiniMax H3 Community License. MiniMax asks users in the USA, the
  EU, the UK and South Korea to apply with the form on its model page first.

Reading needs `transformers` 4.48 or newer (`uv pip install transformers`).

## Run it

Once per creature:

```bash
# turned to face -Y, sized, left and right named, each limb painted; writes ref.png and preview.png
python scripts/twocam_mocap.py prepare wolf_rigged.glb wolf/ --length-m 1.6 --noun "grey wolf"
# 2,200 random poses from both cameras, then the joint reader learns them
python scripts/twocam_mocap.py dataset wolf/rigged.blend wolf/rig_info.json wolf/dataset
python scripts/twocam_mocap.py train wolf/dataset wolf/rig_info.json wolf/reader.pt --yes
```

Check `preview.png` after `prepare`: the creature faces you on the left and looks right on
the right, with its left limbs pink and yellow. `--yaw-deg` turns it by hand if the
symmetry search got it wrong.

Then per move:

```bash
# three takes on this machine, each read and judged; the best is named
python scripts/twocam_mocap.py film wolf/ "the wolf rears up and howls" clips/     --end "standing on all four legs" --takes 3 --reader wolf/reader.pt
python scripts/twocam_mocap.py capture clips/take1_s1.webm wolf/reader.pt wolf/rig_info.json     wolf/rigged.blend out/ --ref wolf/ref.png

# or step by step: read, check (are the two views in step?), fit, apply
python scripts/twocam_mocap.py exam wolf/rig_info.json   # no video: fit a made-up motion, score in cm
```

`check` says `good`, `doubtful` or `bad`, judged on the typical joint (a limb hidden in
one view is guessed there) and on whether moving feet agree. `fit` refuses a `bad` clip
unless you pass `--force`.

Every step leaves a licence record: each take's `.provenance.json` (video model, licence,
prompt, seed), `det.provenance.json` (video, reader, rig, with hashes), copied into
`motion.json`, and `<name>.provenance.json` beside the GLB, which also carries the model's
own licence (`prepare` copies it next to `rigged.blend`).

## In the studio

A rigged model that is not a humanoid gets **Animate from video** in its Animate step, in
place of the preset moves. Four stages, each one run of `scripts/twocam_mocap.py`:

1. **Prepare** (about 10 s): how long it is nose to tail, and what it is. The card shows the
   reference frame with its joints, left pink and right blue: check the sides and the facing.
2. **Teach the reader** (about an hour, once per creature). The button names DINOv2 (346 MB,
   Apache-2.0) when it still has to download it. It keeps going with the tab closed and can
   be cancelled; the practice poses already rendered are kept.
3. **Film**: what it does in one sentence, how it ends (optional), one to three takes (about
   6.5 minutes each). A missing video model is shown first (MiniMax H3, about 30 GB, its
   licence, and the form users in the USA, the EU, the UK and South Korea must fill in), with
   its own **Download the video model** button. Each take plays in the card with its verdict.
4. **Capture**: **Use this take** adds the move to the model. A bad take goes only through
   **Fit anyway**.

The move lands beside the presets as `output/animate/<run>/<name>_<move>.glb`, with its
`.provenance.json`. The rest stays in `output/animate/<run>/twocam/`: `creature/`,
`clips/<move>/`, `captures/<move>/` and `steps/run.log`. Agents run the script: the studio's
routes for this are not MCP tools.

## Tested on

Windows 11, an RTX 5070 Ti (16 GB), Blender 5.2:

- AnyMove's dragon roar (release `examples-v1`, filmed by H3 Max on fal): 30 s from clip
  to GLB. The joints match AnyMove's own run to 0.01 cm on average; Blender's joints
  match the solver's to 0.1 mm.
- The same dragon filmed here by MiniMax H3 (pruned, Q4): 1024 x 576, 3 s, in 6.5 min;
  judged doubtful (typical joint 2.8% apart against fal's 1.2%), fitted at 13-22 px.
- A wolf the studio made from a sentence (picture, Pixal3D, Finish, SkinTokens rig):
  prepared, filmed here, reader trained and captured. See the pull request for numbers.

## What changed from AnyMove

- **The scale comes from the rig.** AnyMove read it from an environment variable and fell
  back to the wolf's: a dragon fitted without it had its hips at 0.33 m, not 0.86 m.
- **Limbs come from the skeleton**, not from the wolf's leg names. `--fix-swaps` crashed
  on the spider; the floor and the feet score now work for every creature.
- **A locked joint axis** (limits `[0, 0]`) no longer turns every angle into NaN.
- **Left/right swaps are decided over the whole clip**, so one unclear frame cannot swap
  a leg (an idea from [video2rig](https://github.com/canitcode/video2rig); see below).
- **Turns past half a turn** stay smooth, and rotations near 180 degrees convert correctly.
- **A reader or detections for another rig are refused**, instead of putting every joint
  on the wrong bone.
- The fit runs on the CPU by default: faster than the GPU here, and the same on every machine.

## What works best

Big single gestures: rearing, roaring, slamming, taking off, pouncing. AnyMove found that
eight-legged walks do not fit (the two views disagree on which legs are up) and that lying
down does not either; neither was retried here.

## Not done (yet)

- A LoRA that teaches the video model the two-camera layout (@Graalitoo's idea in
  AnyMove, trained there on fal per creature). The plan is to train one here, for every
  creature, from clips rendered in Blender.
- The browser: the studio's Animate step still offers only the humanoid presets.

## Licences

- AnyMove's code is Apache-2.0, like ours: ported with credit (see `NOTICE`).
- [video2rig](https://github.com/canitcode/video2rig), @CanItCode's own tool, is
  **AGPL-3.0**. None of its code is here: copying it would put this whole repo under the
  AGPL. Only ideas were taken.
- The clip's terms are the video model's. Declare them with `--made-with` and
  `--video-license`; they go into the records.
