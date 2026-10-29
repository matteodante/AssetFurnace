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
| `rigged.blend` + `rig_info.json` | AnyMove's rig builder (`tools/prep.sh`, `tools/rig.sh`) |
| `reader.pt`, the trained joint reader | AnyMove's renders and trainer (about 45 min on one RTX 4090, per AnyMove), or its release |
| a two-camera clip + its first frame `ref.png` | a video model (AnyMove used MiniMax H3 on fal) |
| Blender 4.2+ | found the same way Finish finds it |

Reading needs `transformers` 4.48 or newer (`uv pip install transformers`). A reader file holds its
whole network, so reading downloads nothing.

## Run it

```bash
# all at once: read the joints, fit the rig, export the GLB
python scripts/twocam_mocap.py capture clip.mp4 reader.pt rig_info.json rigged.blend out/ --ref ref.png

# or step by step
python scripts/twocam_mocap.py read clip.mp4 reader.pt rig_info.json out/det.npz --ref ref.png
python scripts/twocam_mocap.py check out/det.npz rig_info.json    # are the two views in step?
python scripts/twocam_mocap.py fit out/det.npz rig_info.json out/motion.json
python scripts/twocam_mocap.py apply rigged.blend rig_info.json out/motion.json out/

# no video at all: fit a made-up motion and score it in centimetres
python scripts/twocam_mocap.py exam rig_info.json
```

`check` says `good`, `doubtful` or `bad`. `fit` refuses a `bad` clip (two halves showing
different motions cannot make one true motion) unless you pass `--force`.

Every step leaves a licence record: `det.provenance.json` (video, reader, rig, with
hashes), copied into `motion.json`, and `<name>.provenance.json` beside the GLB, which
also carries the model's own licence.

## Tested on

AnyMove's dragon roar (release `examples-v1`), on Windows 11 with an RTX 5070 Ti and
Blender 5.2: 30 s from clip to GLB (reading 15 s on the GPU, fitting 18 s on the CPU).
The joints match AnyMove's own run to 0.01 cm on average; Blender's joints match the
solver's to 0.1 mm.

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

## Not ported (yet)

Rendering the training poses and training a reader, the rig builder, and video
generation. They still run in AnyMove.

## Licences

- AnyMove's code is Apache-2.0, like ours: ported with credit (see `NOTICE`).
- [video2rig](https://github.com/canitcode/video2rig), @CanItCode's own tool, is
  **AGPL-3.0**. None of its code is here: copying it would put this whole repo under the
  AGPL. Only ideas were taken.
- The clip's terms are the video model's. Declare them with `--made-with` and
  `--video-license`; they go into the records.
