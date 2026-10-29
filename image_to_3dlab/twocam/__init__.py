"""Two-camera motion capture: animate any rigged creature from one AI video.

A video model films the creature from two orthographic cameras in one frame (front on
the left, side on the right). A joint reader finds every joint in both halves; a joint
sits on the same row in both, so the pair pins it in 3D, and the rig is fitted to those
joints frame by frame.

The method is @CanItCode's. This package is a port of the capture half of Patch's
AnyMove (https://github.com/Patch-gg/anymove, Apache-2.0, see NOTICE), with these
changes:

* the layout comes from the rig's own `rig_info.json`, never from an environment
  variable (AnyMove silently used the wolf's scale for any creature run without
  `MOCAP_CREATURE`, putting a dragon's hips at 0.33 m instead of 0.86 m);
* left/right limb pairs, feet and the ground term are found from the rig, not from the
  wolf's joint names, so `fix_swaps` no longer crashes on a spider;
* a locked joint axis (limits `[0, 0]`) no longer turns the whole fit into NaN;
* rotation conversions stay correct near half a turn;
* it runs on CUDA, or on the CPU (float64 is not available on Apple's MPS).

`layout`, `solve`, `exam` and `motion` need only numpy and torch. `reader` also needs
OpenCV and `transformers`, imported only when a video is read.
"""
