# Edit a Library model in Blender

Select a model in Studio, then use **Edit in Blender** beside the preview.

1. Select the version or original clip you want to edit and press **Open in Blender**.
   Studio captures that GLB and its provenance, then opens a separate Blender workspace.
2. Edit the model, skeleton or weights in Blender. Save the workspace to keep unfinished
   work. To return to Studio, use **File → Export → glTF 2.0**, choose **glTF Binary (.glb)**,
   include textures and enable the animations you want to retain. Export to a new file.
3. Press **Choose exported GLB…**, then **Import as new version**. The Version selector
   shows the result and lets you return to the original or another revision. Metadata
   and Provenance links record the source, hashes, date and Blender exporter information.

The selected GLB is the editing source. Separate animation files are not bundled into
it automatically. A returned GLB contains only what you exported from Blender: include
the armature and required objects, and check the exporter animation settings. Preview
the returned clips and inspect the rig before using it in a game.

## Local and remote Studio

Opening a desktop app requires Studio and the browser on the same computer, using a
localhost address, and Blender installed. The usual installation locations and
`I2L_BLENDER` are supported. Nothing installs Blender or downloads model weights.

On a remote/headless server, or without Blender installed, **Download editing copy**
captures and downloads the source. Import this GLB into your local Blender, edit and
export, then upload the result to the same editing copy in Studio. The normal Library
download is also available, but capturing an editing copy first preserves its exact
parent even if the original later changes.

Serve remote Studio only on your existing trusted network or behind authentication.
This feature uses the server's existing access model; it does not add user accounts.

## What is preserved, and what needs checking

- Original files are never overwritten. A version stores the uploaded GLB byte for byte,
  its parent version and SHA-256 hashes, and a copy of the available source license record.
  Additional source sidecars are retained too; `source_sidecars` lists their hashes and
  records (the primary record references `source_record` to avoid duplicating ancestry).
  Editing does not change the source license or establish rights to added content.
- Previous clips, LODs, collision files, Unreal exports and pipeline results stay on the
  original. They are not attached to a changed mesh or skeleton. Generate a new Unreal
  delivery separately when needed.
- The steps go on from a version. With a version selected, the Steps panel shows the
  steps after the one it stands in for:
  - A version that kept its skeleton is a rig, so Animate runs on it. The first move copies
    it once into `output/animate/<name>__rig__<time>/`; it is not rigged again, and its
    record names the version.
  - A version without a skeleton stands in for its 3D model, or for its finished model when
    it was made from one. Finish and Rig run on its file.
  - What they make joins the same asset in the Library, followed by the version's SHA-256,
    and the Steps panel then shows **Original pipeline** with the result.
  - Redo is not offered on a version, because it would draw over the edit.
  - Outside the studio, a version's `model.glb` is a file like any other. On the command
    line, give it to `scripts/autorig.py`, `scripts/retopo_repaint.py` or
    `scripts/export_unreal.py`. In a batch or `run_steps`, use `"replace": {"finish":
    "output/blender-edits/versions/<id>/model.glb"}` (or `generate`, `rig`). Its
    `model.provenance.json` carries the licence.
- Studio previews the imported file's own materials, skeleton and embedded animations.
  Missing rigs, fewer animations/textures, or changed rig/node metadata produce notices.
  These comparisons are conservative and cannot establish animation compatibility.
- GLB has limits: Blender modifiers, constraints, drivers and editing history do not
  all travel through glTF. Keep the `.blend` workspace. Exporting can change triangulation,
  vertex splits, transforms or material representation. Studio does not compensate for
  these changes, retarget animation, or certify Unreal compatibility.
- Uploads must be self-contained glTF 2.0 binary files up to 512 MiB, with embedded
  PNG/JPEG/WebP/AVIF images. Draco, meshopt, KTX textures and unknown required extensions
  are refused. The active scene must contain a mesh; skinning is limited to four
  influences per vertex for Studio's preview. Use the exporter's bone-influence limit.
  Structural checks include references, buffers, finite values and memory
  limits. They are not full glTF conformance, image decoding or semantic rig validation.

## Recovery

Pending editing copies survive a page reload or Studio restart. Select one from
**Editing copy** to reopen its saved workspace or import its export. Save in Blender
before closing: Studio does not autosave unsaved Blender changes. Retrying an identical
upload produces the same version; a different export needs a new editing copy. Select
**New edit of the selected version** for another iteration.

Files live under `output/blender-edits/`:

```text
sessions/<id>/input/source.glb          captured parent
sessions/<id>/input/source.*.json       captured provenance/sidecars
sessions/<id>/work/model.blend          editable workspace
sessions/<id>/session.json             source identity and hashes
sessions/<id>/blender.log              opening errors
versions/<id>/model.glb                unchanged uploaded bytes
versions/<id>/model.provenance.json     inherited license record and manual-edit facts
versions/<id>/version.json             version metadata
```

Publication uses a staging directory and an atomic rename. Disk failures leave hidden
`.pending-*` directories instead of a half-published version; these can be inspected and
removed manually when no import is running. There is no automatic cleanup of workspaces.
Back up this folder with your original output. A version remains in the Library if its
original is later removed. Do not edit published version files in place.

Blender processes belong to the user and remain open when Studio stops. After restarting
Studio, close or save a previously open workspace before reopening it: process tracking
is local to the running server, so a restart cannot reliably detect that old window.
Run one Studio server against an output folder; locks coordinate threads within it.
