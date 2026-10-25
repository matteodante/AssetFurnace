Title: Security and robustness fixes: request guard, UTF-8, Windows processes, pinned downloads

Fixes from our fork, rebuilt on 0.5.2. One commit per fix, each with a test.

## Security
- **Write requests need a header a web page can't send.** Before, any page open in the browser could start jobs, remove weights or change the HF token in a running studio, and DNS rebinding could read the repository. Now a write without `X-AssetFurnace-Request` (the studio) or `X-AssetFurnace-Client` (your MCP client) gets 403. A cross-origin `Origin` is refused. The MCP server and the plan runner keep working; the MCP route-parity test still passes.
- **A studio on a network address (RunPod, LAN) asks for an access token**, carried by the printed link. Loopback is unchanged.
- **An uploaded .blend never runs its own scripts:** re-rig opens it with `--factory-startup --disable-autoexec`.

## Windows and text
- Text files and job output are UTF-8 on every locale, so non-Latin prompts and names stop breaking jobs. Your earlier UTF-8 fixes covered scripts; this covers the studio APIs and job pipes.
- sd-cli and other helpers stop when the studio is killed (a Windows job object).
- A damaged pid file can no longer signal pid 0 or -1.
- The Windows installer installs `requirements.lock`, with CUDA PyTorch at the locked version, like `install.sh`.

## Data and downloads
- Library names, hidden flags, main versions and deletes write the meta file atomically, one change at a time, so two changes at once or a crash don't lose one.
- sd-cli archives (`master-975-95b3090`) are checked against SHA-256 before unpacking; Qwen weights come from pinned Hugging Face commits (the same files as today, so existing caches are reused).

## Finish
- Retopology UVs no longer overlap: about 10% overlapping islands down to under 0.1%.
- Finish starts Blender with `--factory-startup`, so a user's add-ons can't change the result.
- Props records write paths with `/`; auto-rig pins SkinTokens' `bpy` to 5.0.1; pytest keeps temp files only for failed tests.

## Worth knowing
- Any other script that posts to the studio without one of the two headers now gets 403. Four tests that did were updated.
- `install.ps1` stops with a clear error if PyTorch's CUDA index has no build of the locked torch for the driver's CUDA version.
- The Blender UV test and the Windows job-object test are skipped without Blender or Windows.

## Tests
Before: 2 failed, 3644 passed. After: 2 failed, 3753 passed, 21 skipped. The 2 failures are already on `main` (`test_bootstrap_pixal3d` re-run, `test_viewer_skins_api` "made before skins"). Ruff is clean.
