Title: Security and robustness fixes: request guard, UTF-8, Windows processes

Fixes from our fork, rebuilt on 0.5.2. One commit per fix, each with a test.

## Security
- **A web page can no longer drive the studio behind your back.** Before, any page open in the browser could start jobs, remove weights or change the HF token in a running studio, and DNS rebinding could read the repository. Now a write needs `X-AssetFurnace-Request` (the studio) or `X-AssetFurnace-Client` (your MCP client), and a cross-site `Origin` is refused. The Host check applies only when the studio listens on loopback, so RunPod and LAN setups work as before. The MCP server and the plan runner keep working; the MCP route-parity test still passes.
- **Re-rigging never runs an uploaded .blend's own Python:** Blender opens it with `--factory-startup --disable-autoexec`.

## Windows and processes
- Text files are read and written as UTF-8 on every locale, and jobs talk UTF-8 on Windows, so a non-Latin prompt or name no longer breaks a job. Your earlier UTF-8 fixes covered scripts; this covers the studio APIs and job pipes.
- sd-cli and the other helpers stop when the studio does, even when it is ended from Task Manager (a Windows job object), so they no longer hold GPU memory.
- A damaged pid file can no longer signal pid 0 (the studio's own group) or -1 (every process you own).
- The Windows installer installs `requirements.lock`, with CUDA PyTorch at the locked versions, like `install.sh`.

## Finish
- The retopology is unwrapped without overlapping UV islands: about 10% overlap on two real models, now under 0.1%.

## Worth knowing
- Any other script that writes to the studio without one of the two headers now gets 403. Four tests that did were updated.
- `install.ps1` stops with a clear error if PyTorch's CUDA index has no build of the locked torch for the driver's CUDA version.
- The Blender UV test and the Windows job-object test are skipped without Blender or Windows.

## Tests
Before: 2 failed, 3644 passed. After: 2 failed, 3729 passed, 21 skipped. The 2 failures are already on `main` (`test_bootstrap_pixal3d` re-run, `test_viewer_skins_api` "made before skins"). Ruff is clean.
