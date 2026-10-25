Title: Security and robustness fixes: request guard, UTF-8, Windows processes

Fixes from our fork, rebuilt on 0.5.2. One commit per fix, each with a test.

## Security
- **A web page can no longer drive the studio behind your back.** Before, any page open in the browser could start jobs, remove weights or change the HF token in a running studio, and DNS rebinding could read the repository. Now every write needs the studio's own origin and a request header a form on another site cannot send. The studio pages, the plan runner, your MCP client (`X-AssetFurnace-Client`) and `scripts/pod_smoke_test.py` send one; any other script that POSTs must send `X-AssetFurnace-Request`.
- A studio on 127.0.0.1 answers only loopback names. Behind tailscale serve, cloudflared, ngrok or VS Code port forwarding, list the name in `I3D_ALLOWED_HOSTS`. A studio bound to another address (`--host`, a pod) trusts its network and is not protected against DNS rebinding. **Not tested on RunPod.**
- **Re-rigging does not run the uploaded .blend's startup scripts, drivers or Rigify finalize script:** Blender opens it with `--factory-startup --disable-autoexec`, and the metarig's finalize script is cleared before generating.

## Windows and processes
- Text files are read and written as UTF-8, and jobs talk UTF-8 on Windows: non-ASCII text from `prompt_recipes.json` no longer comes out garbled, and a non-Latin prompt no longer breaks a job.
- sd-cli and the other helpers stop when the studio does, even when it is ended from Task Manager (a Windows job object).
- A damaged pid file can no longer signal pid 0 (the studio's own group) or -1 (every process you own).
- The Windows installer installs `requirements.lock`, with CUDA PyTorch at the locked versions, like `install.sh`. If that build can't be installed it warns and installs as before.

## Finish
- The retopology is unwrapped without overlapping UV islands: about 10% overlap on two real models, now under 0.1%. Islands are packed 0.002 apart instead of 0.0002, since there are about twice as many.

## Not verified here
No Blender, Windows or PowerShell in the test environment: the UV unwrap (with a fallback for Blender 5.0's UV selection), the job object and `install.ps1` were not run. The Blender version behind the 10% figure isn't recorded.

## Tests
Before: 2 failed, 3644 passed. After: 2 failed, 3750 passed, 21 skipped. The 2 failures are already on `main` (`test_bootstrap_pixal3d` re-run, `test_viewer_skins_api` "made before skins"). Ruff is clean.
