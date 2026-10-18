#!/usr/bin/env python3
"""Serve the studio, and the repo so it can load GLBs by relative path.

    python viewer/serve.py                      # http://127.0.0.1:8777/viewer/studio.html
    python viewer/serve.py --auto               # what ./lab runs: picks how to listen, opens the browser
    python viewer/serve.py --host 100.x.y.z      # explicit Tailscale/LAN interface

Bound to 127.0.0.1 by default. An explicit --host exposes the whole repository, including
`output/`, on that interface, so a server on any address other than loopback asks for an
access token: the printed link carries it (set ASSETFURNACE_TOKEN to choose it; otherwise
a new one is made at every start; a chosen one is at least 24 letters, digits, '-' or '_').
Scripts send it as `Authorization: Bearer <token>`, which every endpoint takes. On such a
server a batch spec, and the files it names, are read from the repository only.

The viewer exists because judging a mesh by eye is the acceptance test in this repo, and
every such judgement previously required opening Blender. It is also driveable by browser
automation, which means the agent can look at its own output instead of asking.
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import signal
import socketserver
import sys
import webbrowser
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

# Sibling import: works when run as `python viewer/serve.py` (script dir on sys.path)
# and when loaded by the test suite via importlib (script dir not on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_api
from blender_edit_api import local_address
from generate_api import OUTPUT_ROOT, Handler, _reconcile_orphaned_jobs, _terminate_active_job

from image_to_3dlab import processes

REPO = Path(__file__).resolve().parents[1]


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Allow SSE and static assets while the one generator job runs."""

    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address) -> None:
        # A browser that closes a tab or cancels a download mid-response is normal, not an
        # error: say nothing rather than print a traceback into the terminal.
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


class LaunchPlan(NamedTuple):
    host: str
    open_browser: bool
    message: str


# what a cookie value carries unchanged (no ; , space or quote), and long enough to guess at
TOKEN_SHAPE = re.compile(r"[A-Za-z0-9_-]{24,}")


def access_token(host: str, env: Mapping[str, str]) -> str | None:
    """None on loopback, where only this computer can connect. On any other address the
    whole repository would be open to the network (RunPod's proxy is public), so a token.
    A chosen one goes into a cookie as it is: one that a cookie would cut short, or a short
    one, is refused at start rather than locking the page out after its first redirect."""
    if local_address(host):
        return None
    chosen = env.get("ASSETFURNACE_TOKEN")
    if chosen and not TOKEN_SHAPE.fullmatch(chosen):
        raise SystemExit("ASSETFURNACE_TOKEN must be at least 24 letters, digits, '-' or '_' (no spaces, ';', ',' "
                         "or quotes); leave it unset and the server makes one")
    return chosen or secrets.token_urlsafe(32)


def launch_plan(env: Mapping[str, str], port: int = 8777, token: str | None = None) -> LaunchPlan:
    """How `./lab` (serve.py --auto) should listen, for a new user who just ran the installer.

    On their own machine: privately, and open the browser. On a RunPod pod: on all
    interfaces, because RunPod's HTTP proxy is the only way in, and print that link. Over
    plain SSH: still privately, with the tunnel command, because listening on a network by
    default would expose the whole repository to everyone on it.
    """
    local = f"http://127.0.0.1:{port}/viewer/studio.html"
    pod = env.get("RUNPOD_POD_ID")
    if pod:
        return LaunchPlan("0.0.0.0", False, (
            f"Open https://{pod}-{port}.proxy.runpod.net/viewer/studio.html"
            f"{f'?token={token}' if token else ''}\n"
            f"(the pod needs HTTP port {port} exposed; add it in the pod's settings if not)"))
    if env.get("SSH_CONNECTION") or env.get("SSH_CLIENT"):
        return LaunchPlan("127.0.0.1", False, (
            f"This is a remote machine. On your own computer, run\n"
            f"    ssh -L {port}:127.0.0.1:{port} <this machine>\n"
            f"then open {local}"))
    return LaunchPlan("127.0.0.1", True, f"Opening {local}")


JOURNAL = REPO / "journal"


def write_pid_file(port: int, folder: Path = JOURNAL) -> Path:
    """`journal/server-<port>.pid`: `<pid> <port>`, where Ferma-AssetFurnace.ps1 looks."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"server-{port}.pid"
    path.write_text(f"{os.getpid()} {port}", encoding="utf-8")
    return path


def remove_pid_file(path: Path) -> None:
    """Only this server's: another one may have started on the port since."""
    try:
        if path.read_text(encoding="utf-8").split()[0] == str(os.getpid()):
            path.unlink()
    except (OSError, IndexError):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1", help="interface address to bind")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--auto", action="store_true",
                        help="pick how to listen for this machine (what ./lab uses)")
    args = parser.parse_args()
    plan = launch_plan(os.environ, args.port) if args.auto else None
    if plan is not None:
        args.host = plan.host
    generate_api.ACCESS_TOKEN = token = access_token(args.host, os.environ)
    if plan is not None and token:
        plan = launch_plan(os.environ, args.port, token)

    url = f"http://{args.host}:{args.port}/viewer/studio.html" + (f"?token={token}" if token else "")
    print(url, flush=True)

    # A prior server life may have died mid-generation (crash, closed terminal) without
    # ever recording how that job ended, and its detached child can still be running with
    # nobody tracking it -- resolve those before accepting new jobs.
    if OUTPUT_ROOT.is_dir():
        orphaned = _reconcile_orphaned_jobs(OUTPUT_ROOT)
        if orphaned:
            print(f"reconciled {len(orphaned)} orphaned job(s) from a previous session: "
                  f"{', '.join(orphaned)}", flush=True)

    # A deliberate stop (Ctrl-C, `kill`) must not orphan an in-flight job the way an
    # uncaught crash does -- SIGTERM has no default Python handler (it would just kill this
    # process outright, mid-job, with no chance to clean up), so both signals get one here.
    def _handle_shutdown_signal(signum, frame):
        _terminate_active_job()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)

    # Windows: sd-cli, llama-server and the workers die with this server, also when it is
    # killed from Task Manager (TerminateProcess runs none of the handlers above).
    processes.kill_children_with_this_process()

    import functools
    handler = functools.partial(Handler, directory=str(REPO))
    with ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        pid_file = write_pid_file(args.port)
        if plan is not None:
            print(plan.message, flush=True)
            if plan.open_browser and not args.no_browser:
                webbrowser.open(f"http://127.0.0.1:{args.port}/viewer/studio.html")
        print(f"serving {REPO} — ctrl-c to stop", flush=True)
        try:
            httpd.serve_forever()
        except (KeyboardInterrupt, SystemExit):
            print("\nstopped")
        finally:
            remove_pid_file(pid_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
