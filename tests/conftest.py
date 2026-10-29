"""Shared pytest setup: make the repo's own modules importable the ordinary way.

`viewer/` and `scripts/` are not packages and are not on Python's search path, so test
files had to hand-load the modules they exercise with `importlib`. That works, but it
quietly breaks Python's one-module-one-object rule: **every hand-load creates a new copy**.
When production code imports the same module normally, the test and the code under test
end up holding two different objects with the same name, and a monkeypatch applied to one
is invisible to the other.

That is not hypothetical. `tests/test_download_api.py` patched `backend_catalog` while
`download_api` held a different copy, so the "an unsupported machine is refused before
anything downloads" guard passed without refusing anything. It failed only in the file
order the alphabetical suite never takes, so it sat green for weeks.

Putting the two directories on the path here lets a test write `import download_api` and
get the same object everyone else has. Neither directory shadows a standard-library name
(checked: 7 and 116 modules, no clashes), so this is additive.

Existing files that hand-load are not wrong to do so where nothing else imports the same
module -- a private copy of a module no one shares cannot diverge from anyone. Migrate
them when convenient; migrate immediately if production code imports the same name.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

for directory in ("viewer", "scripts"):
    path = str(REPO / directory)
    if path not in sys.path:
        sys.path.insert(0, path)

# The suite never counts or sends usage from a real install (tests of it pass their own env).
os.environ["I3D_NO_USAGE"] = "1"


@pytest.fixture
def fake_program(tmp_path):
    """Make a program in tmp_path that prints `lines` and exits with `code`, whatever it is
    given. A `#!/bin/sh` script on Mac and Linux; Windows cannot start a `#!` file, so there
    it is a batch file (`name.cmd`), which Windows runs through cmd.exe. Keep cmd's special
    characters (`<>|&^%`) out of `lines`."""
    def make(name: str, lines: tuple[str, ...] = (), code: int = 0) -> Path:
        if os.name == "nt":
            program = tmp_path / f"{name}.cmd"
            body = ["@echo off", *(f"echo {line}" for line in lines), f"exit /b {code}"]
            program.write_text("\n".join(body) + "\n")   # text mode writes \r\n on Windows
        else:
            program = tmp_path / name
            body = ["#!/bin/sh", *(f"echo '{line}'" for line in lines), f"exit {code}"]
            program.write_text("\n".join(body) + "\n")
            program.chmod(0o755)
        return program
    return make
