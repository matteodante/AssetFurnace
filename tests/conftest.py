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

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

for directory in ("viewer", "scripts"):
    path = str(REPO / directory)
    if path not in sys.path:
        sys.path.insert(0, path)


import pytest


@pytest.fixture(autouse=True)
def _no_real_judge(monkeypatch):
    """No test starts the real picture judge (llama.cpp on the GPU) because it happens to be
    installed: a test that wants a judge gives its own."""
    from image_to_3dlab.style import judge, models

    def missing(hub=None):
        raise models.MissingModel("the picture judge is not used in tests")

    monkeypatch.setattr(judge, "find_parts", missing)


@pytest.fixture(scope="session")
def _suite_gpu_lock(tmp_path_factory):
    return tmp_path_factory.mktemp("gpu") / "assetfurnace-gpu.lock"


@pytest.fixture(autouse=True)
def _private_gpu_lock(_suite_gpu_lock, monkeypatch):
    """No test takes the machine's real GPU lock: a job of the user's would wait on the
    suite, or the suite on it, for up to an hour."""
    from image_to_3dlab import gpu_lock, text_image

    lock = _suite_gpu_lock
    monkeypatch.setattr(gpu_lock, "LOCK", lock)
    monkeypatch.setattr(text_image, "GPU_LOCK", lock)
