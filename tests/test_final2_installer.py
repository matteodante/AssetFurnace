"""Second final review, the Windows installer: a driver whose PyTorch index stops at an older
torch than the lock's installs its own lock, compiled from the same requirements with that
torch; and a checkout older than the lock (requirements.txt, no exact pins) installs instead
of failing on a missing match. The PowerShell part runs in PowerShell where there is one."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from image_to_3dlab import host

REPO = Path(__file__).resolve().parents[1]
PS1 = REPO / "install.ps1"
SHELL = shutil.which("pwsh") or shutil.which("powershell")
needs_powershell = pytest.mark.skipif(SHELL is None, reason="PowerShell is required to run install.ps1's functions")
# torch's own closure: what changes with it from one torch to another
WITH_TORCH = re.compile(r"^(torch|torchvision|triton|setuptools|nvidia-.*|cuda-.*)$")


def _pins(path: Path) -> dict[str, str]:
    return dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^ ;\n]+)", path.read_text(encoding="utf-8"), re.MULTILINE))


def test_the_cu128_index_has_a_lock_of_its_own():
    # download.pytorch.org/whl/cu128/torch/ has Windows cp311 wheels up to torch 2.11.0 and
    # torchvision 0.26.0 (read 2026-10-18); the main lock's torch is only on cu130
    main, cu128 = _pins(REPO / "requirements.lock"), _pins(REPO / "requirements-cu128.lock")
    assert (cu128["torch"], cu128["torchvision"]) == ("2.11.0", "0.26.0")
    assert main["torch"] != cu128["torch"]
    # the same packages tested together, but for torch's own: compiled starting from the lock
    assert {k for k in main if not WITH_TORCH.match(k)} == {k for k in cu128 if not WITH_TORCH.match(k)}
    assert all(main[k] == cu128[k] for k in main if not WITH_TORCH.match(k)), \
        "requirements.lock changed: compile requirements-cu128.lock again (requirements-cu128.txt says how)"
    assert "requirements.txt -c requirements-cu128.txt -o requirements-cu128.lock" in \
        (REPO / "requirements-cu128.lock").read_text(encoding="utf-8")


def test_every_index_the_installer_can_pick_has_a_torch_to_install():
    newest = host.TORCH_CUDA_BUILDS[0][1]
    for _, tag in host.TORCH_CUDA_BUILDS[1:]:
        assert (REPO / f"requirements-{tag}.lock").is_file(), tag
    assert not (REPO / f"requirements-{newest}.lock").exists()   # the main lock is for the newest


def _function(name: str) -> str:
    text = PS1.read_text(encoding="utf-8")
    start = text.index(f"function {name}")
    return text[start:text.index("\n}\n", start) + 3]


@needs_powershell
def test_an_older_checkout_s_requirements_read_as_no_pin(tmp_path):
    reqs = tmp_path / "requirements.txt"
    reqs.write_text("numpy>=1.26\ntorch>=2.4\ntorchvision\n", encoding="utf-8")
    script = (f"$ErrorActionPreference = 'Stop'\n{_function('Get-Pin')}\n"
              f"'[' + (Get-Pin '{reqs}' 'torch') + ']'\n"
              f"'[' + (Get-Pin '{REPO / 'requirements-cu128.lock'}' 'torchvision') + ']'")
    done = subprocess.run([SHELL, "-NoProfile", "-NonInteractive", "-Command", script],
                          capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["[]", "[0.26.0]"]


def test_the_installer_takes_the_driver_s_lock_and_its_torch():
    ps1 = PS1.read_text(encoding="utf-8")
    assert '"requirements-$Tag.lock"' in ps1
    assert 'Get-Pin $Own "torch"' in ps1 and 'Get-Pin $Own "torchvision"' in ps1
    # torch from the index first, then the same lock, so the lock finds it already there
    assert ps1.index("$Install = $Own") < ps1.index("--index-url $TorchIndex @Torch") \
        < ps1.index("& uv pip install --quiet --python $Py -r $Install")
    assert ".Matches.Groups[1]" not in ps1                     # a missing match no longer throws


@needs_powershell
def test_the_installer_parses():
    script = ("$e = $null; [void][System.Management.Automation.Language.Parser]::ParseFile("
              f"'{PS1}', [ref]$null, [ref]$e); $e | ForEach-Object {{ \"$($_.Extent.StartLineNumber): $($_.Message)\" }}")
    done = subprocess.run([SHELL, "-NoProfile", "-NonInteractive", "-Command", script],
                          capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0 and not done.stdout.strip(), done.stdout
