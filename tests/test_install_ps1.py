"""install.ps1 and the Windows guide: what can be read without running them.

Not in test_install_script.py, whose module is skipped on Windows, the one system that
runs install.ps1.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PS1 = REPO / "install.ps1"


def test_windows_installer_uses_the_same_lockfile_as_the_shell_one():
    ps1 = PS1.read_text()
    assert ps1.index('"requirements.lock"') < ps1.index("& uv pip install --quiet --python $Py -r $Install")
    assert "$Install = $Reqs" in ps1
    # CUDA torch at the lock's own versions, so the lock does not swap it for PyPI's CPU wheel
    assert 'Get-Pin $Reqs "torch"' in ps1 and 'Get-Pin $Reqs "torchvision"' in ps1
    assert '"torch==$TorchVersion"' in ps1 and '"torchvision==$VisionVersion"' in ps1


def test_windows_installer_defaults_to_this_repository():
    ps1 = PS1.read_text()
    assert '"https://github.com/matteodante/AssetFurnace.git"' in ps1
    assert "Bingeljell/image-to-3dlab.git" not in ps1


def test_windows_installer_picks_releases_of_the_repository_it_installs():
    """A checkout that once fetched the upstream's tags keeps them; the newest release must
    come from the repository being installed, or an update lands on upstream's v0.4.3."""
    ps1 = PS1.read_text()
    assert "git ls-remote --tags --refs $Repo" in ps1
    assert "git -C $Dir tag --list" not in ps1
    # install.sh finds it the same way (code read 18/10/2026: it still read the clone's tags)
    sh = (REPO / "install.sh").read_text(encoding="utf-8")
    assert 'git ls-remote --tags --refs "$REPO_URL"' in sh and "tag --list" not in sh


def test_windows_installer_run_from_a_clone_updates_that_clone():
    ps1 = PS1.read_text()
    assert "$PSScriptRoot" in ps1


def test_windows_guide_runs_the_installer_from_the_clone_not_the_upstream_site():
    guide = (REPO / "WINDOWS.md").read_text(encoding="utf-8")
    assert "assetfurnace.com/install.ps1" not in guide and "| iex" not in guide
    assert r".\install.ps1" in guide
