"""Tests for the FLUX.2 klein bootstrap: everything that happens before a byte moves."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "bootstrap_flux2_klein", SCRIPTS / "bootstrap_flux2_klein.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


boot = _load()


def test_announcement_names_backend_route_size_and_licence():
    text = boot.announcement()
    assert "FLUX.2 klein 4B" in text
    assert "stable-diffusion.cpp" in text
    assert "12.3 GB" in text
    assert "Apache-2.0" in text
    for _, filename, _ in boot.WEIGHTS:
        assert filename in text


def test_no_file_comes_from_the_non_commercial_flux2_dev_repo():
    """FLUX.2-dev's VAE would put a non-commercial file into an Apache-2.0 route."""
    assert all(repo != "black-forest-labs/FLUX.2-dev" for repo, _, _ in boot.WEIGHTS)


def test_nothing_downloads_without_a_yes(monkeypatch, capsys):
    monkeypatch.setattr(boot, "binary_present", lambda: True)
    monkeypatch.setattr(boot.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    monkeypatch.setattr(boot, "install_weights", lambda: (_ for _ in ()).throw(AssertionError))
    assert boot.main([]) == 1
    assert "Nothing downloaded" in capsys.readouterr().out


def test_a_non_interactive_run_refuses_rather_than_hanging(monkeypatch, capsys):
    monkeypatch.setattr(boot, "binary_present", lambda: True)
    monkeypatch.setattr(boot.sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(boot, "install_weights", lambda: (_ for _ in ()).throw(AssertionError))
    assert boot.main([]) == 1
    assert "Refusing" in capsys.readouterr().out


def test_yes_proceeds_without_asking(monkeypatch):
    calls = []
    monkeypatch.setattr(boot, "binary_present", lambda: True)
    monkeypatch.setattr(boot, "install_weights", lambda: calls.append("weights"))
    monkeypatch.setattr("builtins.input", lambda _: (_ for _ in ()).throw(AssertionError))
    assert boot.main(["--yes"]) == 0
    assert calls == ["weights"]


def test_a_fresh_install_gets_the_shared_sd_cli_first(monkeypatch, capsys):
    """Set up on a machine without sd-cli used to stop at "install Qwen-Image first": the
    studio's button for FLUX did nothing until another route had been installed."""
    calls = []
    monkeypatch.setattr(boot, "binary_present", lambda: False)
    monkeypatch.setattr(boot.qwen, "target", lambda: "windows-nvidia")
    monkeypatch.setattr(boot.qwen, "install_binary", lambda: calls.append("sd-cli"))
    monkeypatch.setattr(boot, "install_weights", lambda: calls.append("weights"))
    assert boot.main(["--yes"]) == 0
    assert calls == ["sd-cli", "weights"]
    out = capsys.readouterr().out
    assert "prebuilt sd-cli" in out and "~850 MB" in out  # announced before anything moved


def test_without_sd_cli_and_no_prebuilt_for_this_machine_nothing_is_fetched(monkeypatch, capsys):
    monkeypatch.setattr(boot, "binary_present", lambda: False)
    monkeypatch.setattr(boot.qwen, "target", lambda: "some-other-machine")
    monkeypatch.setattr(boot.qwen, "install_binary", lambda: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(boot, "install_weights", lambda: (_ for _ in ()).throw(AssertionError))
    assert boot.main(["--yes"]) == 1
    assert "Nothing downloaded" in capsys.readouterr().out


def test_without_a_yes_not_even_sd_cli_is_fetched(monkeypatch, capsys):
    monkeypatch.setattr(boot, "binary_present", lambda: False)
    monkeypatch.setattr(boot.qwen, "target", lambda: "windows-nvidia")
    monkeypatch.setattr(boot.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    monkeypatch.setattr(boot.qwen, "install_binary", lambda: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(boot, "install_weights", lambda: (_ for _ in ()).throw(AssertionError))
    assert boot.main([]) == 1


def test_every_file_fetched_is_one_the_catalogue_counts_at_its_exact_size():
    import backend_catalog

    counted = {(w.source, w.filename): w.bytes_expected
               for w in backend_catalog.BY_ID["flux2-klein"].weights}
    assert counted == {(repo, filename): size for repo, filename, size in boot.WEIGHTS}
