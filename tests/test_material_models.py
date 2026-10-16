import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "viewer"))

import backend_catalog as bc  # noqa: E402
import bootstrap_materials as boot  # noqa: E402
import download_api  # noqa: E402

from image_to_3dlab import material_models as mm  # noqa: E402


def test_every_model_is_pinned_and_states_its_licence_and_size():
    for model in mm.MODELS.values():
        assert len(model.revision) == 40 and model.size > 0
        assert model.license and model.license_url.startswith("https://")


def test_marigold_is_fetched_in_half_precision_only():
    for model in (mm.MARIGOLD_LIGHTING, mm.MARIGOLD_APPEARANCE):
        weights = [f for f in model.files if f.endswith((".safetensors", ".bin"))]
        assert weights and all(f.endswith(".fp16.safetensors") for f in weights)


def test_a_missing_model_names_the_command_that_installs_it(tmp_path):
    with pytest.raises(mm.MissingModel, match="bootstrap_materials.py --only marigold-lighting"):
        mm.require(mm.MARIGOLD_LIGHTING, tmp_path)


def test_a_complete_snapshot_is_found_at_its_pinned_revision(tmp_path):
    folder = mm.model_folder(mm.MARIGOLD_LIGHTING, tmp_path)
    for name in mm.MARIGOLD_LIGHTING.files:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"x")
    assert mm.require(mm.MARIGOLD_LIGHTING, tmp_path) == folder


def test_the_environment_can_live_outside_the_checkout(monkeypatch, tmp_path):
    monkeypatch.setenv("I2L_MATERIALS_ENV", str(tmp_path / "env"))
    assert mm.env_dir() == tmp_path / "env"
    with pytest.raises(mm.MissingModel, match="--only env"):
        mm.require_env()


def test_the_licence_conditions_carry_the_use_restrictions():
    text = " ".join(mm.OPENRAIL_CONDITIONS)
    assert "medical advice" in text and "law enforcement" in text and "outputs" in text


def test_baked_light_removal_is_a_tool_in_the_catalogue_with_its_licence():
    backend = bc.BY_ID["materials-light"]
    assert backend.kind == "tool"
    assert "Open RAIL++-M" in backend.license_name
    assert [w.source for w in backend.weights] == [mm.MARIGOLD_LIGHTING.repo]
    assert backend.bytes_expected == mm.MARIGOLD_LIGHTING.size


def test_the_setup_button_installs_only_what_the_card_lists():
    command = download_api.command_for("materials-light", host=bc.NVIDIA)
    assert command[-3:] == ["--only", "env,marigold-lighting", "--yes"]


def test_the_announcement_names_size_licence_and_source():
    text = boot.announcement(["marigold-lighting", "qwen35"])
    assert "2.40 GB" in text and "8.70 GB" in text
    assert "Open RAIL++-M" in text and "Apache-2.0" in text
    assert "huggingface.co/prs-eth/marigold-iid-lighting-v1-1" in text and "total 11.11 GB" in text


def test_nothing_is_fetched_without_a_yes_when_nobody_can_answer(monkeypatch):
    fetched = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert boot.main(["--only", "marigold-lighting"], fetcher=fetched.append) == 1
    assert fetched == []


def test_sam3_is_skipped_until_its_owner_has_requested_access(capsys):
    fetched = []
    code = boot.main(["--only", "sam3", "--yes"], access=lambda repo: False, fetcher=fetched.append)
    assert code == 1 and fetched == []
    assert "personal data" in capsys.readouterr().out


def test_with_a_yes_each_chosen_model_is_fetched(monkeypatch, tmp_path):
    monkeypatch.setenv("I2L_MATERIALS_ENV", str(tmp_path / "env"))
    fetched, made = [], []
    assert boot.main(["--only", "env,marigold-lighting", "--yes"], fetcher=fetched.append,
                     env_maker=lambda: made.append(True)) == 0
    assert fetched == [mm.MARIGOLD_LIGHTING] and made == [True]


def test_the_environment_gets_torch_from_its_own_index_first(tmp_path):
    commands = boot.env_commands("uv", tmp_path, tmp_path / "python", "https://download.pytorch.org/whl/cu130")
    assert commands[0][:2] == ["uv", "venv"]
    assert "torch==2.14.1" in commands[1] and commands[1][-2:] == ["--index-url",
                                                                   "https://download.pytorch.org/whl/cu130"]
    assert "diffusers==0.40.0" in commands[2]


def test_an_unknown_part_is_refused():
    with pytest.raises(SystemExit, match="unknown part"):
        boot.parts("marigold")
