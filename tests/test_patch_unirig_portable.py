"""scripts/patch_unirig_portable.py: UniRig's skeleton model without flash-attn, safely re-runnable."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import patch_unirig_portable as pu

# upstream at 6793c66, word for word (no newline at the end of parse.py)
PARSE = '''from .unirig_ar import UniRigAR
from .unirig_skin import UniRigSkin

from .spec import ModelSpec

def get_model(**kwargs) -> ModelSpec:
    MAP = {
        'unirig_ar': UniRigAR,
        'unirig_skin': UniRigSkin,
    }
    __target__ = kwargs['__target__']
    del kwargs['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](**kwargs)'''
CONFIG = "llm:\n  do_layer_norm_before: True\n  _attn_implementation: flash_attention_2\n\nmesh_encoder:\n"


def _vendor(tmp_path: Path) -> Path:
    (tmp_path / "src" / "model").mkdir(parents=True)
    (tmp_path / "configs" / "model").mkdir(parents=True)
    (tmp_path / pu.PARSE).write_text(PARSE, encoding="utf-8")
    (tmp_path / pu.CONFIG).write_text(CONFIG, encoding="utf-8")
    return tmp_path


def test_the_skeleton_model_no_longer_imports_the_skin_model_or_asks_for_flash_attn(tmp_path):
    root = _vendor(tmp_path)
    assert pu.patch(root) == [pu.CONFIG, pu.PARSE]
    parse = (root / pu.PARSE).read_text(encoding="utf-8")
    tree = ast.parse(parse)
    top_imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert all(n.module != "unirig_skin" for n in top_imports)        # nothing at import time
    assert "from .unirig_skin import UniRigSkin" in parse              # still there for the skin model
    assert "_attn_implementation: sdpa" in (root / pu.CONFIG).read_text(encoding="utf-8")


def test_running_it_twice_changes_nothing_more(tmp_path):
    root = _vendor(tmp_path)
    pu.patch(root)
    before = [(root / f).read_text(encoding="utf-8") for f in (pu.PARSE, pu.CONFIG)]
    assert pu.patch(root) == []
    assert [(root / f).read_text(encoding="utf-8") for f in (pu.PARSE, pu.CONFIG)] == before


def test_an_upstream_that_moved_on_is_refused_not_half_patched(tmp_path):
    root = _vendor(tmp_path)
    (root / pu.PARSE).write_text("def get_model(**kwargs):\n    return None\n", encoding="utf-8")
    with pytest.raises(pu.PatchError):
        pu.patch(root)
    assert "flash_attention_2" in (root / pu.CONFIG).read_text(encoding="utf-8")   # nothing written
    assert pu.main([str(root)]) == 1
