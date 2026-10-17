"""One material per piece (design v2 of game materials, section 11): proposed by a rule, corrected by a person.

Splitting an asset into material zones automatically did not pass (72% of the points), so each
piece of a character in pieces, and each single asset, gets one material from the label
vocabulary. The rule that proposes it reads like this:

1. the piece's own words (its prompt in the Pieces page, the weapon or shield words): the
   first material word in them;
2. else, for a piece of a set, the clause of the outfit that names the piece
   ("... and leather boots" -> boots: leather);
3. else the default of the slot (the slots draw armour: metal; a shield: wood), or of the
   asset's kind for a single asset.

A person corrects it: in Studio, or in the set's ``materials.json`` / the export's
``--material`` from the command line and agents. The chosen material sets the piece's
roughness and metallic (``PBR``), which the Unreal export packs into its ORM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from image_to_3dlab.material_labels import MATERIALS as LABEL_WORDS

MATERIALS = tuple(m for m in LABEL_WORDS if m != "other")
SCHEMA = "assetfurnace.piece-materials/1"
CORRECTIONS = "materials.json"

# Roughness and metallic per material, after MaterialSeg3D++'s material table (2026).
PBR = {
    "cloth": (0.9, 0.0), "leather": (0.6, 0.0), "metal": (0.35, 1.0), "skin": (0.55, 0.0),
    "fur": (0.9, 0.0), "hair": (0.6, 0.0), "feathers": (0.85, 0.0), "scales": (0.45, 0.0),
    "wood": (0.7, 0.0), "stone": (0.8, 0.0), "bone": (0.6, 0.0), "horn": (0.5, 0.0),
    "plant": (0.7, 0.0), "gem": (0.1, 0.0), "glass": (0.05, 0.0), "plastic": (0.45, 0.0),
}

# Words in a prompt that name a material. Garments are cloth.
WORDS = {
    **{m: m for m in MATERIALS},
    "steel": "metal", "iron": "metal", "plate": "metal", "plated": "metal", "chainmail": "metal",
    "mail": "metal", "gold": "metal", "golden": "metal", "silver": "metal", "bronze": "metal",
    "brass": "metal", "copper": "metal", "metallic": "metal", "chrome": "metal",
    "hide": "leather", "suede": "leather",
    "fabric": "cloth", "cotton": "cloth", "wool": "cloth", "woollen": "cloth", "woolen": "cloth",
    "linen": "cloth", "silk": "cloth", "velvet": "cloth", "cloak": "cloth", "cape": "cloth",
    "robe": "cloth", "robes": "cloth", "tunic": "cloth", "hood": "cloth", "scarf": "cloth",
    "sash": "cloth", "gown": "cloth", "dress": "cloth",
    "wooden": "wood", "oak": "wood", "timber": "wood", "bark": "wood",
    "furred": "fur", "pelt": "fur", "feather": "feathers", "feathered": "feathers",
    "scale": "scales", "scaled": "scales", "bones": "bone", "horns": "horn",
    "crystal": "gem", "jewel": "gem", "gems": "gem", "rock": "stone", "marble": "stone",
    "leaf": "plant", "leaves": "plant", "vine": "plant", "vines": "plant", "flesh": "skin",
}

SLOT_DEFAULTS = {"helmet": "metal", "torso": "metal", "shoulders": "metal", "gloves": "metal", "legs": "metal",
                 "boots": "metal", "weapon": "metal", "shield": "wood"}
# The nouns that name each slot in an outfit's words.
SLOT_NOUNS = {
    "helmet": ("helmet", "helm", "hood", "hat", "cap", "headpiece", "crown", "mask"),
    "torso": ("torso", "breastplate", "cuirass", "chest", "chestplate", "tunic", "jerkin", "vest", "coat", "robe",
              "armour", "armor", "doublet", "shirt"),
    "shoulders": ("shoulder", "shoulders", "pauldron", "pauldrons", "spaulders"),
    "gloves": ("glove", "gloves", "gauntlet", "gauntlets", "bracers", "vambraces"),
    "legs": ("legs", "leg", "greaves", "trousers", "pants", "leggings", "cuisses", "tassets", "skirt", "kilt"),
    "boots": ("boot", "boots", "shoes", "sabatons", "sandals"),
    "weapon": ("sword", "axe", "mace", "spear", "dagger", "club", "hammer", "staff", "weapon", "bow"),
    "shield": ("shield", "buckler"),
}
KIND_DEFAULTS = {"prop": "wood", "creature": "skin", "equipment": "metal", "character": "cloth"}
DEFAULT = "cloth"
ARTICLES = {"helmet": "a helmet", "torso": "a torso", "shoulders": "shoulders", "gloves": "gloves",
            "legs": "legs", "boots": "boots", "weapon": "a weapon", "shield": "a shield"}


def check(material: str) -> str:
    """`material` as a word of the vocabulary, or a ValueError naming the words."""
    word = str(material).strip().lower()
    if word not in MATERIALS:
        raise ValueError(f"the material is one of {', '.join(MATERIALS)}, not {material!r}")
    return word


def pbr(material: str) -> tuple[float, float]:
    """(roughness, metallic) of a material."""
    return PBR[check(material)]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text or "").lower())


def material_in(text: str) -> str | None:
    """The first word of `text` that names a material, as a material; None when none does."""
    for token in _tokens(text):
        if token in WORDS:
            return WORDS[token]
    return None


def _clause_for(slot: str, outfit: str) -> str | None:
    """The clause of the outfit's words that names `slot` and a material, if any."""
    nouns = set(SLOT_NOUNS[slot])
    for clause in re.split(r",|;|\band\b|\bwith\b|\bplus\b", str(outfit or "").lower()):
        if nouns & set(_tokens(clause)) and material_in(clause):
            return clause.strip()
    return None


def propose(*, slot: str | None = None, kind: str | None = None, words: str = "", outfit: str = "") -> dict:
    """The rule's material for a piece (`slot`) or a single asset (`kind`), and why, in words."""
    if slot is not None and slot not in SLOT_DEFAULTS:
        raise ValueError(f"unknown slot {slot!r}; the slots are {', '.join(SLOT_DEFAULTS)}")
    if kind is not None and kind not in KIND_DEFAULTS:
        raise ValueError(f"unknown kind {kind!r}; the kinds are {', '.join(KIND_DEFAULTS)}")
    found = material_in(words)
    if found:
        whose = "the piece's words say" if slot else "the prompt says"
        return {"material": found, "by": "rule", "reason": f"{whose} {found!r}"}
    if slot is not None:
        clause = _clause_for(slot, outfit)
        if clause:
            return {"material": material_in(clause), "by": "rule", "reason": f"the outfit says {clause!r}"}
        return {"material": SLOT_DEFAULTS[slot], "by": "rule", "reason": f"default for {ARTICLES[slot]}"}
    if kind is not None:
        return {"material": KIND_DEFAULTS[kind], "by": "rule", "reason": f"default for a {kind}"}
    return {"material": DEFAULT, "by": "rule", "reason": "default"}


RECORD_KEYS = ("prompt", "input", "source_record")


def _record(path: Path) -> dict | None:
    """A file's record: `<stem>.provenance.json`, else `<stem>.json` (what Pixal3D writes)
    when it reads as one."""
    for name in (path.stem + ".provenance.json", path.stem + ".json"):
        record = path.with_name(name)
        if not record.is_file():
            continue
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and any(key in data for key in RECORD_KEYS):
            return data
    return None   # no record, or not a provenance record: no words to read, never a crash


def prompt_for(path: Path, depth: int = 4) -> str:
    """The words a single asset was made from: the prompt in its provenance record, or in its
    input's, following the chain of inputs (model <- picture), and into the source's record a
    rig keeps (`source_record`). Empty when none is found."""
    path = Path(path)
    for _ in range(depth):
        data, source = _record(path), None
        for _ in range(depth):   # a record may hold its source's record inside it (a rig's)
            if not isinstance(data, dict):
                break
            if isinstance(data.get("prompt"), str) and data["prompt"].strip():
                return data["prompt"]
            given = data.get("input") if isinstance(data.get("input"), dict) else {}
            if isinstance(given.get("path"), str) and given["path"]:
                source = given["path"]
                break
            data = data.get("source_record")
        if not source:
            return ""
        path = Path(source)
    return ""


def piece_of(file_name: str) -> str | None:
    """The slot a fitted piece's file belongs to (``gloves_L.glb`` -> gloves); None for body files."""
    stem = re.sub(r"_[LR]$", "", Path(str(file_name)).stem)
    return stem if stem in SLOT_DEFAULTS else None


def _parts(folder: Path) -> dict:
    found = sorted(Path(folder).glob("*.parts.json"))
    if not found:
        raise FileNotFoundError(f"no <set>.parts.json in {folder}: the set is not finished")
    return json.loads(found[0].read_text(encoding="utf-8"))


def corrections(folder: Path) -> dict[str, str]:
    """The person's corrections saved for a set: {slot: material}."""
    path = Path(folder) / CORRECTIONS
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {slot: check(word) for slot, word in (data.get("pieces") or {}).items()}


def outfit_words(concept_prompt: str) -> str:
    """The outfit's own words inside the concept's edit prompt ("Add <outfit> to the character in image 1...")."""
    found = re.match(r"\s*Add (.*?) to the character in image 1", str(concept_prompt or ""), re.S)
    return found.group(1) if found else str(concept_prompt or "")


def set_materials(folder: Path) -> dict[str, dict]:
    """Every piece of the set in `folder` (``output/sets/<set>/<body>/``) with its material.

    A person's correction wins, then the material chosen before the run (``slots[slot].material``),
    then the rule.
    """
    parts = _parts(folder)
    outfit = outfit_words((parts.get("concept") or {}).get("prompt", ""))
    fixed = corrections(folder)
    out = {}
    for slot, asked in (parts.get("slots") or {}).items():
        piece = (parts.get("pieces") or {}).get(slot) or {}
        if slot in fixed:
            out[slot] = {"material": fixed[slot], "by": "person", "reason": "corrected by a person"}
        elif asked.get("material"):
            out[slot] = {"material": check(asked["material"]), "by": "asked", "reason": "chosen before the run"}
        else:
            words = asked.get("prompt") or (piece.get("prompt", "") if piece.get("held") else "")
            out[slot] = propose(slot=slot, words=words, outfit=outfit)
    return out


def save_corrections(folder: Path, changes: dict[str, str | None]) -> dict[str, str]:
    """Save a person's materials for some pieces (None clears one); returns all the corrections."""
    slots = set(_parts(folder).get("slots") or {})
    saved = corrections(folder)
    for slot, word in changes.items():
        if slot not in slots:
            raise ValueError(f"{slot!r} is not a piece of this set; its pieces are {', '.join(sorted(slots))}")
        if word is None:
            saved.pop(slot, None)
        else:
            saved[slot] = check(word)
    path = Path(folder) / CORRECTIONS
    staged = path.with_suffix(".json.tmp")
    staged.write_text(json.dumps({"schema": SCHEMA, "pieces": saved}, indent=1) + "\n", encoding="utf-8")
    staged.replace(path)
    return saved
