"""Item exclusion lists (CSV with columns item_id, reason).

Used to drop MILU items that do not test medical knowledge (government
initiatives, non-medical, current affairs); see data/excluded_items.csv.
"""

import csv
import hashlib
from pathlib import Path


def load_exclusions(path: Path) -> dict[str, str]:
    """Return {item_id: reason}. Raises ValueError on missing columns or duplicate ids."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"item_id", "reason"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
        out: dict[str, str] = {}
        for row in reader:
            iid = row["item_id"].strip()
            if iid in out:
                raise ValueError(f"{path}: duplicate item_id {iid!r}")
            out[iid] = row["reason"].strip()
    return out


def apply_exclusions(items: list[dict], excluded: dict[str, str], key: str = "id") -> list[dict]:
    """Drop excluded items, keeping order. Every excluded id must match an item,
    so a typo or a changed content-hash id fails loudly instead of excluding nothing."""
    present = {it[key] for it in items}
    unknown = sorted(set(excluded) - present)
    if unknown:
        raise ValueError(f"{len(unknown)} excluded id(s) match no item, e.g. {unknown[:5]}")
    return [it for it in items if it[key] not in excluded]


def exclusion_sidecar_fields(paths: list[Path] | None) -> dict:
    """Config-sidecar keys recording which exclusion file(s) a run used."""
    if not paths:
        return {"exclude_file": None, "exclude_sha256": None}
    paths = [Path(p) for p in paths]
    return {"exclude_file": [p.name for p in paths],
            "exclude_sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]}
