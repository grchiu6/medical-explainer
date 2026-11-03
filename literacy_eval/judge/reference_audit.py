"""Reference audit: let a non-Hindi reader flag garbled HiMed-Trad reference items (design section 4).

    python -m literacy_eval.judge.reference_audit --export

Then open literacy_eval/reference/reference_audit.csv in Google Sheets and translate with the free formula
    question_en:        =GOOGLETRANSLATE(D2, "hi", "en")
    correct_option_en:  =GOOGLETRANSLATE(E2, "hi", "en")
    cot_en:             =GOOGLETRANSLATE(F2, "hi", "en")
(fill down all 60 rows). For each row set ref_suspect = y if the item is an INVALID DATASET SAMPLE:
  - the question is corrupted, incomplete, or outside the research scope, or
  - the correct answer looks like a garbled term (e.g. "सदु कोष" for "मृदु कोष्ठ"), or
  - the explanation does not support the marked answer,
and say why in `note`. Download as CSV over the same file. Flagged items are excluded from every primary
metric in the report (and listed there); agreement including them is kept as a robustness column.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from literacy_eval.analysis.clean import question_stem
from literacy_eval.judge.jobs import load_reference

AUDIT_CSV = Path(__file__).resolve().parents[1] / "reference" / "reference_audit.csv"
COLUMNS = ["item_id", "system", "subject", "question_hi", "correct_option_hi", "cot_hi",
           "question_en", "correct_option_en", "cot_en", "ref_suspect", "note"]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def export_audit(reference: dict[str, dict], path: Path = AUDIT_CSV) -> None:
    if path.exists() and (_read(path)["ref_suspect"].str.strip() != "").any():
        raise FileExistsError(f"{path} already has ref_suspect values; not overwriting the audit")
    rows = [{"item_id": i, "system": it["system"], "subject": it.get("subject", ""),
             "question_hi": question_stem(it["topic_hi"]), "correct_option_hi": it["correct_option_text"],
             "cot_hi": it["cot"], "question_en": "", "correct_option_en": "", "cot_en": "",
             "ref_suspect": "", "note": ""} for i, it in sorted(reference.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


def load_suspect(path: Path = AUDIT_CSV) -> set[str] | None:
    """Suspect item ids, or None if the audit has not been done (no file, or nothing filled in)."""
    if not path.exists():
        return None
    df = _read(path)
    marks = df["ref_suspect"].str.strip().str.lower()
    if (marks == "").all():
        return None
    return set(df.loc[marks.isin(["y", "yes"]), "item_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--export", action="store_true", help="write the audit sheet")
    args = parser.parse_args()
    if args.export:
        export_audit(load_reference())
        print(f"wrote {AUDIT_CSV}\n{__doc__.split('python -m literacy_eval.judge.reference_audit --export')[1]}")
    suspect = load_suspect()
    print("audit pending" if suspect is None else f"{len(suspect)} suspect items: {sorted(suspect)}")


if __name__ == "__main__":
    main()
