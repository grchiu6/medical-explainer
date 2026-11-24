"""Phase A flip table: items HiMed gets right in only one of Hindi / English (plan A6).

  python -m performance_eval.flips export \\
      --hi performance_eval/outputs/milu_hi_med/himed_c0_baseline_scored.csv \\
      --en performance_eval/outputs/milu_en_med/himed_c0_en_google_scored.csv \\
      --alt-en performance_eval/outputs/milu_en_med/himed_c0_en_gemini_scored.csv \\
      --exclude-file performance_eval/data/excluded_items.csv \\
      --exclude-file performance_eval/data/bad_translation_items.csv \\
      --out performance_eval/analysis_outputs/phase_a/flips_himed_google.csv
  # fill `category` (one of CATEGORIES) and `note` per row, then:
  python -m performance_eval.flips summarize \\
      --labeled performance_eval/analysis_outputs/phase_a/flips_himed_google.csv \\
      --out performance_eval/analysis_outputs/phase_a/flips_summary.md
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from performance_eval.exclusions import load_exclusions

# Plan A6 codebook; one primary category per item.
CATEGORIES = ("terminology", "question_form", "traditional_medicine", "translation_artifact",
              "item_noise", "format", "other")
CATEGORY_DEFINITIONS = {
    "terminology": "Hindi medical term rare/Sanskritized/ambiguous; English uses a common term",
    "question_form": "negation, 'which is NOT', multi-statement, assertion–reason",
    "traditional_medicine": "Ayurveda/Yoga/Unani/Indian-specific",
    "translation_artifact": "translation changed meaning, dropped or added a cue",
    "item_noise": "gold looks wrong or item ambiguous in both languages",
    "format": "no parseable answer in one language",
    "other": "none of the above",
}
DIRECTIONS = ("en_only_correct", "hi_only_correct")
COLUMNS = ["item_id", "direction", "gold", "hi_pred", "en_pred", "same_flip_alt",
           "hindi_item", "english_item", "category", "note"]


def _correct(v) -> bool:
    return v in (True, 1, "1", "True", "true")


def flip_rows(hi_scored: list[dict], en_scored: list[dict], alt_en_scored: list[dict] | None = None) -> list[dict]:
    """One row per item whose correctness differs between the Hindi and English runs, in Hindi order.

    same_flip_alt: whether the alternative English run has the same correctness as en_scored on that
    item (i.e. the flip reproduces under the other translation); "" when no alternative is given."""
    en = {r["item_id"]: r for r in en_scored}
    alt = {r["item_id"]: r for r in alt_en_scored} if alt_en_scored is not None else None
    hi_ids = [r["item_id"] for r in hi_scored]
    for name, other in (("en", en), ("alt_en", alt)):
        if other is not None and set(other) != set(hi_ids):
            raise ValueError(f"{name} covers different items than hi ({len(set(hi_ids) ^ set(other))} differ); "
                             "apply the same exclusions to every run")
    out = []
    for h in hi_scored:
        e = en[h["item_id"]]
        hc, ec = _correct(h["is_correct"]), _correct(e["is_correct"])
        if hc == ec:
            continue
        out.append({
            "item_id": h["item_id"],
            "direction": "en_only_correct" if ec else "hi_only_correct",
            "gold": h["reference_answer"],
            "hi_pred": h.get("parsed_letter") or "",
            "en_pred": e.get("parsed_letter") or "",
            "same_flip_alt": "" if alt is None else _correct(alt[h["item_id"]]["is_correct"]) == ec,
            "hindi_item": h["question_with_options"],
            "english_item": e["question_with_options"],
            "category": "",
            "note": "",
        })
    return out


def summarize(rows: list[dict]) -> str:
    """Direction x category counts. Refuses rows with an empty or unknown category."""
    bad = [r["item_id"] for r in rows if r.get("category", "") not in CATEGORIES]
    if bad:
        raise ValueError(f"{len(bad)} row(s) have an empty or unknown category: {', '.join(bad)}; "
                         f"allowed: {', '.join(CATEGORIES)}")
    counts = Counter((r["category"], r["direction"]) for r in rows)
    n_dir = Counter(r["direction"] for r in rows)
    lines = ["Labels: one LLM rater (Claude), no human check.", "",
             "# HiMed Hindi/English flip table: summary", "",
             f"Items whose correctness differs between `himed c0_baseline` (Hindi) and `himed c0_en_google` "
             f"(English): {len(rows)} ({n_dir['en_only_correct']} English-only correct, "
             f"{n_dir['hi_only_correct']} Hindi-only correct).", "",
             "| category | en_only_correct | hi_only_correct | total |", "|---|---|---|---|"]
    for c in CATEGORIES:
        a, b = counts[(c, "en_only_correct")], counts[(c, "hi_only_correct")]
        lines.append(f"| {c} | {a} | {b} | {a + b} |")
    lines.append(f"| total | {n_dir['en_only_correct']} | {n_dir['hi_only_correct']} | {len(rows)} |")
    alt = [r for r in rows if str(r.get("same_flip_alt", "")) in ("True", "False")]
    if alt:
        same = Counter((r["direction"], str(r["same_flip_alt"]) == "True") for r in alt)
        lines += ["", "## Same flip under the gemini translation", "",
                  f"{sum(str(r['same_flip_alt']) == 'True' for r in alt)} of {len(alt)} flipped items flip "
                  "the same way when the English item comes from the gemini translation "
                  f"(English-only correct: {same[('en_only_correct', True)]} of {n_dir['en_only_correct']}; "
                  f"Hindi-only correct: {same[('hi_only_correct', True)]} of {n_dir['hi_only_correct']})."]
    lines += ["", "## Codebook", ""]
    lines += [f"- `{c}`: {CATEGORY_DEFINITIONS[c]}" for c in CATEGORIES]
    return "\n".join(lines) + "\n"


def _read(path: Path, excluded: set[str]) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["item_id"] not in excluded]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--hi", type=Path, required=True, help="Hindi scored CSV")
    ex.add_argument("--en", type=Path, required=True, help="English scored CSV")
    ex.add_argument("--alt-en", type=Path, default=None, help="English scored CSV from the other translation")
    ex.add_argument("--exclude-file", action="append", type=Path, default=[])
    ex.add_argument("--out", type=Path, required=True)
    sm = sub.add_parser("summarize")
    sm.add_argument("--labeled", type=Path, required=True)
    sm.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "export":
        excluded: set[str] = set()
        for f in args.exclude_file:
            excluded |= set(load_exclusions(f))
        rows = flip_rows(_read(args.hi, excluded), _read(args.en, excluded),
                         _read(args.alt_en, excluded) if args.alt_en else None)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"{args.out}: {len(rows)} rows")
    else:
        with open(args.labeled, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(summarize(rows), encoding="utf-8")
        print(args.out)


if __name__ == "__main__":
    sys.exit(main())
