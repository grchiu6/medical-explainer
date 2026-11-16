"""Accuracy, Wilson CIs and paired McNemar tests across the MCQ condition ladder.

CLI: python -m performance_eval.analyze --model M [--family-size 3] --ladder cond:mode [cond:mode ...]
Conditions are listed in ladder order (kept conditions, then the current
candidate); each is compared to the entry listed immediately before it.
McNemar p-values are Bonferroni-adjusted with a pre-registered family size m
(default 3: the C2, C3 and C4 comparisons per model). Every ladder entry must
cover exactly the same item_ids, with no duplicates.
"""

import argparse
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from literacy_eval.analysis.stats import mcnemar_exact, wilson_ci
from performance_eval.score import score_rows

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = PACKAGE_DIR / "outputs" / "milu_hi_med"
DEFAULT_OUT_DIR = PACKAGE_DIR / "analysis_outputs"
MODES = ("first_letter", "after_marker")
EXPECTED_N = 489  # MILU Hindi Health-and-Medicine test items after the few-shot split
DEFAULT_FAMILY_SIZE = 3
ALPHA = 0.05
_OPTION_LINE_RE = re.compile(r"^[A-J]\. ", re.MULTILINE)


def accuracy_summary(scored_rows: list[dict], mode: str | None = None) -> dict:
    """no_marker_rate (rows whose CoT answer had no explicit marker) is only
    meaningful for after_marker entries; it is None otherwise."""
    n = len(scored_rows)
    n_correct = sum(bool(r["is_correct"]) for r in scored_rows)
    none_rate = (sum(r["parsed_letter"] is None for r in scored_rows) / n) if n else float("nan")
    ci_low, ci_high = wilson_ci(n_correct, n)
    return {
        "n": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else float("nan"),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "none_rate": none_rate,
        "no_marker_rate": (sum(not r["marker_found"] for r in scored_rows) / n)
        if mode == "after_marker" and n else None,
    }


def compare_conditions(prior_scored: list[dict], new_scored: list[dict]) -> dict:
    """Paired comparison on the item_ids present on both sides."""
    prior = {r["item_id"]: bool(r["is_correct"]) for r in prior_scored}
    new = {r["item_id"]: bool(r["is_correct"]) for r in new_scored}
    ids = sorted(set(prior) & set(new))
    a = np.array([prior[i] for i in ids], dtype=bool)
    b = np.array([new[i] for i in ids], dtype=bool)
    if not ids:
        return {"n_paired": 0, "accuracy_delta": float("nan"), "mcnemar_p": float("nan"),
                "only_prior_correct": 0, "only_new_correct": 0}
    m = mcnemar_exact(a, b)
    return {
        "n_paired": len(ids),
        "accuracy_delta": float(b.mean() - a.mean()),
        "mcnemar_p": m["p"],
        "only_prior_correct": m["only_a"],
        "only_new_correct": m["only_b"],
    }


def build_ladder_rows(
    model: str, ladder: list[dict], n_options_mode: int, family_size: int = DEFAULT_FAMILY_SIZE
) -> list[dict]:
    """Flatten one model's ladder; Bonferroni-adjust each McNemar p with the
    pre-registered family size (p_adj = min(1, p * m)). Holm would need the
    whole family at once, which a stepwise ladder never has."""
    rows = []
    for entry in ladder:
        s, cmp_ = entry["summary"], entry.get("comparison")
        p = float("nan") if cmp_ is None else cmp_["mcnemar_p"]
        p_adj = float("nan") if math.isnan(p) else min(1.0, p * family_size)
        delta = None if cmp_ is None else cmp_["accuracy_delta"]
        if cmp_ is None:
            kept = None
        else:
            kept = "yes" if (delta is not None and delta > 0 and p_adj < ALPHA) else "no"
        rows.append({
            "model": model,
            "condition": entry["condition"],
            "n": s["n"],
            "accuracy": s["accuracy"],
            "ci_low": s["ci_low"],
            "ci_high": s["ci_high"],
            "none_rate": s["none_rate"],
            "no_marker_rate": s.get("no_marker_rate"),
            "n_options_mode": n_options_mode,
            "accuracy_delta": delta,
            "n_paired": None if cmp_ is None else cmp_["n_paired"],
            "only_prior_correct": None if cmp_ is None else cmp_["only_prior_correct"],
            "only_new_correct": None if cmp_ is None else cmp_["only_new_correct"],
            "mcnemar_p": None if math.isnan(p) else p,
            "mcnemar_p_adj": None if math.isnan(p_adj) else p_adj,
            "kept": kept,
            "family_size": family_size,
        })
    return rows


def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.1f}%"


def _signed_pp(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.1f} pp"


def _flips(r) -> str:
    new, prior = r.get("only_new_correct"), r.get("only_prior_correct")
    return "—" if new is None or prior is None else f"+{new} / −{prior}"


def _p(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def majority_baseline(rows: list[dict]) -> tuple[str, float]:
    """(most common reference letter, its share); ties go to the first letter seen."""
    counts = Counter(r["reference_answer"].strip().upper() for r in rows)
    letter, count = counts.most_common(1)[0]
    return letter, count / len(rows)


def write_results_md(
    summary_rows: list[dict], out_path: Path, majority_baseline: tuple[str, float] | None = None
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    k = summary_rows[0]["n_options_mode"] if summary_rows else None
    m = summary_rows[0]["family_size"] if summary_rows else DEFAULT_FAMILY_SIZE
    chance = _pct(1 / k) if k else "—"
    majority = (f" Majority-letter baseline: always answering {majority_baseline[0]} would score "
                f"{_pct(majority_baseline[1])}." if majority_baseline else "")
    lines = [
        "# MCQ accuracy by condition",
        "",
        f"Chance baseline: {chance} (1 / {k} options, the most common option count in the data)."
        + majority,
        "",
        "Δ vs prior (in percentage points, pp) and McNemar p compare each row to the row above it "
        "(paired on n_paired items; every row covers the same item_ids). "
        "Flips = items newly correct / newly wrong relative to the row above (+N / −M; net = N − M). "
        f"p (Bonferroni, m={m}) = min(1, p × {m}), with m the pre-registered number of comparisons "
        f"per model. kept? = yes when Δ > 0 and the adjusted p < {ALPHA}. "
        "Accuracy CIs are 95% Wilson intervals. none_rate is the share of answers with no "
        "parseable letter (counted as incorrect). no_marker_rate (CoT / after_marker rows only) is "
        "the share with no explicit Answer marker, scored from the last letter instead; a high "
        "value suggests truncated reasoning. — means not applicable (first row has no prior).",
        "",
        f"| model | condition | n | n_paired | accuracy (95% CI) | none_rate | no_marker_rate "
        f"| Δ vs prior (pp) | flips (newly correct / newly wrong) | McNemar p "
        f"| p (Bonferroni, m={m}) | kept? | chance baseline |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary_rows:
        n_paired = "—" if r.get("n_paired") is None else str(r["n_paired"])
        lines.append(
            f"| {r['model']} | {r['condition']} | {r['n']} | {n_paired} "
            f"| {_pct(r['accuracy'])} ({_pct(r['ci_low'])}–{_pct(r['ci_high'])}) "
            f"| {_pct(r['none_rate'])} | {_pct(r.get('no_marker_rate'))} "
            f"| {_signed_pp(r['accuracy_delta'])} "
            f"| {_flips(r)} "
            f"| {_p(r['mcnemar_p'])} | {_p(r['mcnemar_p_adj'])} "
            f"| {r.get('kept') or '—'} "
            f"| {_pct(1 / r['n_options_mode'])} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def n_options_mode(rows: list[dict]) -> int:
    counts = Counter(len(_OPTION_LINE_RE.findall(r["question_with_options"])) for r in rows)
    return counts.most_common(1)[0][0]


def check_item_ids(entries: list[tuple[str, list[dict]]]) -> None:
    """Exit unless every entry has unique item_ids and all entries share one id set;
    warn (no exit) when n differs from EXPECTED_N."""
    for cond, rows in entries:
        ids = [r["item_id"] for r in rows]
        dupes = sorted(i for i, c in Counter(ids).items() if c > 1)
        if dupes:
            raise SystemExit(f"condition {cond!r} has {len(dupes)} duplicate item_id(s) "
                             f"({len(ids)} rows, {len(set(ids))} unique), e.g. {dupes[:5]}")
    first_cond, first_rows = entries[0]
    first_ids = {r["item_id"] for r in first_rows}
    for cond, rows in entries[1:]:
        ids = {r["item_id"] for r in rows}
        if ids != first_ids:
            raise SystemExit(
                f"item_id sets differ: {first_cond!r} has {len(first_ids)} items, {cond!r} has "
                f"{len(ids)} ({len(first_ids - ids)} only in {first_cond!r}, "
                f"{len(ids - first_ids)} only in {cond!r}); every ladder entry must cover the "
                f"same items (finish or re-run the incomplete condition)")
    if len(first_ids) != EXPECTED_N:
        print(f"WARNING: {len(first_ids)} items per condition, expected {EXPECTED_N} "
              f"(full MILU Hindi medicine test split)", file=sys.stderr)


def score_raw_csv(raw_path: Path, mode: str, scored_path: Path) -> list[dict]:
    with open(raw_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    scored = score_rows(rows, mode)
    out_fields = fieldnames + [c for c in ("parsed_letter", "marker_found", "is_correct") if c not in fieldnames]
    scored_path = Path(scored_path)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scored_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in scored:
            w.writerow({**r,
                        "parsed_letter": r["parsed_letter"] or "",
                        "marker_found": int(r["marker_found"]),
                        "is_correct": int(r["is_correct"])})
    return scored


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ladder", nargs="+", required=True, metavar="COND:MODE",
                    help="condition:mode entries in ladder order; mode is first_letter or after_marker")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--family-size", type=int, default=DEFAULT_FAMILY_SIZE,
                    help="pre-registered Bonferroni family size m per model (default 3: C2, C3, C4)")
    args = ap.parse_args(argv)
    if args.family_size < 1:
        ap.error("--family-size must be >= 1")

    entries = []
    for spec in args.ladder:
        cond, sep, mode = spec.rpartition(":")
        if not sep or not cond or mode not in MODES:
            ap.error(f"bad ladder entry {spec!r}: expected COND:MODE with MODE in {MODES}")
        raw_path = args.raw_dir / f"{args.model}_{cond}_raw.csv"
        if not raw_path.is_file():
            ap.error(f"raw CSV not found: {raw_path}")
        entries.append((cond, mode, raw_path))

    all_scored = [
        (cond, mode, score_raw_csv(raw_path, mode, args.raw_dir / f"{args.model}_{cond}_scored.csv"))
        for cond, mode, raw_path in entries
    ]
    check_item_ids([(cond, scored) for cond, _, scored in all_scored])

    ladder, prev_rows = [], None
    for cond, mode, scored in all_scored:
        ladder.append({
            "condition": cond,
            "summary": accuracy_summary(scored, mode),
            "comparison": None if prev_rows is None else compare_conditions(prev_rows, scored),
        })
        prev_rows = scored

    first_rows = all_scored[0][2]
    rows = build_ladder_rows(args.model, ladder, n_options_mode(first_rows), args.family_size)
    out_path = args.out_dir / f"results_{args.model}.md"
    write_results_md(rows, out_path, majority_baseline(first_rows))
    print(out_path)


if __name__ == "__main__":
    main()
