"""Accuracy, Wilson CIs and paired McNemar tests across the MCQ condition ladder.

CLI: python -m performance_eval.analyze --model M --ladder cond:mode [cond:mode ...]
Conditions are listed in ladder order (kept conditions, then the current
candidate); each is compared to the entry listed immediately before it.
"""

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from literacy_eval.analysis.stats import holm, mcnemar_exact, wilson_ci
from performance_eval.score import score_rows

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = PACKAGE_DIR / "outputs" / "milu_hi_med"
DEFAULT_OUT_DIR = PACKAGE_DIR / "analysis_outputs"
MODES = ("first_letter", "after_marker")
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


def build_ladder_rows(model: str, ladder: list[dict], n_options_mode: int) -> list[dict]:
    """Flatten one model's ladder and Holm-correct its McNemar p-values together."""
    raw_p = []
    for entry in ladder:
        cmp_ = entry.get("comparison")
        raw_p.append(float("nan") if cmp_ is None else cmp_["mcnemar_p"])
    corrected = holm(raw_p)
    rows = []
    for entry, p, p_holm in zip(ladder, raw_p, corrected):
        s, cmp_ = entry["summary"], entry.get("comparison")
        rows.append({
            "model": model,
            "condition": entry["condition"],
            "n": s["n"],
            "accuracy": s["accuracy"],
            "ci_low": s["ci_low"],
            "ci_high": s["ci_high"],
            "none_rate": s["none_rate"],
            "n_options_mode": n_options_mode,
            "accuracy_delta": None if cmp_ is None else cmp_["accuracy_delta"],
            "n_paired": None if cmp_ is None else cmp_["n_paired"],
            "only_prior_correct": None if cmp_ is None else cmp_["only_prior_correct"],
            "only_new_correct": None if cmp_ is None else cmp_["only_new_correct"],
            "mcnemar_p": None if math.isnan(p) else p,
            "mcnemar_p_holm": None if math.isnan(p_holm) else p_holm,
        })
    return rows


def _pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.1f}%"


def _signed_pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.1f}%"


def _flips(r) -> str:
    new, prior = r.get("only_new_correct"), r.get("only_prior_correct")
    return "—" if new is None or prior is None else f"+{new} / −{prior}"


def _p(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def write_results_md(summary_rows: list[dict], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    k = summary_rows[0]["n_options_mode"] if summary_rows else None
    chance = _pct(1 / k) if k else "—"
    lines = [
        "# MCQ accuracy by condition",
        "",
        f"Chance baseline: {chance} (1 / {k} options, the most common option count in the data).",
        "",
        "Δ vs prior and McNemar p compare each row to the row above it (paired, same items). "
        "Flips = items newly correct / newly wrong relative to the row above (+N / −M; net = N − M). "
        "Holm p corrects the McNemar p-values across this model's ladder. "
        "Accuracy CIs are 95% Wilson intervals. none_rate is the share of answers with no "
        "parseable letter (counted as incorrect). — means not applicable (first row has no prior).",
        "",
        "| model | condition | n | accuracy (95% CI) | none_rate | Δ vs prior | flips (newly correct / newly wrong) | McNemar p | McNemar p (Holm) | chance baseline |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['model']} | {r['condition']} | {r['n']} "
            f"| {_pct(r['accuracy'])} ({_pct(r['ci_low'])}–{_pct(r['ci_high'])}) "
            f"| {_pct(r['none_rate'])} | {_signed_pct(r['accuracy_delta'])} "
            f"| {_flips(r)} "
            f"| {_p(r['mcnemar_p'])} | {_p(r['mcnemar_p_holm'])} "
            f"| {_pct(1 / r['n_options_mode'])} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def n_options_mode(rows: list[dict]) -> int:
    counts = Counter(len(_OPTION_LINE_RE.findall(r["question_with_options"])) for r in rows)
    return counts.most_common(1)[0][0]


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
    args = ap.parse_args(argv)

    entries = []
    for spec in args.ladder:
        cond, sep, mode = spec.rpartition(":")
        if not sep or not cond or mode not in MODES:
            ap.error(f"bad ladder entry {spec!r}: expected COND:MODE with MODE in {MODES}")
        raw_path = args.raw_dir / f"{args.model}_{cond}_raw.csv"
        if not raw_path.is_file():
            ap.error(f"raw CSV not found: {raw_path}")
        entries.append((cond, mode, raw_path))

    ladder, prev_rows, first_rows = [], None, None
    for cond, mode, raw_path in entries:
        scored = score_raw_csv(raw_path, mode, args.raw_dir / f"{args.model}_{cond}_scored.csv")
        first_rows = scored if first_rows is None else first_rows
        ladder.append({
            "condition": cond,
            "summary": accuracy_summary(scored, mode),
            "comparison": None if prev_rows is None else compare_conditions(prev_rows, scored),
        })
        prev_rows = scored

    rows = build_ladder_rows(args.model, ladder, n_options_mode(first_rows))
    out_path = args.out_dir / f"results_{args.model}.md"
    write_results_md(rows, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
