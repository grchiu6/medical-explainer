"""Step-1 diagnostic report: is accuracy limited by the open models, by Hindi, or by the questions?

Grid: {himed, nanda, gemini_flash} x {hi, en}, all paired on the same item_ids after exclusions.
Five pre-registered comparisons (plan D7), exact McNemar, Bonferroni m = 5; verdict rule fixed in
advance (plan D8).

CLI:
  python -m performance_eval.diagnostic \\
      --cell himed:hi:performance_eval/outputs/milu_hi_med/himed_c0_baseline_raw.csv ... \\
      --exclude-file performance_eval/data/excluded_items.csv \\
      --exclude-file performance_eval/data/bad_translation_items.csv \\
      --out performance_eval/analysis_outputs/step1_diagnostic.md
A cell is MODEL:LANG:RAW_CSV; LANG is "hi" or "en" (use --en-tag to label which translation).
"""

import argparse
import json
import sys
from pathlib import Path

from performance_eval.analyze import accuracy_summary, compare_conditions, score_raw_csv
from performance_eval.exclusions import load_exclusions

DEFAULT_FAMILY_SIZE = 5
ALPHA = 0.05
OPEN_MODELS = ("himed", "nanda")
GEMINI = "gemini_flash"
# (base, target): delta = accuracy(target) - accuracy(base). Order = plan D7.
COMPARISONS = [
    ((GEMINI, "hi"), ("himed", "hi")),
    ((GEMINI, "hi"), ("nanda", "hi")),
    (("himed", "en"), ("himed", "hi")),
    (("nanda", "en"), ("nanda", "hi")),
    ((GEMINI, "en"), (GEMINI, "hi")),
]
CEILING_GAP = 0.10  # plan D8: gemini/hi >= 10 pp above the better open model
LANGUAGE_GAP = 0.05  # plan D8: en >= 5 pp above hi for both open models


def check_same_items(cells: dict[tuple[str, str], list[dict]]) -> None:
    keys = list(cells)
    first = {r["item_id"] for r in cells[keys[0]]}
    for k in keys[1:]:
        ids = {r["item_id"] for r in cells[k]}
        if ids != first:
            raise SystemExit(
                f"item sets differ: {'/'.join(keys[0])} has {len(first)} items, {'/'.join(k)} has {len(ids)} "
                f"({len(first - ids)} only in the first, {len(ids - first)} only in {'/'.join(k)}); "
                "every cell must cover the same items after exclusions")


def _significant(comp: dict, family_size: int = DEFAULT_FAMILY_SIZE) -> bool:
    return comp["accuracy_delta"] > 0 and min(1.0, comp["mcnemar_p"] * family_size) < ALPHA


def run_comparisons(cells: dict[tuple[str, str], list[dict]], family_size: int = DEFAULT_FAMILY_SIZE) -> list[dict]:
    """Each D7 comparison, reported as 'target vs base' with delta = target - base."""
    out = []
    for target, base in COMPARISONS:
        label = f"{'/'.join(target)} vs {'/'.join(base)}"
        if target not in cells or base not in cells:
            out.append({"label": label, "target": target, "base": base, "skipped": True})
            continue
        c = compare_conditions(cells[base], cells[target])
        out.append({"label": label, "target": target, "base": base, "skipped": False, **c,
                    "p_adj": min(1.0, c["mcnemar_p"] * family_size),
                    "significant": _significant(c, family_size)})
    return out


def _acc(rows: list[dict]) -> float:
    return sum(bool(r["is_correct"]) for r in rows) / len(rows)


def decide(cells: dict[tuple[str, str], list[dict]], comps: list[dict]) -> str:
    """Plan D8, applied mechanically."""
    lines = []
    skipped = [c["label"] for c in comps if c["skipped"]]
    if skipped:
        lines.append(f"Verdict incomplete: missing cells for {', '.join(skipped)}.")
    c1, c2, c3, c4, _ = comps
    found = False
    if not (c1["skipped"] or c2["skipped"]):
        gap = _acc(cells[(GEMINI, "hi")]) - max(_acc(cells[(m, "hi")]) for m in OPEN_MODELS)
        if c1["significant"] and c2["significant"] and gap >= CEILING_GAP:
            lines.append(f"The open Indic models are the bottleneck: Gemini is {gap * 100:.1f} pp above "
                         "the better open model in Hindi (comparisons 1 and 2 significant). "
                         "Next: fine-tuning (e.g. translated MedMCQA) or a stronger base model.")
            found = True
    if not (c3["skipped"] or c4["skipped"]):
        if all(c["significant"] and c["accuracy_delta"] >= LANGUAGE_GAP for c in (c3, c4)):
            lines.append("Hindi is the bottleneck: both open models score >= 5 pp higher on the English "
                         "translations (comparisons 3 and 4 significant). "
                         "Next: translate-then-answer pipeline, more Hindi medical training data.")
            found = True
    complete = not any(c["skipped"] for c in (c1, c2, c3, c4))
    if not found and complete:
        lines.append("Neither the models nor the language explain the ceiling: it is in the questions. "
                     "Next: audit the items all three models get wrong.")
    return " ".join(lines)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_report(cells, comps, verdict, n_info: list[str], cost_lines: list[str], family_size: int) -> str:
    lines = ["# Step 1 diagnostic", "", *n_info, "", "## Accuracy (95% Wilson CI)", "",
             "| model | language | n | accuracy | 95% CI | none_rate |", "|---|---|---|---|---|---|"]
    for (model, lang), rows in cells.items():
        s = accuracy_summary(rows)
        lines.append(f"| {model} | {lang} | {s['n']} | {_pct(s['accuracy'])} | "
                     f"{_pct(s['ci_low'])}–{_pct(s['ci_high'])} | {_pct(s['none_rate'])} |")
    lines += ["", f"## Pre-registered comparisons (exact McNemar, Bonferroni m = {family_size}, alpha = {ALPHA})", "",
              "| # | comparison | Δ (pp) | flips (newly correct / newly wrong) | McNemar p | p (Bonferroni) | significant? |",
              "|---|---|---|---|---|---|---|"]
    for i, c in enumerate(comps, 1):
        if c["skipped"]:
            lines.append(f"| {i} | {c['label']} | — | — | — | — | not run |")
            continue
        lines.append(f"| {i} | {c['label']} | {c['accuracy_delta'] * 100:+.1f} | +{c['only_new_correct']} / "
                     f"−{c['only_prior_correct']} | {c['mcnemar_p']:.4g} | {c['p_adj']:.4g} | "
                     f"{'yes' if c['significant'] else 'no'} |")
    lines += ["", "## Verdict (rule fixed before the runs; plan D8)", "", verdict or "—", ""]
    if cost_lines:
        lines += ["## Gemini cost", "", *cost_lines, ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", required=True, metavar="MODEL:LANG:RAW_CSV")
    ap.add_argument("--exclude-file", action="append", type=Path, default=[])
    ap.add_argument("--family-size", type=int, default=DEFAULT_FAMILY_SIZE)
    ap.add_argument("--en-tag", default="", help="label for the translation used, e.g. google or gemini")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    excluded: set[str] = set()
    n_info = []
    for f in args.exclude_file:
        ex = load_exclusions(f)
        excluded |= set(ex)
        n_info.append(f"- excluded by `{f.name}`: {len(ex)} items")
    cells, cost_lines = {}, []
    for spec in args.cell:
        model, lang, raw = spec.split(":", 2)
        raw = Path(raw)
        if not raw.is_file():
            ap.error(f"raw CSV not found: {raw}")
        scored = score_raw_csv(raw, "first_letter", raw.with_name(raw.name.replace("_raw.csv", "_scored.csv")))
        before = len(scored)
        cells[(model, lang)] = [r for r in scored if r["item_id"] not in excluded]
        n_info.append(f"- {model}/{lang}: {before} rows in `{raw.name}`, {len(cells[(model, lang)])} after exclusions")
        usage = raw.with_name(raw.name + ".usage.json")
        if usage.is_file():
            u = json.loads(usage.read_text(encoding="utf-8"))
            cost_lines.append(f"- {model}/{lang}: {u['calls']} calls, ${u['cost_usd']:.4f} "
                              f"({u['thoughts_tokens']} thinking tokens, {u['blocked']} blocked)")
    check_same_items(cells)
    if args.en_tag:
        n_info.append(f"- English translation: {args.en_tag}")
    comps = run_comparisons(cells, args.family_size)
    report = build_report(cells, comps, decide(cells, comps), n_info, cost_lines, args.family_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    sys.exit(main())
