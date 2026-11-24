"""Phase A gap-localization report: which part of a Hindi MCQ costs HiMed its Hindi->English gap?

Cells (all paired on the same item_ids after exclusions):
  hi        c0_baseline            Hindi stem, Hindi options
  en        c0_en_T                English stem, English options
  q_hi_o_en c0_mix_qhi_oen_T       Hindi stem, English options
  q_en_o_hi c0_mix_qen_ohi_T       English stem, Hindi options
  cot_hi    c2_cot                 Hindi item, Hindi chain-of-thought
  cot_en    c2_cot_en              Hindi item, English chain-of-thought

Three pre-registered comparisons (plan A4), exact McNemar, two-sided, Bonferroni m = 3; the decision
rule (plan A5) is applied mechanically.

CLI:
  python -m performance_eval.gap_report \\
      --cell hi:first_letter:performance_eval/outputs/milu_hi_med/himed_c0_baseline_raw.csv \\
      --cell en:first_letter:performance_eval/outputs/milu_en_med/himed_c0_en_google_raw.csv \\
      --cell q_hi_o_en:first_letter:.../himed_c0_mix_qhi_oen_google_raw.csv \\
      --cell q_en_o_hi:first_letter:.../himed_c0_mix_qen_ohi_google_raw.csv \\
      --cell cot_hi:after_marker:performance_eval/outputs/milu_hi_med/himed_c2_cot_raw.csv \\
      --cell cot_en:after_marker:performance_eval/outputs/milu_hi_med/himed_c2_cot_en_raw.csv \\
      --exclude-file performance_eval/data/excluded_items.csv \\
      --exclude-file performance_eval/data/bad_translation_items.csv \\
      --en-tag google --out performance_eval/analysis_outputs/phase_a/gap_report_google.md
A cell is NAME:MODE:RAW_CSV. A missing cell makes its comparison "not run".
"""

import argparse
import csv
import re
import sys
from pathlib import Path

from performance_eval.analyze import MODES, accuracy_summary, compare_conditions, score_raw_csv
from performance_eval.exclusions import load_exclusions

FAMILY_SIZE = 3
ALPHA = 0.05
MIN_GAP = 0.05  # plan A5: below this there is no gap to localize
MIN_RECOVERED = 0.50  # plan A5: a part must recover >= half the gap
MIN_COMPLIANCE = 0.80  # plan A5: share of c2_cot_en responses with mostly-English reasoning
MOSTLY_LATIN = 0.50

CELL_NAMES = ("hi", "en", "q_hi_o_en", "q_en_o_hi", "cot_hi", "cot_en")
CELL_LABELS = {"hi": "c0_baseline (hi stem, hi options)", "en": "c0_en (en stem, en options)",
               "q_hi_o_en": "c0_mix_qhi_oen (hi stem, en options)",
               "q_en_o_hi": "c0_mix_qen_ohi (en stem, hi options)",
               "cot_hi": "c2_cot (hi item, Hindi CoT)", "cot_en": "c2_cot_en (hi item, English CoT)"}
# (target, control); order = plan A4.
COMPARISONS = [("q_hi_o_en", "hi"), ("q_en_o_hi", "hi"), ("cot_en", "cot_hi")]
# What each comparison localizes and the Phase B direction it points to (plan A5).
_PARTS = {
    "q_hi_o_en": ("Hindi option terms",
                  "bilingual term-level data (Hindi term ↔ English term) in SFT"),
    "q_en_o_hi": ("Hindi question stem", "parallel bilingual SFT on full questions"),
    "cot_en": ("reasoning language", "SFT on Hindi question → English reasoning → answer"),
}
_NONE_MESSAGE = ("No single part explains it; only the fully English item recovers it → Phase B: "
                 "cross-lingual self-distillation (English-item teacher → Hindi-item student).")
_NONCOMPLIANT = "English-reasoning instruction not followed; comparison 3 inconclusive."

_LATIN_RE = re.compile(r"[A-Za-z]")
_DEVANAGARI_LETTER_RE = re.compile(r"[ऀ-ॿ]")
_ANSWER_RE = re.compile(r"Answer\s*:")
_THINKING_HEADER_RE = re.compile(r"^\s*##\s*Thinking\s*", re.IGNORECASE)


def reasoning_latin_share(response: str) -> float | None:
    """Latin / (Latin + Devanagari) letters in the reasoning part of a CoT response.

    Reasoning = text before HiMed's '## Response' header if present, else before the last 'Answer:'
    (else the whole text); a leading '## Thinking' header is dropped. Devanagari counts only
    alphabetic code points (vowel signs and virama are combining marks, not letters).
    None when the reasoning part has no letters."""
    text = response or ""
    if "## Response" in text:
        text = text.split("## Response", 1)[0]
    else:
        matches = list(_ANSWER_RE.finditer(text))
        if matches:
            text = text[: matches[-1].start()]
    text = _THINKING_HEADER_RE.sub("", text)
    latin = len(_LATIN_RE.findall(text))
    deva = sum(c.isalpha() for c in _DEVANAGARI_LETTER_RE.findall(text))
    if latin + deva == 0:
        return None
    return latin / (latin + deva)


def compliance_rate(responses: list[str]) -> float | None:
    """Share of responses whose reasoning is mostly (>= 50%) Latin; no-letter responses count as not."""
    if not responses:
        return None
    shares = [reasoning_latin_share(r) for r in responses]
    return sum(s is not None and s >= MOSTLY_LATIN for s in shares) / len(shares)


def _acc(rows: list[dict]) -> float:
    return sum(bool(r["is_correct"]) for r in rows) / len(rows)


def gap(cells: dict[str, list[dict]]) -> float | None:
    """G = acc(en) - acc(hi); None if either cell is missing or empty."""
    if not cells.get("hi") or not cells.get("en"):
        return None
    return _acc(cells["en"]) - _acc(cells["hi"])


def run_comparisons(cells: dict[str, list[dict]], family_size: int = FAMILY_SIZE) -> list[dict]:
    """Each A4 comparison as 'target vs control' with delta = target - control and recovered = delta / G."""
    g = gap(cells)
    out = []
    for target, base in COMPARISONS:
        label = f"{target} vs {base}"
        if target not in cells or base not in cells:
            out.append({"label": label, "target": target, "base": base, "skipped": True})
            continue
        c = compare_conditions(cells[base], cells[target])
        p_adj = min(1.0, c["mcnemar_p"] * family_size)
        out.append({"label": label, "target": target, "base": base, "skipped": False, **c,
                    "p_adj": p_adj, "significant": p_adj < ALPHA,
                    "recovered": (c["accuracy_delta"] / g) if g else None})
    return out


def _fires(c: dict) -> bool:
    return (not c["skipped"] and c["significant"] and c["accuracy_delta"] > 0
            and c["recovered"] is not None and c["recovered"] >= MIN_RECOVERED)


def localize(cells: dict[str, list[dict]], comps: list[dict], compliance: float | None) -> str:
    """Plan A5, applied mechanically."""
    g = gap(cells)
    if g is None:
        return "Verdict incomplete: the hi and en cells are both needed to measure the gap; no rule applied."
    if g < MIN_GAP:
        return f"G = {g * 100:.1f} pp < {MIN_GAP * 100:.0f} pp: no gap to localize; no rule applied."
    lines, fired = [], []
    for i, c in enumerate(comps, 1):
        if c["skipped"]:
            continue
        if c["target"] == "cot_en" and (compliance is None or compliance < MIN_COMPLIANCE):
            lines.append(_NONCOMPLIANT)
            continue
        if _fires(c):
            part, phase_b = _PARTS[c["target"]]
            fired.append(c)
            lines.append(f"Comparison {i} fires: {part} explains the gap ("
                         f"Δ {c['accuracy_delta'] * 100:+.1f} pp, recovers {c['recovered'] * 100:.0f}% of "
                         f"G = {g * 100:.1f} pp) → Phase B: {phase_b}.")
    skipped = [c["label"] for c in comps if c["skipped"]]
    if not fired:
        if skipped:
            lines.append(f"Verdict incomplete: no part fired among the comparisons run; not run: {', '.join(skipped)}.")
        else:
            lines.append(_NONE_MESSAGE)
    elif skipped:
        lines.append(f"(Not run: {', '.join(skipped)}.)")
    return " ".join(lines)


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _sig_label(c: dict) -> str:
    if not c["significant"]:
        return "no"
    return "yes (better)" if c["accuracy_delta"] > 0 else "yes (worse)"


def build_report(cells, comps, verdict, n_info: list[str], compliance_lines: list[str]) -> str:
    g = gap(cells)
    lines = ["# Phase A: gap localization (HiMed)", "", *n_info, "",
             f"Gap G = acc(en) − acc(hi) = {'—' if g is None else f'{g * 100:+.1f} pp'}", "",
             "## Accuracy (95% Wilson CI)", "",
             "| cell | setup | n | accuracy | 95% CI | none_rate |", "|---|---|---|---|---|---|"]
    for name in CELL_NAMES:
        if name not in cells:
            lines.append(f"| {name} | {CELL_LABELS[name]} | — | — | — | not run |")
            continue
        s = accuracy_summary(cells[name])
        lines.append(f"| {name} | {CELL_LABELS[name]} | {s['n']} | {_pct(s['accuracy'])} | "
                     f"{_pct(s['ci_low'])}–{_pct(s['ci_high'])} | {_pct(s['none_rate'])} |")
    lines += ["", f"## Pre-registered comparisons (exact McNemar, Bonferroni m = {FAMILY_SIZE}, alpha = {ALPHA})", "",
              "| # | comparison | Δ (pp) | flips (newly correct / newly wrong) | McNemar p | p (Bonferroni) | "
              "significant? | recovered share of G |",
              "|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(comps, 1):
        if c["skipped"]:
            lines.append(f"| {i} | {c['label']} | — | — | — | — | not run | — |")
            continue
        lines.append(f"| {i} | {c['label']} | {c['accuracy_delta'] * 100:+.1f} | +{c['only_new_correct']} / "
                     f"−{c['only_prior_correct']} | {c['mcnemar_p']:.4g} | {c['p_adj']:.4g} | {_sig_label(c)} | "
                     f"{_pct(c['recovered'])} |")
    lines += ["", "## English-reasoning compliance (share of responses whose reasoning letters are ≥ 50% Latin)", "",
              *(compliance_lines or ["—"]), "",
              "## Verdict (rule fixed before the runs; plan A5)", "", verdict or "—", "",
              "A conclusion counts only if the google and gemini reports agree (step-1 D5); "
              "comparison 3 uses no translation and is identical in both.", ""]
    return "\n".join(lines)


def _read_raw_answers(raw: Path) -> dict[str, str]:
    with open(raw, newline="", encoding="utf-8") as f:
        return {r["item_id"]: r["model_answer"] for r in csv.DictReader(f)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", required=True, metavar="NAME:MODE:RAW_CSV")
    ap.add_argument("--exclude-file", action="append", type=Path, default=[])
    ap.add_argument("--en-tag", default="", help="translation used for en / mixed cells: google or gemini")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    specs = {}
    for spec in args.cell:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            ap.error(f"--cell must be NAME:MODE:RAW_CSV, got {spec!r}")
        name, mode, raw = parts
        if name not in CELL_NAMES:
            ap.error(f"unknown cell {name!r}; choose from {CELL_NAMES}")
        if mode not in MODES:
            ap.error(f"unknown mode {mode!r}; choose from {MODES}")
        if name in specs:
            ap.error(f"cell {name!r} given twice")
        raw = Path(raw)
        if not raw.is_file():
            ap.error(f"raw CSV not found: {raw}")
        specs[name] = (mode, raw)

    excluded: set[str] = set()
    n_info = []
    for f in args.exclude_file:
        ex = load_exclusions(f)
        excluded |= set(ex)
        n_info.append(f"- excluded by `{f.name}`: {len(ex)} items")
    cells, answers = {}, {}
    for name in CELL_NAMES:
        if name not in specs:
            continue
        mode, raw = specs[name]
        scored = score_raw_csv(raw, mode, raw.with_name(raw.name.replace("_raw.csv", "_scored.csv")))
        cells[name] = [r for r in scored if r["item_id"] not in excluded]
        answers[name] = {i: a for i, a in _read_raw_answers(raw).items() if i not in excluded}
        n_info.append(f"- {name} ({mode}): {len(scored)} rows in `{raw.name}`, {len(cells[name])} after exclusions")
    names = list(cells)
    first = {r["item_id"] for r in cells[names[0]]}
    for k in names[1:]:
        ids = {r["item_id"] for r in cells[k]}
        if ids != first:
            raise SystemExit(f"item sets differ: {names[0]} has {len(first)} items, {k} has {len(ids)} "
                             f"({len(first - ids)} only in {names[0]}, {len(ids - first)} only in {k}); "
                             "every cell must cover the same items after exclusions")
    if args.en_tag:
        n_info.append(f"- English translation: {args.en_tag}")

    compliance = compliance_rate(list(answers["cot_en"].values())) if "cot_en" in answers else None
    compliance_lines = []
    for name, label in (("cot_en", "c2_cot_en"), ("cot_hi", "c2_cot")):
        if name in answers:
            compliance_lines.append(f"- {label}: {_pct(compliance_rate(list(answers[name].values())))} "
                                    "responses with mostly-English reasoning"
                                    + (f" (threshold {MIN_COMPLIANCE * 100:.0f}%)" if name == "cot_en" else
                                       " (contrast)"))
    comps = run_comparisons(cells)
    report = build_report(cells, comps, localize(cells, comps, compliance), n_info, compliance_lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    sys.exit(main())
