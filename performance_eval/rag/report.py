"""RAG results report: scores c0_baseline, rag_rand_k5, rag_k5 for HiMed and runs the
two pre-registered paired McNemar tests (family m = 2, Bonferroni).

CLI: python -m performance_eval.rag.report --raw-dir DIR [--spotcheck SHEET]
     [--types question_types.csv] [--out performance_eval/analysis_outputs/results_himed_rag.md]
"""

import argparse
import csv
import re
from pathlib import Path

from performance_eval import analyze
from performance_eval.analyze import _p, _pct, _signed_pp

CONDITIONS = ("c0_baseline", "rag_rand_k5", "rag_k5")
FAMILY_SIZE = 2
ALPHA = analyze.ALPHA
NON_LETTER_WARN = 0.05
_LETTER_ONLY = re.compile(r"^[A-Ja-j][.)]?$")
DEFAULT_OUT = analyze.DEFAULT_OUT_DIR / "results_himed_rag.md"
FORBIDDEN_OUT = "results_himed.md"
# (new, prior, tested)
COMPARISONS = (("rag_k5", "c0_baseline", True),
               ("rag_k5", "rag_rand_k5", True),
               ("rag_rand_k5", "c0_baseline", False))


def rag_comparisons(scored: dict[str, list[dict]], family_size: int = FAMILY_SIZE) -> list[dict]:
    rows = []
    for new, prior, tested in COMPARISONS:
        c = analyze.compare_conditions(scored[prior], scored[new])
        row = {"new": new, "prior": prior, "tested": tested, "n_paired": c["n_paired"],
               "delta": c["accuracy_delta"], "only_new_correct": c["only_new_correct"],
               "only_prior_correct": c["only_prior_correct"],
               "mcnemar_p": None, "p_adj": None, "significant": None, "harm": None}
        if tested:
            p = c["mcnemar_p"]
            row["mcnemar_p"] = p
            row["p_adj"] = min(1, p * family_size)
            row["significant"] = bool(c["accuracy_delta"] > 0 and row["p_adj"] < ALPHA)
            row["harm"] = bool(c["accuracy_delta"] < 0 and row["p_adj"] < ALPHA)
        rows.append(row)
    return rows


def verdict(comparisons: list[dict]) -> str:
    vs_c0, vs_rand = comparisons[0], comparisons[1]
    if vs_c0["harm"]:
        return "RAG hurts (significant)"
    if vs_c0["significant"] and vs_rand["significant"]:
        return "relevance effect"
    if vs_c0["significant"]:
        return "extra-context effect"
    return "no effect"


def non_letter_rate(answers: list[str]) -> float:
    """Share of raw answers whose Response part (HiMed Thinking stripped) is not a bare
    option letter such as "D", "d." or "B)". Descriptive; the scorer is not changed."""
    from literacy_eval.analysis.clean import extract_answer

    if not answers:
        return 0.0
    bad = sum(not _LETTER_ONLY.match(extract_answer(a or "")[0].strip()) for a in answers)
    return bad / len(answers)


def _per_type_lines(scored: dict[str, list[dict]], types: dict[str, str]) -> list[str]:
    by = {c: {r["item_id"]: bool(r["is_correct"]) for r in scored[c]} for c in CONDITIONS}
    groups: dict[str, list[str]] = {}
    for i in by["c0_baseline"]:
        groups.setdefault(types.get(i, "untagged"), []).append(i)
    lines = ["## Per-type accuracy and flips (exploratory, no p-values)", "",
             "Flips are rag_k5 vs c0_baseline: items newly correct / newly wrong.", "",
             "| tag | n | c0_baseline | rag_rand_k5 | rag_k5 | flips (newly correct / newly wrong) |",
             "|---|---|---|---|---|---|"]
    for tag in sorted(groups, key=lambda t: (t == "untagged", t)):
        ids = groups[tag]
        n = len(ids)
        accs = [_pct(sum(by[c].get(i, False) for i in ids) / n) for c in CONDITIONS]
        up = sum(by["rag_k5"].get(i, False) and not by["c0_baseline"][i] for i in ids)
        down = sum(by["c0_baseline"][i] and not by["rag_k5"].get(i, False) for i in ids)
        lines.append(f"| {tag} | {n} | {' | '.join(accs)} | +{up} / −{down} |")
    return lines + [""]


def non_letter_warning(high: list[str], rates: dict) -> str:
    det = ", ".join(f"{c} {_pct(rates[c])}" for c in high)
    return (f"WARNING: non_letter_rate above {_pct(NON_LETTER_WARN)} for {det}; the first-letter "
            "scorer may misread sentence answers. Report after_marker-mode accuracy and hand-audit those rows.")


def write_report(path, summaries, comparisons, verdict, spotcheck=None, types=None,
                 scored=None, n_options=None, majority=None,
                 non_letter=None, family_size=FAMILY_SIZE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# HiMed-8B RAG results", ""]
    if n_options:
        lines.append(f"Chance baseline: {_pct(1 / n_options)} (1 / {n_options} options, the most "
                     "common option count in the data)."
                     + (f" Majority-letter baseline: always answering {majority[0]} would score "
                        f"{_pct(majority[1])}." if majority else ""))
        lines.append("")
    lines += ["## Accuracy", "",
              "| condition | n | accuracy (95% CI) | none_rate | non_letter_rate |",
              "|---|---|---|---|---|"]
    for c in CONDITIONS:
        s = summaries[c]
        nl = "—" if non_letter is None else _pct(non_letter[c])
        lines.append(f"| {c} | {s['n']} | {_pct(s['accuracy'])} "
                     f"({_pct(s['ci_low'])}–{_pct(s['ci_high'])}) | {_pct(s['none_rate'])} | {nl} |")
    lines += ["", "non_letter_rate = share of responses whose Response part (HiMed Thinking stripped) "
              "is not a bare option letter. Pre-registered: if a RAG condition's non_letter_rate "
              "exceeds 5%, the first-letter scorer may misread sentence answers; also report "
              "after_marker-mode accuracy and hand-audit those rows."]
    high = [c for c in CONDITIONS if non_letter is not None and non_letter[c] > NON_LETTER_WARN]
    if high:
        lines += ["", non_letter_warning(high, non_letter)]
    lines += ["", "## Pre-registered comparisons", "",
              f"Pre-registered family m = {family_size}: paired McNemar tests (rag_k5 vs c0_baseline, "
              f"rag_k5 vs rag_rand_k5). Bonferroni p_adj = min(1, p×{family_size}); significance = "
              f"Δ > 0 and p_adj < {ALPHA}. Harm = Δ < 0 and p_adj < {ALPHA} on the same test; "
              "a significant harm for rag_k5 vs c0_baseline gives the verdict \"RAG hurts (significant)\" "
              "rather than \"no effect\". Δ = new − prior (first-named minus second). "
              "rag_rand_k5 vs c0_baseline is descriptive only (not tested). "
              "Flips = newly correct / newly wrong relative to the prior condition.", "",
              "| comparison (new vs prior) | n_paired | Δ (pp) | flips | McNemar p | p_adj | significant |",
              "|---|---|---|---|---|---|---|"]
    for r in comparisons:
        sig = "—" if r["significant"] is None else ("yes" if r["significant"] else "no")
        lines.append(f"| {r['new']} vs {r['prior']}{'' if r['tested'] else ' (descriptive only)'} "
                     f"| {r['n_paired']} | {_signed_pp(r['delta'])} "
                     f"| +{r['only_new_correct']} / −{r['only_prior_correct']} "
                     f"| {_p(r['mcnemar_p'])} | {_p(r['p_adj'])} | {sig} |")
    lines += ["", f"**Verdict: {verdict}**", ""]
    if spotcheck is not None:
        lines += ["## Retrieval spot-check", "",
                  f"relevant = {spotcheck['relevant']}, partial = {spotcheck['partial']}, "
                  f"none = {spotcheck['none']}. Gate: {'PASS' if spotcheck['passes_gate'] else 'FAIL'}.", ""]
    if types is not None and scored is not None:
        lines += _per_type_lines(scored, types)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_types(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["item_id"]: r["tag"] for r in csv.DictReader(f)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=analyze.DEFAULT_RAW_DIR)
    ap.add_argument("--spotcheck", type=Path)
    ap.add_argument("--types", type=Path)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    if args.out.name == FORBIDDEN_OUT:
        raise SystemExit(f"refusing to write {FORBIDDEN_OUT}; use results_himed_rag.md")

    scored, non_letter = {}, {}
    for c in CONDITIONS:
        raw = args.raw_dir / f"himed_{c}_raw.csv"
        if not raw.is_file():
            raise SystemExit(f"raw CSV not found: {raw}")
        with open(raw, newline="", encoding="utf-8") as f:
            non_letter[c] = non_letter_rate([r["model_answer"] for r in csv.DictReader(f)])
        scored[c] = analyze.score_raw_csv(raw, "first_letter", args.raw_dir / f"himed_{c}_scored.csv")
    analyze.check_item_ids([(c, scored[c]) for c in CONDITIONS])

    summaries = {c: analyze.accuracy_summary(scored[c]) for c in CONDITIONS}
    comps = rag_comparisons(scored)
    spot = None
    if args.spotcheck:
        from performance_eval.rag.spotcheck import summarize
        spot = summarize(args.spotcheck)
    types = _read_types(args.types) if args.types else None
    write_report(args.out, summaries, comps, verdict(comps), spot, types, scored=scored,
                 n_options=analyze.n_options_mode(scored["c0_baseline"]),
                 majority=analyze.majority_baseline(scored["c0_baseline"]),
                 non_letter=non_letter, family_size=FAMILY_SIZE)
    high = [c for c in CONDITIONS if non_letter[c] > NON_LETTER_WARN]
    if high:
        print(non_letter_warning(high, non_letter))
    print(args.out)


if __name__ == "__main__":
    main()
