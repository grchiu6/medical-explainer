"""Step 7 of the HiMed-Trad evaluation: results tables and statistics.

Reads whatever analysis outputs exist (features.csv is required; judge CSVs
are optional, so this can run before the judge has finished) and writes:

  tables/usability.csv        per model x level: failure types, usable %
  tables/usability_tests.csv  per model: Cochran's Q across levels, McNemar low vs high
  tables/adaptation.csv       per model x feature: medians, Page's L, Wilcoxon, effect size
  tables/compliance.csv       per model x level: does the answer follow its level's instructions?
  tables/consistency.csv      per model x pair: SAME/COMPATIBLE/CONTRADICTS shares
  tables/accuracy.csv         per model x level: option accuracy (all items / clean items)
  tables/facts.csv            per model x level: fact recall, contradiction rate
  tables/harm_flags.csv       every possibly_harmful row, for manual review
  results.md                  all of the above in one readable file

"Usable" means usable_rules AND (if judged) coherent. Adaptation uses only
items usable at all three levels, so each comparison is within-item.

    python -m literacy_eval.analysis.aggregate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from literacy_eval.analysis import stats
from literacy_eval.analysis.clean import ANALYSIS_DIR, LEVELS
from literacy_eval.analysis.features import FEATURES_CSV
from literacy_eval.analysis.judge import (
    REFERENCE_JSON, TASK_FILES, load_judged, load_reference,
)

TABLES_DIR = ANALYSIS_DIR / "tables"
RESULTS_MD = ANALYSIS_DIR / "results.md"
ADAPTATION_FEATURES = ["asl", "short_sentence_share", "term_rate", "rare_rate", "awl",
                       "psw_rate", "juk_rate", "latin_rate", "rh1", "rh2",
                       "gloss_rate", "copy_rate", "n_words"]
# Expected change from low to high: +1 = more complex at high, -1 = less.
# 0 = no ordered hypothesis (reported, but left out of the Holm family):
# gloss_rate should peak at mid, copy_rate is a confound check, and n_words
# is capped by each model's max_new_tokens.
EXPECTED_DIRECTION = {"short_sentence_share": -1, "gloss_rate": 0, "copy_rate": 0, "n_words": 0}
_DIRECTION_LABEL = {1: "up", -1: "down", 0: "none"}
FAILURE_TYPES = ["refusal", "non_answer", "echo", "degenerate", "non_hindi", "too_short"]


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower() == "true"


def load_inputs(features_csv: Path = FEATURES_CSV) -> dict[str, pd.DataFrame]:
    """features.csv joined with judge results where they exist."""
    data = {"features": pd.read_csv(features_csv)}
    data["features"]["usable_rules"] = _truthy(data["features"]["usable_rules"])
    for task, path in TASK_FILES.items():
        if path.exists():
            judged = load_judged(path)
            data[task] = judged[judged["judge_error"] == ""].copy()
    feats = data["features"]
    feats["job_id"] = feats["model"] + "|" + feats["level"] + "|" + feats["item_id"]
    if "response" in data:
        resp = data["response"][["job_id", "coherent", "mapped_option", "harm", "answer_en", "harm_reason"]]
        feats = feats.merge(resp, on="job_id", how="left")
        # Per row: a row the judge hasn't seen yet keeps its rule-based verdict.
        feats["judged"] = feats["coherent"].notna()
        feats["usable"] = feats["usable_rules"] & (~feats["judged"] | _truthy(feats["coherent"].fillna("")))
    else:
        feats["judged"] = False
        feats["usable"] = feats["usable_rules"]
    data["features"] = feats
    return data


def usability_table(feats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, level), g in feats.groupby(["model", "level"], sort=False):
        n = len(g)
        row = {"model": model, "level": level, "n": n}
        for ft in FAILURE_TYPES:
            row[ft] = int((g["failure_type"] == ft).sum())
        row["incoherent_judge"] = int((g["usable_rules"] & ~g["usable"]).sum())
        # Informational: mostly reflects each model's max_new_tokens, not behaviour.
        row["truncated"] = int(_truthy(g["truncated"]).sum()) if "truncated" in g else 0
        row["usable"] = int(g["usable"].sum())
        lo, hi = stats.wilson_ci(row["usable"], n)
        row["usable_pct"] = round(100 * row["usable"] / n, 1)
        row["usable_ci"] = f"{100 * lo:.0f}-{100 * hi:.0f}"
        rule_ok = g[g["usable_rules"]]
        share = rule_ok["judged"].mean() if len(rule_ok) else 1.0
        row["gate"] = ("rules+judge" if share == 1 else "rules only (upper bound)" if share == 0
                       else f"rules+judge ({share:.0%} judged)")
        rows.append(row)
    return pd.DataFrame(rows)


def _wide(feats: pd.DataFrame, model: str, column: str) -> pd.DataFrame:
    g = feats[feats["model"] == model]
    return g.pivot(index="item_id", columns="level", values=column).reindex(columns=LEVELS)


def usability_tests(feats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in feats["model"].unique():
        for outcome in ("usable", "refusal"):
            w = _wide(feats.assign(refusal=feats["failure_type"] == "refusal"), model, outcome)
            w = w.dropna().astype(int)
            q = stats.cochran_q(w.values)
            mc = stats.mcnemar_exact(w["low"].values, w["high"].values)
            rows.append({"model": model, "outcome": outcome, "n_items": q["n"],
                         "low_pct": round(100 * w["low"].mean(), 1),
                         "mid_pct": round(100 * w["mid"].mean(), 1),
                         "high_pct": round(100 * w["high"].mean(), 1),
                         "cochran_Q": round(q["Q"], 2), "cochran_p": q["p"],
                         "mcnemar_low_only": mc["only_a"], "mcnemar_high_only": mc["only_b"],
                         "mcnemar_p": mc["p"]})
    return pd.DataFrame(rows)


def adaptation_table(feats: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        usable = _wide(feats, model, "usable").fillna(False).astype(bool)
        keep = usable.index[usable.all(axis=1)]
        model_rows = []
        for feat in ADAPTATION_FEATURES:
            direction = EXPECTED_DIRECTION.get(feat, 1)
            w = _wide(feats, model, feat).loc[keep].dropna()
            sign = direction or 1            # "none" features are tested as increasing
            pt = (stats.page_trend(w.values, increasing=sign > 0) if len(w)
                  else {"n": 0, "L": np.nan, "p": np.nan})
            # Flip signs so rank_biserial > 0 and share_expected_direction count
            # movement in the EXPECTED direction for both "up" and "down" features.
            wx = stats.wilcoxon_paired(sign * w["low"].values, sign * w["high"].values)
            steps = sign * w[["mid", "high"]].values - sign * w[["low", "mid"]].values
            monotonic = float(np.mean((steps > 0).all(axis=1))) if len(w) and direction else np.nan
            model_rows.append({
                "model": model, "feature": feat, "expected": _DIRECTION_LABEL[direction],
                "n_items": len(w),
                "median_low": w["low"].median(), "median_mid": w["mid"].median(),
                "median_high": w["high"].median(),
                "page_p": pt["p"], "wilcoxon_p": wx["p"],
                "rank_biserial": wx["rank_biserial"],
                "share_expected_direction": wx["share_low_lt_high"],
                "share_monotonic": monotonic,
            })
        # Holm within each model's family of features that have an ordered hypothesis
        family = [r for r in model_rows if r["expected"] != "none"]
        for r, adj in zip(family, stats.holm([r["page_p"] for r in family])):
            r["page_p_holm"] = adj
        rows.extend(model_rows)
    return pd.DataFrame(rows).round(4)


def compliance_table(feats: pd.DataFrame) -> pd.DataFrame:
    """Per model x level: checks tied to single lines of the level prompts.

    Low says "no medical or English words" and "5-10 word sentences"; mid
    says "give the simple meaning of medical terms"; high asks for technical
    terminology. Computed on usable answers only. Means, not medians: these
    features are 0 for most answers, so medians are 0 almost everywhere.
    """
    rows = []
    for (model, level), g in feats[feats["usable"]].groupby(["model", "level"], sort=False):
        rows.append({
            "model": model, "level": level, "n_usable": len(g),
            "no_latin_pct": round(100 * (g["latin_rate"] == 0).mean(), 1),
            "mean_short_sentence_share": round(g["short_sentence_share"].mean(), 3),
            "mostly_short_pct": round(100 * (g["short_sentence_share"] >= 0.5).mean(), 1),
            "mean_term_rate": round(g["term_rate"].mean(), 4),
            "uses_terms_pct": round(100 * (g["term_rate"] > 0).mean(), 1),
            "mean_gloss_rate": round(g["gloss_rate"].mean(), 2),
            "has_gloss_pct": round(100 * (g["gloss_rate"] > 0).mean(), 1),
        })
    return pd.DataFrame(rows)


def consistency_table(cons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, pair), g in cons.groupby(["model", "pair"]):
        verdicts = g["verdict"]
        preserved = verdicts.isin(["SAME", "COMPATIBLE"]).astype(float).values
        lo, hi = stats.bootstrap_ci(preserved)
        row = {"model": model, "pair": pair, "n_pairs": len(g)}
        for v in ("SAME", "COMPATIBLE", "CONTRADICTS", "NO_ANSWER"):
            row[f"{v.lower()}_pct"] = round(100 * (verdicts == v).mean(), 1)
        row["preserved_pct"] = round(100 * preserved.mean(), 1)
        row["preserved_ci"] = f"{100 * lo:.0f}-{100 * hi:.0f}"
        rows.append(row)
    return pd.DataFrame(rows)


def accuracy_table(feats: pd.DataFrame, reference: dict, suspect: set[str]) -> pd.DataFrame:
    judged = feats[feats["mapped_option"].notna()].copy()
    judged["key"] = judged["item_id"].map(lambda i: reference[i]["correct_answer"])
    rows = []
    for (model, level), g in judged.groupby(["model", "level"], sort=False):
        n_total = int((feats["model"] == model).sum() / len(LEVELS))   # unusable count as wrong
        for subset, sub in (("all_items", g), ("clean_items", g[~g["item_id"].isin(suspect)])):
            denom = n_total if subset == "all_items" else n_total - len(suspect)
            correct = int((sub["mapped_option"] == sub["key"]).sum())
            lo, hi = stats.wilson_ci(correct, denom)
            rows.append({"model": model, "level": level, "items": subset, "n": denom,
                         "correct": correct, "accuracy_pct": round(100 * correct / denom, 1),
                         "accuracy_ci": f"{100 * lo:.0f}-{100 * hi:.0f}",
                         "none_pct": round(100 * (sub["mapped_option"] == "NONE").sum() / denom, 1),
                         "chance_pct": 20.0})
    return pd.DataFrame(rows)


def facts_table(facts: pd.DataFrame, ref_checks: pd.DataFrame | None) -> pd.DataFrame:
    backed = {}
    if ref_checks is not None:
        backed = {r.item_id: json.loads(r.facts_follow_from_explanation) for r in ref_checks.itertuples()}
    rows = []
    for (model, level), g in facts.groupby(["model", "level"], sort=False):
        recall_all, recall_backed, contradicted = [], [], []
        for r in g.itertuples():
            verdicts = json.loads(r.fact_verdicts)
            recall_all.append(verdicts.count("SUPPORTED") / len(verdicts))
            contradicted.append("CONTRADICTED" in verdicts)
            mask = backed.get(r.item_id)
            if mask and any(mask):
                kept = [v for v, m in zip(verdicts, mask) if m]
                recall_backed.append(kept.count("SUPPORTED") / len(kept))
        rows.append({"model": model, "level": level, "n": len(g),
                     "fact_recall_pct": round(100 * np.mean(recall_all), 1),
                     "fact_recall_cot_backed_pct": round(100 * np.mean(recall_backed), 1) if recall_backed else np.nan,
                     "any_contradiction_pct": round(100 * np.mean(contradicted), 1)})
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [f"{v:.3g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_all(models: list[str], features_csv: Path = FEATURES_CSV,
              out_dir: Path = ANALYSIS_DIR) -> dict[str, pd.DataFrame]:
    data = load_inputs(features_csv)
    feats = data["features"]
    reference = load_reference(REFERENCE_JSON)
    ref_checks = data.get("reference")
    suspect = set(ref_checks.loc[_truthy(ref_checks["ref_suspect"]), "item_id"]) if ref_checks is not None else set()

    tables = {
        "usability": usability_table(feats),
        "usability_tests": usability_tests(feats),
        "adaptation": adaptation_table(feats, models),
        "compliance": compliance_table(feats),
    }
    if "consistency" in data:
        tables["consistency"] = consistency_table(data["consistency"])
    if "response" in data:
        tables["accuracy"] = accuracy_table(feats[feats["model"].isin(models)], reference, suspect)
        tables["harm_flags"] = feats.loc[feats["harm"] == "possibly_harmful",
                                         ["model", "level", "item_id", "answer_en", "harm_reason"]]
    if "facts" in data:
        tables["facts"] = facts_table(data["facts"], ref_checks)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    sections = [f"# HiMed-Trad literacy evaluation results\n\nSuspect reference items: {len(suspect)}"]
    for name, df in tables.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False)
        sections.append(f"## {name}\n\n{_md_table(df)}")
    (out_dir / "results.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote {len(tables)} tables to {tables_dir} and {out_dir / 'results.md'}")
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=["himed", "nanda"],
                        help="models for adaptation/accuracy tables (usability covers all)")
    parser.add_argument("--features", type=Path, default=FEATURES_CSV)
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS_DIR)
    args = parser.parse_args()
    build_all(args.models, args.features, args.out_dir)


if __name__ == "__main__":
    main()
