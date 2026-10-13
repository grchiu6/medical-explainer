"""Turn register-classifier scores into results: is the classifier trustworthy, and do the LLMs adapt?

WHAT THIS FILE DOES (reads register_runs/, writes literacy_eval/analysis_outputs/register_results.md):

  A. Training check, per run: test accuracy / macro-F1 and the mean expected_level for each true class
     (a healthy run gives about 0 / 1 / 2). Runs that never left chance are marked FAILED and skipped.
  B. Outside validation, per run, if the score files exist (scripts/register_score.sh):
       ReadMe++  Spearman correlation between expected_level and the human 1-6 difficulty rating
       MIRACL    where Hindi Wikipedia medical passages land (should be 10th/professional, rarely 5th)
  C. The LLM outputs, with the PRIMARY run (default: IndicBERT on the Class 10 corpus):
       - one score per response: windows averaged, weighted by their word counts
       - per model x requested level: n, mean expected_level with a 95% bootstrap CI, share predicted
         as 5th / 10th / professional, and the match rate (low->5th, mid->10th, high->professional)
       - paired tests over the questions answered usably at every compared level (each question is its
         own control): Page's trend test for low < mid < high, Wilcoxon for low vs high; Holm-corrected
         across models
     plus the mean levels from the other passing runs, to show whether conclusions depend on the run.

Needs scipy, so run it with venv-analysis:
    venv-analysis/bin/python -m literacy_eval.register.analyze [--primary register_corpus/indicbert_seed42]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from literacy_eval.analysis.stats import bootstrap_ci, holm, page_trend, wilcoxon_paired

_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = _ROOT / "register_runs"
REPORT_MD = _ROOT / "literacy_eval" / "analysis_outputs" / "register_results.md"
LABELS = ["grade5", "grade10", "professional"]
LEVELS = ["low", "mid", "high"]
TARGET = dict(zip(LEVELS, LABELS))
MODELS = ["himed", "nanda", "apollo", "airavata", "mmed_llama"]
FAILED_MACRO_F1 = 0.40          # below this a run never really learned (chance = 0.17 macro-F1 when one class)


def load_jsonl(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def md(df: pd.DataFrame) -> str:
    df = df.reset_index()
    rows = [" | ".join(map(str, df.columns)), " | ".join("---" for _ in df.columns)]
    rows += [" | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r)
             for r in df.itertuples(index=False)]
    return "\n".join(f"| {r} |" for r in rows)


def fmt_p(p: float) -> str:
    return "n/a" if p is None or np.isnan(p) else (f"{p:.2g}" if p >= 0.001 else "<0.001")


# ---------- A. training check ----------
def training_check(run_dirs: list[Path]) -> tuple[pd.DataFrame, list[Path]]:
    rows, passing = [], []
    for d in run_dirs:
        m = json.loads((d / "metrics.json").read_text())
        test = load_jsonl(d / "test_scores.jsonl")
        means = test.groupby("label")["expected_level"].mean().reindex(LABELS)
        ok = m["test"]["macro_f1"] >= FAILED_MACRO_F1
        passing += [d] if ok else []
        rows.append({"run": run_name(d), "test_accuracy": round(m["test"]["accuracy"], 3),
                     "test_macro_f1": round(m["test"]["macro_f1"], 3),
                     **{f"mean_level_{k}": round(v, 2) for k, v in means.items()},
                     "status": "ok" if ok else "FAILED (never left chance; excluded)"})
    return pd.DataFrame(rows).set_index("run"), passing


def run_name(d: Path) -> str:
    return f"{d.parent.name}/{d.name}"


# ---------- B. outside validation ----------
def spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    rho = sps.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    boots = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        boots.append(sps.spearmanr(x[s], y[s]).statistic)
    return float(rho), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def validation(run_dirs: list[Path]) -> list[str]:
    lines = []
    readme_rows, miracl_rows = [], []
    for d in run_dirs:
        rp = d / "readmepp_hi_scores.jsonl"
        if rp.exists():
            r = load_jsonl(rp)
            rho, lo, hi = spearman_ci(r["expected_level"].to_numpy(), r["rating"].to_numpy())
            row = {"run": run_name(d), "n": len(r), "spearman_rho": round(rho, 3), "95% CI": f"{lo:.2f} to {hi:.2f}"}
            for dom in ("Wikipedia", "Textbooks", "Research and Innovation"):
                sub = r[r["domain"] == dom]
                if len(sub) >= 30:
                    row[f"rho_{dom.split()[0]}"] = round(sps.spearmanr(sub["expected_level"], sub["rating"]).statistic, 2)
            row["mean_level_by_rating_1..6"] = " / ".join(
                f"{v:.2f}" for v in r.groupby("rating")["expected_level"].mean().sort_index())
            readme_rows.append(row)
        mp = d / "miracl_medical_scores.jsonl"
        if mp.exists():
            m = load_jsonl(mp)
            share = m["predicted"].value_counts(normalize=True).reindex(LABELS).fillna(0)
            miracl_rows.append({"run": run_name(d), "n": len(m), "mean_level": round(m["expected_level"].mean(), 2),
                                **{f"share_{k}": round(v, 2) for k, v in share.items()}})
    lines += ["## B. Outside validation (text the classifier never trained on)", ""]
    if readme_rows:
        lines += ["### ReadMe++ Hindi: does the score rank sentences like human difficulty ratings?", "",
                  "Spearman ρ between `expected_level` and the human rating (1 = A1 easiest ... 6 = C2 hardest). "
                  "ρ > 0 means harder-rated sentences get higher levels. ~0.3 is a modest, ~0.5+ a strong agreement. "
                  "Caveat: ReadMe++ items are single sentences (median 16 words), shorter than our 20-200-word "
                  "training passages.", "", md(pd.DataFrame(readme_rows).set_index("run")), ""]
    if miracl_rows:
        lines += ["### Hindi Wikipedia medical passages: where do they land?", "",
                  "Encyclopedic medical text for educated adults: expected mostly 10th / professional, rarely 5th.",
                  "", md(pd.DataFrame(miracl_rows).set_index("run")), ""]
    if not readme_rows and not miracl_rows:
        lines += ["*Not scored yet.* Upload `data/register_eval/*.jsonl` to `~/scratch/register_eval/` and run "
                  "`sbatch scripts/register_score.sh` on PACE, then copy the results back.", ""]
    return lines


# ---------- C. LLM outputs ----------
def per_response(windows: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, level, item_id): word-weighted average of its windows."""
    w = windows.copy()
    w["weight"] = w["n_words"].clip(lower=1)
    cols = ["expected_level"] + [f"p_{k}" for k in LABELS]
    for c in cols:
        w[c] = w[c] * w["weight"]
    g = w.groupby(["model", "level", "item_id"])[cols + ["weight"]].sum()
    for c in cols:
        g[c] = g[c] / g["weight"]
    g["predicted"] = g[[f"p_{k}" for k in LABELS]].to_numpy().argmax(axis=1)
    g["predicted"] = g["predicted"].map(dict(enumerate(LABELS)))
    return g.drop(columns="weight").reset_index()


def llm_tables(resp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for model in MODELS:
        for level in LEVELS:
            sub = resp[(resp["model"] == model) & (resp["level"] == level)]
            if sub.empty:
                continue
            lo, hi = bootstrap_ci(sub["expected_level"].to_numpy())
            share = sub["predicted"].value_counts(normalize=True).reindex(LABELS).fillna(0)
            rows.append({"model": model, "level": level, "n_usable": len(sub),
                         "mean_level": round(sub["expected_level"].mean(), 2), "95% CI": f"{lo:.2f}-{hi:.2f}",
                         "pred_5th": round(share["grade5"], 2), "pred_10th": round(share["grade10"], 2),
                         "pred_prof": round(share["professional"], 2),
                         "match_rate": round((sub["predicted"] == TARGET[level]).mean(), 2)})
    table = pd.DataFrame(rows).set_index(["model", "level"])

    tests = []
    for model in MODELS:
        wide = resp[resp["model"] == model].pivot_table(index="item_id", columns="level", values="expected_level")
        all3 = wide.dropna(subset=[c for c in LEVELS if c in wide]) if set(LEVELS) <= set(wide.columns) else wide.iloc[0:0]
        pt = page_trend(all3[LEVELS].to_numpy()) if len(all3) else {"n": 0, "p": float("nan")}
        pair = wide.dropna(subset=["low", "high"]) if {"low", "high"} <= set(wide.columns) else wide.iloc[0:0]
        wx = wilcoxon_paired(pair["low"].to_numpy(), pair["high"].to_numpy()) if len(pair) else \
            {"n": 0, "p": float("nan"), "rank_biserial": float("nan"), "share_low_lt_high": float("nan"), "median_diff": float("nan")}
        tests.append({"model": model, "items_all_3_levels": pt["n"], "page_p": pt["p"],
                      "items_low_and_high": wx["n"], "median_high_minus_low": wx["median_diff"],
                      "share_low_lt_high": wx["share_low_lt_high"], "rank_biserial": wx["rank_biserial"],
                      "wilcoxon_p": wx["p"]})
    t = pd.DataFrame(tests)
    t["page_p_holm"] = holm(list(t["page_p"]))
    t["wilcoxon_p_holm"] = holm(list(t["wilcoxon_p"]))
    for c in ("page_p", "page_p_holm", "wilcoxon_p", "wilcoxon_p_holm"):
        t[c] = t[c].map(fmt_p)
    for c in ("median_high_minus_low", "share_low_lt_high", "rank_biserial"):
        t[c] = t[c].round(2)
    return table, t.set_index("model")


def robustness(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        resp = per_response(load_jsonl(d / "llm_window_scores.jsonl"))
        means = resp.groupby(["model", "level"])["expected_level"].mean()
        for model in MODELS:
            rows.append({"run": run_name(d), "model": model,
                         **{lvl: round(means.get((model, lvl), np.nan), 2) for lvl in LEVELS}})
    return pd.DataFrame(rows).set_index(["model", "run"]).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--primary", default="register_corpus/indicbert_seed42")
    args = parser.parse_args()

    run_dirs = sorted(p.parent for p in args.runs_dir.glob("*/*_seed42/metrics.json"))
    check, passing = training_check(run_dirs)
    primary = args.runs_dir / args.primary
    if primary not in passing:
        raise SystemExit(f"primary run {args.primary} is missing or FAILED; pass --primary <a passing run>")

    lines = ["# Register classifier results", "",
             "Generated by `venv-analysis/bin/python -m literacy_eval.register.analyze`. Levels: 0 = 5th grade, "
             "1 = 10th grade, 2 = professional (`expected_level` is a continuous score on that scale).", "",
             "## A. Training check (held-out test passages from the training sources)", "",
             "A healthy run gives mean levels near 0 / 1 / 2 for the three true classes. High accuracy here does "
             "NOT by itself show the classifier learned register: the word-counting baseline also scores ~0.98, "
             "partly through topic. Section B is the real check.", "", md(check), ""]
    lines += validation(passing)

    resp = per_response(load_jsonl(primary / "llm_window_scores.jsonl"))
    table, tests = llm_tables(resp)
    lines += [f"## C. LLM outputs (primary run: `{args.primary}`)", "",
              "Only usable responses are scored (refusals, echoes, loops etc. excluded; see `results.md` for "
              "usability). `match_rate` = share classified at the requested level (low->5th, mid->10th, "
              "high->professional). Small `n_usable` (Airavata, Apollo, MMed-Llama) means wide intervals.", "",
              md(table), "",
              "### Paired tests: does the level rise with the requested level, question by question?", "",
              "`page_p`: Page's trend test for low < mid < high, over questions usable at all 3 levels. "
              "`wilcoxon_p`: low vs high. `share_low_lt_high`: share of questions whose low answer scored below its "
              "high answer. `rank_biserial`: effect size from -1 to +1 (+1 = every high answer above its low "
              "answer). p-values Holm-corrected across the 5 models. Tests need >= 10 questions.", "",
              md(tests), "",
              "### Robustness: mean level per run (does the conclusion depend on which classifier?)", "",
              md(robustness(passing)), ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
