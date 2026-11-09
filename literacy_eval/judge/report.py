"""Judge results: validation of the two M-Prometheus judges, then the LLM-output tables (design sections 6, 8).

Reads analysis_outputs/judge_scores.csv (from literacy_eval.judge.collect) and writes
analysis_outputs/judge_results.md plus tables/judge_*.csv. Rows with a judge_error are left out of every
metric and counted in section A.

    python -m literacy_eval.judge.report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from literacy_eval.analysis import stats as S
from literacy_eval.analysis.aggregate import TABLES_DIR
from literacy_eval.analysis.clean import ANALYSIS_DIR, CLEAN_CSV, LEVELS, MODELS
from literacy_eval.judge.collect import SCORES_CSV
from literacy_eval.judge.reference_audit import load_suspect
from literacy_eval.judge.validation import EXPECTED_BAND, LABEL_LEVEL, apply_cutpoints, fit_cutpoints
from literacy_eval.register.analyze import fmt_p, md, per_response, spearman_ci

REPORT_MD = ANALYSIS_DIR / "judge_results.md"
BERT_DIR = Path(__file__).resolve().parents[2] / "register_runs" / "register_corpus" / "indicbert_seed42"
TARGET_LEVEL = {"low": 0, "mid": 1, "high": 2}
PASS_PCT = 80.0
LENGTH_FLAG = 0.5
LEVEL_NAMES = ["5th", "10th", "prof"]


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------

def _ok(scores: pd.DataFrame) -> pd.DataFrame:
    return scores[(scores["judge_error"].fillna("") == "") & scores["score"].notna()]


def _set(scores: pd.DataFrame, name: str) -> pd.DataFrame:
    ok = _ok(scores)
    return ok[ok["set"].fillna("") == name]


def llm_rows(scores: pd.DataFrame) -> pd.DataFrame:
    """Scored LLM-output jobs only: no validation sets, no paraphrase reruns, no judge errors."""
    ok = _ok(scores)
    return ok[(ok["set"].fillna("") == "") & ~ok["job_id"].str.startswith("para:")]


def _pct(mask) -> float:
    mask = np.asarray(mask, dtype=bool)
    return round(100 * mask.mean(), 1) if len(mask) else float("nan")


# ---------------------------------------------------------------------------
# A. Run summary
# ---------------------------------------------------------------------------

def run_summary(scores: pd.DataFrame) -> pd.DataFrame:
    err = scores["judge_error"].fillna("")
    status = np.where(err == "", "ok", np.where(err == "missing", "missing", "error"))
    t = pd.crosstab(scores["task"], status)
    for col in ("ok", "error", "missing"):
        if col not in t:
            t[col] = 0
    return t[["ok", "error", "missing"]].reset_index()


# ---------------------------------------------------------------------------
# B. Validation
# ---------------------------------------------------------------------------

def fit_on_val(scores: pd.DataFrame) -> tuple[float, float] | None:
    val = _set(scores, "corpus:val")
    if val.empty:
        return None
    return fit_cutpoints(val["score"].to_numpy(float), val["label"].map(LABEL_LEVEL).to_numpy())


def _confusion(true: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    m = pd.crosstab(pd.Series(true, name="true"), pd.Series(pred, name="predicted")).reindex(
        index=range(3), columns=range(3), fill_value=0)
    m.index, m.columns = LEVEL_NAMES, LEVEL_NAMES
    return m


def corpus_metrics(scores: pd.DataFrame, cuts: tuple[float, float], split: str = "test") -> dict:
    rows = _set(scores, f"corpus:{split}")
    true = rows["label"].map(LABEL_LEVEL).to_numpy()
    pred = apply_cutpoints(rows["score"].to_numpy(float), cuts)
    return {"n": len(rows), "accuracy": float(np.mean(true == pred)) if len(rows) else float("nan"),
            "kappa": S.weighted_kappa(true, pred), "confusion": _confusion(true, pred),
            "mean_score": rows.groupby("label")["score"].mean().reindex(list(LABEL_LEVEL)).round(2)}


def bert_corpus_metrics(bert_dir: Path | None) -> dict | None:
    path = bert_dir / "test_scores.jsonl" if bert_dir else None
    if not path or not path.exists():
        return None
    b = pd.read_json(path, lines=True)
    true, pred = b["label"].map(LABEL_LEVEL).to_numpy(), b["predicted"].map(LABEL_LEVEL).to_numpy()
    return {"n": len(b), "accuracy": float(np.mean(true == pred)), "kappa": S.weighted_kappa(true, pred)}


def readmepp_metrics(scores: pd.DataFrame, bert_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    r = _set(scores, "readmepp").copy()
    if len(r) < 10:
        return None
    r["rating"] = r["rating"].astype(float)
    rho, lo, hi = spearman_ci(r["score"].to_numpy(float), r["rating"].to_numpy())
    rows = [{"scorer": "M-Prometheus complexity", "n": len(r), "spearman_rho": round(rho, 3),
             "95% CI": f"{lo:.2f} to {hi:.2f}"}]
    path = bert_dir / "readmepp_hi_scores.jsonl" if bert_dir else None
    if path and path.exists():
        b = pd.read_json(path, lines=True)
        ids = r["job_id"].str.split("|", n=2).str[2]
        b = b[b["id"].isin(set(ids))]
        if len(b) >= 10:
            rho, lo, hi = spearman_ci(b["expected_level"].to_numpy(), b["rating"].to_numpy())
            rows.append({"scorer": "IndicBERT expected_level (same sentences)", "n": len(b),
                         "spearman_rho": round(rho, 3), "95% CI": f"{lo:.2f} to {hi:.2f}"})
    by_rating = r.groupby(r["rating"].round().astype(int))["score"].agg(["count", "mean"]).round(2)
    return pd.DataFrame(rows), by_rating.rename_axis("rating")


def topic_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    m = _set(scores, "miracl")
    if len(m):
        share = m["score"].value_counts(normalize=True).reindex(range(1, 6), fill_value=0)
        rows.append({"texts": "MIRACL Wikipedia medical", "n": len(m), "mean_score": round(m["score"].mean(), 2),
                     **{f"share_{k}": round(v, 2) for k, v in share.items()}})
    t = _set(scores, "corpus:test")
    for name, sub in (("NCERT Class 5, health-related (all splits)", _set(scores, "corpus:health")),
                      ("NCERT Class 5, corpus:test (all)", t[t["label"] == "grade5"])):
        if len(sub):
            share = sub["score"].value_counts(normalize=True).reindex(range(1, 6), fill_value=0)
            rows.append({"texts": name, "n": len(sub), "mean_score": round(sub["score"].mean(), 2),
                         **{f"share_{k}": round(v, 2) for k, v in share.items()}})
    return pd.DataFrame(rows)


def length_table(llm: pd.DataFrame, scores: pd.DataFrame, clean: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    if clean is not None and "n_hindi_words" in clean:
        merged = llm.merge(clean[["model", "level", "item_id", "n_hindi_words"]], on=["model", "level", "item_id"])
        for task in ("complexity", "reference_agreement"):
            sub = merged[merged["task"] == task]
            if len(sub) >= 10:
                rho = sps.spearmanr(sub["score"].astype(float), sub["n_hindi_words"]).statistic
                rows.append({"texts": f"LLM outputs ({task})", "n": len(sub), "rho_score_vs_words": round(rho, 2)})
    t = _set(scores, "corpus:test")
    if "n_words" in t:
        for label, sub in t.groupby("label"):
            if len(sub) >= 10:
                rho = sps.spearmanr(sub["score"].astype(float), sub["n_words"].astype(float)).statistic
                rows.append({"texts": f"corpus:test {label}", "n": len(sub), "rho_score_vs_words": round(rho, 2)})
    df = pd.DataFrame(rows)
    if len(df):
        df["flag"] = np.where(df["rho_score_vs_words"].abs() > LENGTH_FLAG, "⚠️ length-driven?", "")
    return df


def controlled_table(scores: pd.DataFrame) -> pd.DataFrame:
    c = _set(scores, "controlled")
    rows = []
    for (task, case), sub in c.groupby(["task", "case"]):
        lo, hi = EXPECTED_BAND[case]
        s = sub["score"].astype(int)
        pct = _pct((s >= lo) & (s <= hi))
        rows.append({"task": task, "case": case, "expected": f"{lo}-{hi}" if lo != hi else str(lo), "n": len(sub),
                     "mean_score": round(s.mean(), 2), "pct_in_band": pct, "pass": "✅" if pct >= PASS_PCT else "❌"})
    return pd.DataFrame(rows)


def paraphrase_table(scores: pd.DataFrame) -> pd.DataFrame:
    ok = _ok(scores)
    para = ok[ok["job_id"].str.startswith("para:")]
    original = ok.set_index("job_id")["score"]
    rows = []
    for task, sub in para.groupby("task"):
        orig = original.reindex(sub["job_id"].str[len("para:"):]).to_numpy(float)
        p = sub["score"].to_numpy(float)
        keep = ~np.isnan(orig)
        rows.append({"task": task, "n": int(keep.sum()), "pct_identical": _pct(p[keep] == orig[keep]),
                     "pct_within_1": _pct(np.abs(p[keep] - orig[keep]) <= 1)})
    return pd.DataFrame(rows)


def _bert_levels(bert_dir: Path | None) -> pd.DataFrame | None:
    path = bert_dir / "llm_window_scores.jsonl" if bert_dir else None
    if not path or not path.exists():
        return None
    resp = per_response(pd.read_json(path, lines=True))
    resp["bert_level"] = resp["predicted"].map(LABEL_LEVEL)
    return resp[["model", "level", "item_id", "bert_level", "expected_level"]]


def judge_vs_bert(llm: pd.DataFrame, cuts, bert: pd.DataFrame | None) -> tuple[dict, pd.DataFrame] | None:
    if bert is None or cuts is None:
        return None
    c = llm[llm["task"] == "complexity"].merge(bert, on=["model", "level", "item_id"])
    if c.empty:
        return None
    judge = apply_cutpoints(c["score"].to_numpy(float), cuts)
    bert_lv = c["bert_level"].to_numpy()
    return ({"n": len(c), "kappa": S.weighted_kappa(bert_lv, judge), "pct_same": _pct(judge == bert_lv)},
            _confusion(bert_lv, judge).rename_axis(index="BERT", columns="judge"))


# ---------------------------------------------------------------------------
# C. LLM outputs
# ---------------------------------------------------------------------------

def _rate(mask) -> float | str:
    """Percentage, or "n too small" for cells below MIN_PAIRED_N (rates on a handful of items mislead)."""
    return _pct(mask) if len(mask) >= S.MIN_PAIRED_N else "n too small"


def exclude_invalid(llm: pd.DataFrame, suspect: set[str] | None) -> pd.DataFrame:
    """Drop invalid dataset items (ref_suspect in the audit) from primary metrics. No audit yet: keep all."""
    return llm if not suspect else llm[~llm["item_id"].isin(suspect)]


def generation_failures(clean: pd.DataFrame | None, suspect: set[str] | None = None) -> pd.DataFrame:
    """Responses with no text (empty output / technical error): never judged, counted per model x level."""
    if clean is None:
        return pd.DataFrame(columns=["model", "level", "n_generation_failures"])
    c = clean if not suspect else clean[~clean["item_id"].isin(suspect)]
    empty = c["answer"].fillna("").astype(str).str.strip() == ""
    return (c[empty].groupby(["model", "level"]).size().rename("n_generation_failures").reset_index())


def llm_main_table(llm: pd.DataFrame, cuts, suspect: set[str] | None,
                   clean: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per model x level. Primary metrics exclude invalid items; `agree_pct_incl_invalid` keeps them."""
    primary = exclude_invalid(llm, suspect)
    failures = generation_failures(clean, suspect).set_index(["model", "level"])["n_generation_failures"]
    rows = []
    for model in MODELS:
        for level in LEVELS:
            cell = primary[(primary["model"] == model) & (primary["level"] == level)]
            n_fail = int(failures.get((model, level), 0))
            if cell.empty and not n_fail:
                continue
            task = {t: cell[cell["task"] == t] for t in
                    ("complexity", "term_explanation", "reference_agreement", "consistency")}
            cx, ag, cs = task["complexity"], task["reference_agreement"], task["consistency"]
            row = {"model": model, "level": level, "n": cx["item_id"].nunique() or cell["item_id"].nunique(),
                   "n_generation_failures": n_fail,
                   "complexity_mean": round(cx["score"].mean(), 2) if len(cx) else float("nan")}
            if cuts is not None and len(cx):
                row["complexity_match_pct"] = _pct(apply_cutpoints(cx["score"].to_numpy(float), cuts) == TARGET_LEVEL[level])
            else:
                row["complexity_match_pct"] = "n/a"
            te = task["term_explanation"]
            row["term_explanation_mean"] = round(te["score"].mean(), 2) if len(te) else ""
            row["agree_pct"] = _rate(ag["score"] >= 4) if len(ag) else ""
            row["contradict_pct"] = _rate(ag["score"] <= 2) if len(ag) else ""
            if suspect is None:
                row["agree_pct_incl_invalid"] = "audit pending"
            else:
                everything = llm[(llm["model"] == model) & (llm["level"] == level) &
                                 (llm["task"] == "reference_agreement")]
                row["agree_pct_incl_invalid"] = _rate(everything["score"] >= 4) if len(everything) else ""
            if len(cs):
                ge4 = (cs["score"] >= 4).to_numpy(float)
                lo, hi = S.bootstrap_ci(ge4)
                small = len(ge4) < S.MIN_PAIRED_N
                row.update(consistency_mean=round(cs["score"].mean(), 2), consistency_pct_ge4=_rate(ge4),
                           consistency_95ci="n too small" if small else f"{100 * lo:.0f}-{100 * hi:.0f}")
            else:
                row.update(consistency_mean="", consistency_pct_ge4="", consistency_95ci="")
            rows.append(row)
    return pd.DataFrame(rows)


def by_failure_type(llm: pd.DataFrame) -> pd.DataFrame:
    """Scores per task for rule-usable answers vs each kind of poor response (all are judged and scored)."""
    ft = llm.get("failure_type", pd.Series("ok", index=llm.index)).fillna("ok").replace({"": "ok"})
    d = llm.assign(failure_type=ft.replace({"ok": "ok (rule-usable)"}))
    g = d.groupby(["task", "failure_type"])["score"]
    out = pd.DataFrame({"n": g.size(), "mean_score": g.mean().round(2),
                        "pct_ge4": g.apply(lambda x: _pct(x >= 4))})
    return out.reset_index()


TASK_ORDER = ["complexity", "term_explanation", "reference_agreement", "consistency"]


def mean_scores(llm: pd.DataFrame) -> pd.DataFrame:
    """Mean judge score (1-5) per task x model x level, with n, SD and a 95% normal CI.

    term_explanation and consistency exist only at low/mid (consistency = that level vs the same model's high).
    """
    g = llm.assign(score=llm["score"].astype(float)).groupby(["task", "model", "level"])["score"]
    out = pd.DataFrame({"n": g.size(), "mean": g.mean(), "sd": g.std()}).reset_index()
    half = 1.96 * out["sd"] / np.sqrt(out["n"])
    out["ci_low"], out["ci_high"] = (out["mean"] - half).clip(lower=1), (out["mean"] + half).clip(upper=5)
    out["task"] = pd.Categorical(out["task"], TASK_ORDER, ordered=True)
    out["model"] = pd.Categorical(out["model"], MODELS, ordered=True)
    out["level"] = pd.Categorical(out["level"], LEVELS, ordered=True)
    out = out.sort_values(["task", "model", "level"]).reset_index(drop=True)
    out[["mean", "sd", "ci_low", "ci_high"]] = out[["mean", "sd", "ci_low", "ci_high"]].round(2)
    return out.astype({"task": str, "model": str, "level": str})


def mean_scores_wide(means: pd.DataFrame) -> pd.DataFrame:
    """One row per model x level, one column per task (mean score); '' where a task does not apply."""
    w = means.pivot_table(index=["model", "level"], columns="task", values="mean", aggfunc="first")
    w = w.reindex(columns=[t for t in TASK_ORDER if t in w.columns])
    idx = pd.MultiIndex.from_product([MODELS, LEVELS], names=["model", "level"])
    return w.reindex(idx).dropna(how="all")


def _wide(llm: pd.DataFrame, model: str) -> pd.DataFrame:
    c = llm[(llm["task"] == "complexity") & (llm["model"] == model)]
    return c.pivot_table(index="item_id", columns="level", values="score", aggfunc="first").reindex(columns=LEVELS)


def paired_tests(llm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        w = _wide(llm, model).astype(float)
        if w.empty:
            continue
        all3 = w.dropna()
        lh = w[["low", "high"]].dropna()
        page = S.page_trend(all3.to_numpy())
        wil = S.wilcoxon_paired(lh["low"].to_numpy(), lh["high"].to_numpy())
        rows.append({"model": model, "n_all_levels": len(all3), "page_p_raw": page["p"], "n_low_high": len(lh),
                     "median_high_minus_low": wil["median_diff"], "share_low_lt_high": round(wil["share_low_lt_high"], 2),
                     "rank_biserial": round(wil["rank_biserial"], 2), "wilcoxon_p_raw": wil["p"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("page_p", "wilcoxon_p"):
        raw = df[f"{col}_raw"]
        valid = raw.notna()
        holm = pd.Series(np.nan, index=df.index)
        if valid.any():
            holm[valid] = S.holm(raw[valid].tolist())
        n_col = "n_all_levels" if col == "page_p" else "n_low_high"
        too_small = df[n_col] < S.MIN_PAIRED_N
        df[col] = np.where(too_small, "n too small", raw.map(fmt_p))
        df[f"{col}_holm"] = np.where(too_small, "n too small", holm.map(fmt_p))
    return df.drop(columns=["page_p_raw", "wilcoxon_p_raw"])


def figure_data(llm: pd.DataFrame, bert: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        w = _wide(llm, model).astype(float)[["low", "high"]].dropna() if not _wide(llm, model).empty else pd.DataFrame()
        cons = llm[(llm["task"] == "consistency") & (llm["model"] == model) & (llm["level"] == "low")]
        row = {"model": model,
               "complexity_shift_high_minus_low": float(np.median(w["high"] - w["low"])) if len(w) else float("nan"),
               "consistency_rate_low_high": _pct(cons["score"] >= 4) if len(cons) else float("nan")}
        if bert is not None:
            b = bert[bert["model"] == model].pivot_table(index="item_id", columns="level", values="expected_level")
            b = b.reindex(columns=LEVELS)[["low", "high"]].dropna()
            row["bert_shift_high_minus_low"] = float(np.median(b["high"] - b["low"])) if len(b) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _provenance(scores: pd.DataFrame) -> str:
    vals = {c: sorted({str(v) for v in scores.get(c, pd.Series(dtype=str)).dropna() if str(v)})
            for c in ("judge_model", "judge_revision", "prompt_version")}
    return ", ".join(f"{k}: {'/'.join(v) or 'n/a'}" for k, v in vals.items())


# Columns only some job types carry. A partial run (e.g. the pilot, with no corpus jobs) lacks them.
_OPTIONAL_COLUMNS = {"set": "", "case": "", "label": "", "health": "", "domain": "",
                     "rating": np.nan, "n_words": np.nan}


def build_report(scores: pd.DataFrame, clean: pd.DataFrame | None, bert_dir: Path | None,
                 suspect: set[str] | None) -> tuple[str, dict[str, pd.DataFrame]]:
    scores = scores.assign(**{c: v for c, v in _OPTIONAL_COLUMNS.items() if c not in scores})
    all_llm = llm_rows(scores)
    llm = exclude_invalid(all_llm, suspect)        # primary evaluation set
    cuts = fit_on_val(scores)
    bert = _bert_levels(bert_dir)
    tables: dict[str, pd.DataFrame] = {}
    L = ["# LLM-judge results (M-Prometheus-14B)", "",
         "Generated by `python -m literacy_eval.judge.report`. Design: `docs/2026-10-27-llm-judge-design.md`. "
         "Scores are LLM-judged estimates, not verified readability or clinical correctness.", ""]

    L += ["## A. Run summary", "", "Jobs per task; only `ok` rows enter any metric.", "",
          md(run_summary(scores).set_index("task")), "", _provenance(scores), ""]

    L += ["## B. Validation", ""]
    L += ["### B1. Score -> level cut-points (fitted on corpus:val)", "",
          "Complexity 1-5 is mapped to 5th / 10th / professional with two cut-points chosen for macro-F1 on the "
          "known-level validation passages.", "",
          f"Cut-points: {cuts}" if cuts else "*Not available: corpus:val not scored yet.*", ""]
    val_rows = []
    if cuts:
        m = corpus_metrics(scores, cuts)
        val_rows.append({"check": "corpus:test accuracy (judge)", "n": m["n"], "value": round(m["accuracy"], 3)})
        val_rows.append({"check": "corpus:test quadratic kappa (judge)", "n": m["n"], "value": round(m["kappa"], 3)})
        L += ["### B2. Known-level test passages (run once)", "",
              "Accuracy and quadratic-weighted kappa against the true class; BERT on the same split for comparison.", "",
              f"Judge: n = {m['n']}, accuracy = {m['accuracy']:.3f}, kappa = {m['kappa']:.3f}."]
        b = bert_corpus_metrics(bert_dir)
        if b:
            L += [f"IndicBERT: n = {b['n']}, accuracy = {b['accuracy']:.3f}, kappa = {b['kappa']:.3f}."]
            val_rows.append({"check": "corpus:test accuracy (BERT)", "n": b["n"], "value": round(b["accuracy"], 3)})
        L += ["", "Confusion (rows = true, columns = judge):", "", md(m["confusion"]), "",
              "Mean complexity score per true class:", "", md(m["mean_score"].to_frame()), ""]
    rm = readmepp_metrics(scores, bert_dir)
    if rm:
        L += ["### B3. ReadMe++ human difficulty ratings", "",
              "Spearman rho between score and human rating (1 easiest ... 6 hardest); > 0 means the ranking agrees.", "",
              md(rm[0].set_index("scorer")), "", "Mean judge score by rating:", "", md(rm[1]), ""]
        val_rows.append({"check": "ReadMe++ spearman (judge)", "n": int(rm[0].loc[0, "n"]), "value": rm[0].loc[0, "spearman_rho"]})
    topic = topic_table(scores)
    if len(topic):
        L += ["### B4. Topic confound", "",
              "Class-5 health passages should still score low; encyclopedic medical text should mostly score 3-5.", "",
              md(topic.set_index("texts")), ""]
    length = length_table(llm, scores, clean)
    if len(length):
        L += ["### B5. Length confound", "", f"Spearman rho of score vs word count; |rho| > {LENGTH_FLAG} is flagged.", "",
              md(length.set_index("texts")), ""]
    ctrl = controlled_table(scores)
    if len(ctrl):
        L += ["### B6. Controlled cases (Judge 2)", "",
              f"Cases built from the reference with a known expected score; ✅ = at least {PASS_PCT:.0f}% in band.", "",
              md(ctrl.set_index(["task", "case"])), ""]
    para = paraphrase_table(scores)
    if len(para):
        L += ["### B7. Prompt sensitivity", "", "Same input, reworded rubric: share of identical scores and within ±1.", "",
              md(para.set_index("task")), ""]
    jb = judge_vs_bert(llm, cuts, bert)
    if jb:
        L += ["### B8. Judge vs IndicBERT on LLM outputs", "",
              f"n = {jb[0]['n']}, quadratic kappa = {jb[0]['kappa']:.3f}, same level = {jb[0]['pct_same']}%.", "",
              md(jb[1]), ""]
    tables["judge_validation"] = pd.DataFrame(val_rows)

    main_t = llm_main_table(all_llm, cuts, suspect, clean)
    failure_t = by_failure_type(llm)
    paired = paired_tests(llm)
    fig = figure_data(llm, bert)
    means = mean_scores(llm)
    tables.update(judge_llm=main_t, judge_paired=paired, judge_figure_data=fig, judge_by_failure_type=failure_t,
                  judge_mean_scores=means)
    L += ["## C. LLM outputs", ""]
    fails = generation_failures(clean, suspect)
    judged = llm[llm["task"] == "complexity"].get("failure_type", pd.Series(dtype=str)).fillna("ok")
    L += ["### C0. What is in the evaluation", "",
          "Following the project guideline: invalid dataset items are excluded from primary metrics (and "
          "reported); generation failures are not judged and are counted separately; poor responses (loops, "
          "echoes, refusals, wrong language) ARE judged and scored.", "",
          "- **Excluded dataset items** (ref_suspect in the reference audit): " +
          ("audit pending — all 60 items are in the primary metrics." if suspect is None else
           f"{len(suspect)}: {', '.join(sorted(suspect)) or 'none'}."),
          f"- **Generation failures** (empty output, not judged): {int(fails['n_generation_failures'].sum()) if len(fails) else 0}.",
          f"- **Responses judged**: {len(judged)} — " +
          (", ".join(f"{k}: {v}" for k, v in judged.value_counts().items()) or "none") +
          " (`ok` = passed the rule-based usability gate).", ""]
    L += ["### C1. Main table", "",
          "One row per model x level. `complexity_match_pct`: judge level (via B1 cut-points) equals the requested "
          "level. `agree_pct` / `contradict_pct`: reference agreement >= 4 / <= 2. Consistency compares each "
          "low/mid answer with the same model's high answer (% >= 4, bootstrap 95% CI).", "",
          md(main_t.set_index(["model", "level"])) if len(main_t) else "*No LLM rows scored yet.*", ""]
    L += ["### C1b. Scores by response type", "",
          "Rule-usable answers vs each kind of poor response, per judge task. Poor responses are part of the "
          "primary metrics; this table shows how much they pull them down.", "",
          md(failure_t.set_index(["task", "failure_type"])) if len(failure_t) else "*No LLM rows scored yet.*", ""]
    L += ["### C1c. Mean judge score per task, model and level", "",
          "Mean of the 1-5 judge scores over the primary set (invalid items excluded, poor responses included). "
          "Higher is: more complex (complexity), better explained (term_explanation), closer to the reference "
          "(reference_agreement), same claim as the model's own high answer (consistency). term_explanation and "
          "consistency are only scored at low/mid. n, SD and 95% CI are in `tables/judge_mean_scores.csv`.", "",
          md(mean_scores_wide(means)) if len(means) else "*No LLM rows scored yet.*", ""]
    L += ["### C2. Paired tests on complexity (low < mid < high)", "",
          "Page's L over items scored at all 3 levels; Wilcoxon low vs high; Holm across models. "
          f"Fewer than {S.MIN_PAIRED_N} items: 'n too small'.", "",
          md(paired.set_index("model")) if len(paired) else "*No complexity rows.*", ""]
    L += ["Figure data (register shift vs consistency) is in `tables/judge_figure_data.csv`; the figure itself is "
          "not drawn here (matplotlib is not installed in `venv-analysis`).", ""]
    return "\n".join(L), tables


def load_scores(path: Path = SCORES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"judge_error": str, "set": str, "feedback": str})
    df["score"] = df["score"].astype("Int64")
    df[["judge_error", "set"]] = df[["judge_error", "set"]].fillna("")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", type=Path, default=SCORES_CSV)
    args = parser.parse_args()
    clean = pd.read_csv(CLEAN_CSV) if CLEAN_CSV.exists() else None
    text, tables = build_report(load_scores(args.scores), clean, BERT_DIR if BERT_DIR.exists() else None, load_suspect())
    REPORT_MD.write_text(text, encoding="utf-8")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(TABLES_DIR / f"{name}.csv", index=False)
    print(f"wrote {REPORT_MD} and {len(tables)} tables in {TABLES_DIR}")


if __name__ == "__main__":
    main()
