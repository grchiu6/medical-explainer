"""Systematic error-analysis sample of judge decisions (codebook: docs/judge_error_analysis_codebook.md).

SAMPLING: stratified by task x judge score (4 x 5 = 20 strata), PER_STRATUM decisions each (all of them if the
stratum is smaller), drawn with a fixed seed and spread round-robin over the 5 models. Only the primary set is
sampled (invalid items excluded). Each row carries `stratum_n` and `weight` = stratum_n / sampled_n, so rates
found in the sample can be re-weighted to the whole run (`--analyze`).

    python -m literacy_eval.judge.error_sample            # write the review sheet
    python -m literacy_eval.judge.error_sample --analyze  # summarise a filled-in sheet

Import the CSV into Google Sheets: the *_en columns hold =GOOGLETRANSLATE formulas. Fill the review columns
(human_score ... notes), then download as CSV over the same file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from literacy_eval.judge import report as R
from literacy_eval.judge.jobs import load_clean, load_reference, reference_answer
from literacy_eval.judge.reference_audit import load_suspect

SAMPLE_CSV = Path(__file__).resolve().parents[1] / "reference" / "judge_error_sample.csv"
PER_STRATUM = 5
SEED = 20261108
MAX_CHARS = 1500          # long (often looping) answers are cut so GOOGLETRANSLATE can handle them
REVIEW_COLUMNS = ["human_score", "score_verdict", "feedback_faithful", "judge_error", "answer_error", "notes"]
TEXT_COLUMNS = ["question_hi", "answer_hi", "reference_hi"]


def draw(llm: pd.DataFrame, per_stratum: int = PER_STRATUM, seed: int = SEED) -> pd.DataFrame:
    """per_stratum rows per (task, score), round-robin over models, reproducible."""
    d = llm.dropna(subset=["score"]).sample(frac=1, random_state=seed)
    d = d.assign(_rr=d.groupby(["task", "score", "model"]).cumcount())
    d = d.sort_values(["task", "score", "_rr", "model"], kind="stable")
    picked = d.groupby(["task", "score"], group_keys=False).head(per_stratum)
    sizes = llm.groupby(["task", "score"]).size().rename("stratum_n")
    k = picked.groupby(["task", "score"]).size().rename("sampled_n")
    out = picked.join(sizes, on=["task", "score"]).join(k, on=["task", "score"])
    out["weight"] = (out["stratum_n"] / out["sampled_n"]).round(3)
    return out.drop(columns=["_rr", "sampled_n"]).reset_index(drop=True)


def _cut(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + " …[cut]"


def attach_texts(sample: pd.DataFrame, clean: pd.DataFrame, reference: dict[str, dict]) -> pd.DataFrame:
    """What the judge saw: question, answer, and the reference (Judge 2 only)."""
    ans = clean.set_index(["model", "level", "item_id"])
    rows = []
    for r in sample.itertuples():
        a = ans.loc[(r.model, r.level, r.item_id)]
        if r.task == "reference_agreement":
            ref = reference_answer(reference[r.item_id])
        elif r.task == "consistency":
            ref = ans.loc[(r.model, "high", r.item_id), "answer"]
        else:
            ref = ""
        rows.append({"question_hi": a["topic_hi"], "answer_hi": _cut(a["answer"]), "reference_hi": _cut(ref)})
    return pd.concat([sample.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _col_letter(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, rem = divmod(i - 1, 26)
        s = chr(65 + rem) + s
    return s


def review_sheet(sample: pd.DataFrame) -> pd.DataFrame:
    keep = ["job_id", "task", "model", "level", "item_id", "failure_type", "stratum_n", "weight",
            "score", "feedback", *TEXT_COLUMNS]
    sheet = sample[keep].copy()
    sheet.insert(0, "sample_id", range(1, len(sheet) + 1))
    cols = list(sheet.columns)
    for hi in TEXT_COLUMNS:
        letter = _col_letter(cols.index(hi))
        sheet[hi.replace("_hi", "_en")] = [
            f'=GOOGLETRANSLATE({letter}{n + 2}, "hi", "en")' if str(v).strip() else ""
            for n, v in enumerate(sheet[hi])]
    for c in REVIEW_COLUMNS:
        sheet[c] = ""
    return sheet


def export(path: Path = SAMPLE_CSV, per_stratum: int = PER_STRATUM) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path, dtype=str).fillna("")
        if (old[REVIEW_COLUMNS].apply(lambda c: c.str.strip()) != "").any().any():
            raise FileExistsError(f"{path} already has review entries; not overwriting")
    llm = R.exclude_invalid(R.llm_rows(R.load_scores()), load_suspect())
    sheet = review_sheet(attach_texts(draw(llm, per_stratum), load_clean(), load_reference()))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(path, index=False)
    return sheet


def analyze(sheet: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Weighted (= estimate for the whole run) and raw summaries of a filled-in sheet."""
    d = sheet[sheet["score_verdict"].astype(str).str.strip() != ""].copy()
    d["weight"] = d["weight"].astype(float)
    for c in ("score_verdict", "feedback_faithful", "judge_error", "answer_error"):
        d[c] = d[c].astype(str).str.strip().str.lower()

    def wshare(col: str) -> pd.DataFrame:
        """Weighted % per task: an estimate of the share in the whole run, not just the sample."""
        t = pd.crosstab(d["task"], d[col], values=d["weight"], aggfunc="sum", normalize="index")
        return (t * 100).round(1)

    out = {}
    out["verdict_by_task"] = wshare("score_verdict")
    out["judge_error_by_task"] = wshare("judge_error")
    out["judge_error_counts"] = pd.crosstab(d["judge_error"], d["task"], margins=True)
    out["answer_error_by_model"] = pd.crosstab(d["answer_error"], d["model"], margins=True)
    out["feedback_faithful_by_task"] = pd.crosstab(d["task"], d["feedback_faithful"], margins=True)
    hs = pd.to_numeric(d["human_score"], errors="coerce")
    agree = d.assign(human=hs).dropna(subset=["human"])
    if len(agree):
        js = agree["score"].astype(float)
        out["score_agreement_by_task"] = agree.assign(exact=(js == agree["human"]), within1=(abs(js - agree["human"]) <= 1)) \
            .groupby("task")[["exact", "within1"]].mean().mul(100).round(1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--analyze", action="store_true", help="summarise the filled-in sheet")
    parser.add_argument("--per-stratum", type=int, default=PER_STRATUM)
    args = parser.parse_args()
    if args.analyze:
        for name, t in analyze(pd.read_csv(SAMPLE_CSV, dtype=str).fillna("")).items():
            print(f"\n## {name}\n{t.to_string()}")
        return
    sheet = export(per_stratum=args.per_stratum)
    print(f"wrote {len(sheet)} decisions -> {SAMPLE_CSV}")
    print(sheet.groupby(["task", "score"]).size().unstack(fill_value=0).to_string())
    print(sheet.groupby("model").size().to_string())


if __name__ == "__main__":
    main()
