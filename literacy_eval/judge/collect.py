"""Turn PACE results (results.jsonl) + the jobs that produced them into one scores table.

One row per job: the job's metadata (without the prompt), the judge's English feedback, its 1-5 score
and provenance. Per job the latest ok row wins; otherwise the latest row (judge_error says why).
A job with no result row at all gets judge_error="missing" — resubmit the PACE job to fill it.

    python -m literacy_eval.judge.collect
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from literacy_eval.analysis.clean import ANALYSIS_DIR
from literacy_eval.judge.jobs import JOBS_JSONL, read_jsonl
from literacy_eval.judge.parse import ParseError, parse_result

RESULTS_JSONL = JOBS_JSONL.parent / "results.jsonl"
SCORES_CSV = ANALYSIS_DIR / "judge_scores.csv"
_PROVENANCE = ["judge_model", "judge_revision", "prompt_version", "run_date", "max_tokens"]


def collect(jobs_path: Path = JOBS_JSONL, results_path: Path = RESULTS_JSONL) -> pd.DataFrame:
    latest: dict[str, dict] = {}
    for row in read_jsonl(results_path):
        if row.get("ok") or not latest.get(row["job_id"], {}).get("ok"):
            latest[row["job_id"]] = row

    out = []
    for job in read_jsonl(jobs_path):
        rec = {k: v for k, v in job.items() if k != "prompt"}
        res = latest.get(job["job_id"])
        if res is None:
            rec.update(feedback="", score=None, judge_error="missing")
        else:
            rec.update({k: res.get(k, "") for k in _PROVENANCE})
            rec["prompt_version"] = res.get("prompt_version") or job.get("prompt_version", "")
            try:
                feedback, score = parse_result(res["output"])
                rec.update(feedback=feedback, score=score, judge_error="")
            except ParseError as e:
                rec.update(feedback=res["output"], score=None, judge_error=str(e)[:300])
        out.append(rec)
    df = pd.DataFrame(out)
    df["score"] = df["score"].astype("Int64")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=Path, default=JOBS_JSONL)
    parser.add_argument("--results", type=Path, default=RESULTS_JSONL)
    parser.add_argument("--out", type=Path, default=SCORES_CSV)
    args = parser.parse_args()

    df = collect(args.jobs, args.results)
    df.to_csv(args.out, index=False)
    status = df.assign(status=df["judge_error"].map(lambda e: "ok" if not e else ("missing" if e == "missing" else "error")))
    print(status.groupby(["task", "status"]).size().unstack(fill_value=0).to_string())
    print(f"{len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
