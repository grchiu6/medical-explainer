"""Build every M-Prometheus judge prompt into one JSONL file (design section 5).

WHAT A JOB IS: one prompt = one (task, response) = one 1-5 score. Jobs are built here on the Mac,
uploaded to PACE, run by literacy_eval.judge.run, and read back by literacy_eval.judge.collect.

| task                | rows                      | response           | reference                     |
|---------------------|---------------------------|--------------------|-------------------------------|
| complexity          | usable                    | normalize(answer)  | none (blind: generic instr.)  |
| term_explanation    | usable, low/mid           | answer             | none                          |
| reference_agreement | usable                    | answer             | correct option + CoT          |
| consistency         | low/mid AND high usable   | low/mid answer     | same model's high answer      |
| harm                | usable                    | answer             | none                          |

WHY complexity NORMALISES: the register corpus passages it is validated on went through
register.normalize (numbers -> 0, danda fixes), so the LLM outputs must too. Judge 2 keeps the raw
answer, because numbers (doses) matter there. key_facts are LLM-written and unverified: never used.

    python -m literacy_eval.judge.jobs --out data/judge/jobs.jsonl
    python -m literacy_eval.judge.jobs --pilot 5 --out data/judge/pilot_jobs.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from literacy_eval.analysis.clean import CLEAN_CSV, REFERENCE_JSON
from literacy_eval.judge.prompts import build_prompt
from literacy_eval.judge.rubrics import COMPLEXITY_INSTRUCTION, PROMPT_VERSION, RUBRICS
from literacy_eval.register.normalize import normalize

TASKS = ["complexity", "term_explanation", "reference_agreement", "consistency", "harm"]
JOBS_JSONL = Path(__file__).resolve().parents[2] / "data" / "judge" / "jobs.jsonl"


def reference_answer(item: dict) -> str:
    """Dataset text only: the correct option followed by the CoT."""
    return f"{item['correct_option_text']}। {item['cot']}"


def load_reference(path: Path = REFERENCE_JSON) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {item["id"]: item for item in json.load(f)}


def load_clean(path: Path = CLEAN_CSV) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"answer": str}).fillna({"answer": ""})


def _usable(clean: pd.DataFrame) -> pd.DataFrame:
    ok = clean["usable_rules"].astype(str) == "True"
    return clean[ok & (clean["answer"].astype(str).str.strip() != "")]


def _job(task: str, model: str, level: str, item_id: str, prompt: str, **extra) -> dict:
    return {"job_id": f"{task}|{model}|{level}|{item_id}", "task": task, "prompt": prompt,
            "prompt_version": PROMPT_VERSION, "model": model, "level": level, "item_id": item_id, **extra}


def llm_jobs(clean: pd.DataFrame, reference: dict[str, dict], rubrics: dict = RUBRICS) -> list[dict]:
    rows = _usable(clean)
    jobs = []
    for r in rows.itertuples():
        item = reference[r.item_id]
        jobs.append(_job("complexity", r.model, r.level, r.item_id,
                         build_prompt(COMPLEXITY_INSTRUCTION, normalize(r.answer), rubrics["complexity"])))
        if r.level in ("low", "mid"):
            jobs.append(_job("term_explanation", r.model, r.level, r.item_id,
                             build_prompt(r.topic_hi, r.answer, rubrics["term_explanation"])))
        jobs.append(_job("reference_agreement", r.model, r.level, r.item_id,
                         build_prompt(r.topic_hi, r.answer, rubrics["reference_agreement"],
                                      reference=reference_answer(item))))
        jobs.append(_job("harm", r.model, r.level, r.item_id,
                         build_prompt(r.topic_hi, r.answer, rubrics["harm"])))

    answers = {(r.model, r.level, r.item_id): r for r in rows.itertuples()}
    for (model, level, item_id), r in sorted(answers.items()):
        high = answers.get((model, "high", item_id))
        if level == "high" or high is None:
            continue
        pair = f"{level}-high"
        job = _job("consistency", model, pair, item_id,
                   build_prompt(r.topic_hi, r.answer, rubrics["consistency"], reference=high.answer),
                   pair=pair)
        job["level"] = level
        jobs.append(job)
    return jobs


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL. A broken LAST line (a killed job mid-write) is skipped; a broken middle line raises."""
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            if i == len(lines) - 1:
                print(f"  skipping truncated last line of {Path(path).name}")
                break
            raise ValueError(f"{path}: line {i + 1} is not valid JSON") from e
    return rows


def pilot(jobs: list[dict], n: int) -> list[dict]:
    """First n jobs per task, plus every controlled-case job."""
    seen: Counter = Counter()
    kept = []
    for j in jobs:
        if "|controlled:" in j["job_id"]:
            kept.append(j)
        elif seen[j["task"]] < n:
            seen[j["task"]] += 1
            kept.append(j)
    return kept


def build_all_jobs(clean: pd.DataFrame, reference: dict[str, dict]) -> list[dict]:
    from literacy_eval.judge.validation import validation_jobs   # validation imports this module
    llm = llm_jobs(clean, reference)
    return llm + validation_jobs(reference, llm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=JOBS_JSONL)
    parser.add_argument("--pilot", type=int, default=None, help="keep only the first N jobs per task")
    args = parser.parse_args()

    jobs = build_all_jobs(load_clean(), load_reference())
    if args.pilot:
        jobs = pilot(jobs, args.pilot)
    write_jsonl(jobs, args.out)
    counts = Counter(j["job_id"].split("|")[0] + "|" + j["job_id"].split("|")[1].split(":")[0] for j in jobs)
    for key, n in sorted(counts.items()):
        print(f"  {key:40s} {n}")
    print(f"{len(jobs)} jobs -> {args.out}")


if __name__ == "__main__":
    main()
