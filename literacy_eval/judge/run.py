"""Run judge jobs through M-Prometheus with vLLM (PACE only; see scripts/judge_run.sh).

Greedy decoding (temperature 0), feedback up to 512 tokens. A reply with no parsable [RESULT] score is
retried ONCE at 1024 tokens; if it still fails it stays in results.jsonl with ok=false and the
collect step reports it as a judge error.

RESUME: results.jsonl is append-only. A job is done once it has an ok row, or a row at the retry
budget. Rerunning the same command after a timeout continues where it stopped. A half-written last
line (job killed mid-write) is ignored and that job is redone.

    python -m literacy_eval.judge.run --jobs ~/scratch/judge/jobs.jsonl --out ~/scratch/judge/results.jsonl
    python -m literacy_eval.judge.run ... --match controlled: --limit 40
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Callable

from literacy_eval.judge.jobs import read_jsonl
from literacy_eval.judge.parse import ParseError, parse_result

MODEL_ID = "Unbabel/M-Prometheus-14B"
MAX_TOKENS = 512
RETRY_MAX_TOKENS = 1024
CHUNK = 256          # prompts per generate() call; rows are flushed after each chunk


def _parses(text: str) -> bool:
    try:
        parse_result(text)
        return True
    except ParseError:
        return False


def _status(out_path: Path, retry_max_tokens: int) -> tuple[set[str], set[str]]:
    """(done job ids, job ids that failed at the first budget and still need the retry)."""
    if not out_path.exists():
        return set(), set()
    done, tried = set(), set()
    for row in read_jsonl(out_path):
        if row.get("ok") or row.get("max_tokens") == retry_max_tokens:
            done.add(row["job_id"])
        else:
            tried.add(row["job_id"])
    return done, tried - done


def run(jobs_path: Path, out_path: Path, generate: Callable[[list[str], int], list[str]],
        judge_model: str, judge_revision: str, max_tokens: int = MAX_TOKENS,
        retry_max_tokens: int = RETRY_MAX_TOKENS, limit: int | None = None,
        match: str | None = None) -> int:
    """Run not-yet-done jobs, appending result rows to out_path. Returns the number of jobs run."""
    jobs = [j for j in read_jsonl(jobs_path) if match is None or match in j["job_id"]]
    done, need_retry = _status(out_path, retry_max_tokens)
    todo = [j for j in jobs if j["job_id"] not in done][:limit]
    print(f"{len(jobs)} jobs, {len(done)} done, running {len(todo)}")
    if not todo:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_date = date.today().isoformat()
    # Drop a half-written last line from a killed job, so it never becomes a broken middle line.
    if out_path.exists() and out_path.stat().st_size:
        text = out_path.read_text(encoding="utf-8")
        if not text.endswith("\n"):
            out_path.write_text(text[: text.rfind("\n") + 1], encoding="utf-8")

    def _generate_and_write(batch: list[dict], budget: int) -> list[dict]:
        failed = []
        with open(out_path, "a", encoding="utf-8") as f:
            for start in range(0, len(batch), CHUNK):
                chunk = batch[start:start + CHUNK]
                outputs = generate([j["prompt"] for j in chunk], budget)
                for job, text in zip(chunk, outputs):
                    ok = _parses(text)
                    f.write(json.dumps({"job_id": job["job_id"], "output": text, "ok": ok, "max_tokens": budget,
                                        "judge_model": judge_model, "judge_revision": judge_revision,
                                        "prompt_version": job.get("prompt_version", ""),
                                        "run_date": run_date}, ensure_ascii=False) + "\n")
                    if not ok:
                        failed.append(job)
                f.flush()
                print(f"  [{min(start + CHUNK, len(batch))}/{len(batch)}] at {budget} tokens")
        return failed

    first = [j for j in todo if j["job_id"] not in need_retry]
    retry = [j for j in todo if j["job_id"] in need_retry]
    retry += _generate_and_write(first, max_tokens) if first else []
    if retry:
        print(f"retrying {len(retry)} unparsable replies at {retry_max_tokens} tokens")
        _generate_and_write(retry, retry_max_tokens)
    return len(todo)


def vllm_generate(model_id: str = MODEL_ID) -> Callable[[list[str], int], list[str]]:
    """Greedy chat generation with vLLM. Imported here so local tests never need vllm."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_id, dtype="bfloat16")

    def generate(prompts: list[str], max_tokens: int) -> list[str]:
        params = SamplingParams(temperature=0, max_tokens=max_tokens)
        convs = [[{"role": "user", "content": p}] for p in prompts]
        return [o.outputs[0].text for o in llm.chat(convs, params, use_tqdm=False)]

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--match", default=None, help="only jobs whose job_id contains this text")
    args = parser.parse_args()

    from huggingface_hub import HfApi
    revision = HfApi().model_info(args.model).sha
    print(f"judge: {args.model} @ {revision}")
    run(args.jobs, args.out, generate=vllm_generate(args.model), judge_model=args.model,
        judge_revision=revision, limit=args.limit, match=args.match)


if __name__ == "__main__":
    main()
