"""Validation jobs for the two judges, built so that no Hindi reader is needed (design section 6).

  corpus:val / corpus:test   NCERT Class 5 / Class 10 / ICMR passages with known level. val fits the
                             score -> level cut-points; test is reported ONCE, after prompts are frozen.
  readmepp                   300 ReadMe++ Hindi sentences with human difficulty ratings (1-6).
  corpus:health              health-related Class 5 passages from train/val/test (topic-confound check).
  miracl                     100 Hindi Wikipedia medical passages (topic-confound check).
  controlled:<case>          test cases built from the reference itself, with a known expected score:
                               positive   correct option + CoT                         -> 5
                               swapped    correct option text replaced by a wrong one   -> 1-2
                               unrelated  another item's reference answer              -> 1-3
                               non_answer the question restated                        -> 3
                             Run for reference_agreement (reference = correct option + CoT) and
                             consistency (reference = the positive text, standing in for "the high answer").
  para:<job_id>              10% of the LLM-output jobs with the PARAPHRASED rubric (prompt sensitivity).
"""

from __future__ import annotations

import random
from itertools import combinations
from pathlib import Path

import numpy as np

from literacy_eval.analysis.clean import question_stem
from literacy_eval.judge.jobs import read_jsonl, reference_answer
from literacy_eval.judge.prompts import build_prompt, format_rubric
from literacy_eval.judge.rubrics import COMPLEXITY_INSTRUCTION, PARAPHRASED, PROMPT_VERSION, RUBRICS
from literacy_eval.register.clean_sources import health_hits

_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = _ROOT / "data" / "register_corpus"
EVAL_DIR = _ROOT / "data" / "register_eval"
SEED = 42
READMEPP_N = 300
MIRACL_N = 100
PARAPHRASE_SHARE = 0.10

LABEL_LEVEL = {"grade5": 0, "grade10": 1, "professional": 2}
EXPECTED_BAND = {"positive": (5, 5), "swapped": (1, 2), "unrelated": (1, 3), "non_answer": (3, 3)}
_CUT_CANDIDATES = (1.5, 2.5, 3.5, 4.5)


# ---------------------------------------------------------------------------
# Controlled cases
# ---------------------------------------------------------------------------

def _wrong_option(item: dict) -> str:
    letter = sorted(k for k in item["options"] if k != item["correct_answer"])[0]
    return item["options"][letter]


def controlled_cases(reference: dict[str, dict]) -> list[dict]:
    ids = sorted(reference)
    cases = []
    for k, item_id in enumerate(ids):
        item = reference[item_id]
        positive = reference_answer(item)
        correct, wrong = item["correct_option_text"], _wrong_option(item)
        if correct in item["cot"]:
            swapped = positive.replace(correct, wrong)
        else:
            swapped = f"{wrong}। {item['cot']}"
        responses = {
            "positive": positive,
            "swapped": swapped,
            "unrelated": reference_answer(reference[ids[(k + 1) % len(ids)]]),
            "non_answer": question_stem(item["topic_hi"]),
        }
        for case in EXPECTED_BAND:
            cases.append({"case": case, "item_id": item_id, "response": responses[case],
                          "positive_text": positive})
    return cases


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

def readmepp_sample(rows: list[dict], n: int = READMEPP_N, seed: int = SEED) -> list[dict]:
    """n sentences, as equal a share per rounded rating as the data allows; the rest filled at random."""
    rng = random.Random(seed)
    by_rating: dict[int, list[dict]] = {}
    for r in rows:
        by_rating.setdefault(round(float(r["rating"])), []).append(r)
    for group in by_rating.values():
        rng.shuffle(group)
    per = n // len(by_rating)
    chosen = [r for g in by_rating.values() for r in g[:per]]
    leftover = [r for g in by_rating.values() for r in g[per:]]
    rng.shuffle(leftover)
    chosen += leftover[: max(0, n - len(chosen))]
    return chosen[:n]


def _complexity_job(job_id: str, text: str, **meta) -> dict:
    return {"job_id": job_id, "task": "complexity", "prompt_version": PROMPT_VERSION,
            "prompt": build_prompt(COMPLEXITY_INSTRUCTION, text, RUBRICS["complexity"]),
            "n_words": len(text.split()), **meta}


def _set_jobs(corpus_dir: Path, eval_dir: Path) -> list[dict]:
    jobs = []
    for split in ("val", "test"):
        for i, r in enumerate(read_jsonl(corpus_dir / f"{split}.jsonl")):
            jobs.append(_complexity_job(f"complexity|corpus:{split}|{i}", r["text"], set=f"corpus:{split}",
                                        label=r["label"], source=r.get("source", ""),
                                        health=health_hits(r["text"]) > 0))
    # Topic-confound check (6.3): health-related Class-5 passages pooled from ALL splits (test alone has ~1).
    # Train passages are safe here: the judge never trained on them.
    for split in ("train", "val", "test"):
        path = corpus_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for i, r in enumerate(read_jsonl(path)):
            if r["label"] == "grade5" and health_hits(r["text"]) > 0:
                jobs.append(_complexity_job(f"complexity|corpus:health|{split}:{i}", r["text"],
                                            set="corpus:health", label="grade5", health=True))
    for r in readmepp_sample(read_jsonl(eval_dir / "readmepp_hi.jsonl")):
        jobs.append(_complexity_job(f"complexity|readmepp|{r['id']}", r["text"], set="readmepp",
                                    rating=float(r["rating"]), domain=r.get("domain", "")))
    miracl = read_jsonl(eval_dir / "miracl_medical.jsonl")
    for r in random.Random(SEED).sample(miracl, min(MIRACL_N, len(miracl))):
        jobs.append(_complexity_job(f"complexity|miracl|{r['id']}", r["text"], set="miracl"))
    return jobs


def _controlled_jobs(reference: dict[str, dict]) -> list[dict]:
    jobs = []
    for c in controlled_cases(reference):
        item = reference[c["item_id"]]
        refs = {"reference_agreement": reference_answer(item), "consistency": c["positive_text"]}
        for task, ref in refs.items():
            jobs.append({"job_id": f"{task}|controlled:{c['case']}|{c['item_id']}", "task": task,
                         "prompt_version": PROMPT_VERSION, "set": "controlled", "case": c["case"],
                         "item_id": c["item_id"],
                         "prompt": build_prompt(item["topic_hi"], c["response"], RUBRICS[task], reference=ref)})
    return jobs


def _paraphrase_jobs(llm_job_list: list[dict]) -> list[dict]:
    k = int(round(PARAPHRASE_SHARE * len(llm_job_list)))
    jobs = []
    for j in random.Random(SEED).sample(llm_job_list, k):
        original = format_rubric(RUBRICS[j["task"]])
        if original not in j["prompt"]:
            raise ValueError(f"{j['job_id']}: prompt does not contain its rubric")
        jobs.append({**j, "job_id": f"para:{j['job_id']}", "set": "paraphrase",
                     "prompt": j["prompt"].replace(original, format_rubric(PARAPHRASED[j["task"]]))})
    return jobs


def validation_jobs(reference: dict[str, dict], llm_job_list: list[dict], corpus_dir: Path = CORPUS_DIR,
                    eval_dir: Path = EVAL_DIR, include_sets: bool = True) -> list[dict]:
    jobs = _set_jobs(corpus_dir, eval_dir) if include_sets else []
    return jobs + _controlled_jobs(reference) + _paraphrase_jobs(llm_job_list)


# ---------------------------------------------------------------------------
# Score -> level cut-points
# ---------------------------------------------------------------------------

def apply_cutpoints(scores: np.ndarray, cuts: tuple[float, float]) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    return np.where(scores < cuts[0], 0, np.where(scores < cuts[1], 1, 2))


def _macro_f1(true: np.ndarray, pred: np.ndarray, n_levels: int = 3) -> float:
    f1s = []
    for c in range(n_levels):
        tp = np.sum((pred == c) & (true == c))
        denom = np.sum(pred == c) + np.sum(true == c)
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def fit_cutpoints(scores: np.ndarray, levels: np.ndarray) -> tuple[float, float]:
    """Pair (c1 < c2) from {1.5, 2.5, 3.5, 4.5} with the best macro-F1; ties go closest to (2.5, 3.5)."""
    levels = np.asarray(levels)
    best = max(combinations(_CUT_CANDIDATES, 2),
               key=lambda c: (round(_macro_f1(levels, apply_cutpoints(scores, c)), 12),
                              -(abs(c[0] - 2.5) + abs(c[1] - 3.5))))
    return best
