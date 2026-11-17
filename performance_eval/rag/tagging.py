"""Blind question-type tagging for the MILU test questions.

The type is decided from the question text alone: the tagger prompt and the
hand-check sheet never see gold answers, model answers or retrieval results.
"""

import argparse
import csv
import os
import random
import re
from pathlib import Path
from typing import Callable

TAGS = ("clinical", "public_health", "ayush", "other")
UNTAGGED = "untagged"
HANDCHECK_COLUMNS = ["item_id", "question_with_options", "tag"]

_TAG_RE = re.compile(r"\b(clinical|public[_ ]health|ayush|other)\b", re.IGNORECASE)

_INSTRUCTION = """You are classifying a Hindi medical exam question by topic. Choose exactly one type:
- clinical: clinical or biomedical knowledge (anatomy, physiology, diseases, diagnosis, drugs, treatment).
- public_health: Indian public health, health policy, programmes, epidemiology, or general health knowledge.
- ayush: yoga, Ayurveda, Unani, Siddha, Homeopathy or other traditional medicine.
- other: not medical, or none of the above.

Question:
"""
_SUFFIX = "\n\nReply with one word only: clinical, public_health, ayush, or other."


def build_tag_prompt(item: dict) -> str:
    return _INSTRUCTION + item["question_with_options"] + _SUFFIX


def parse_tag(text: str) -> str | None:
    from literacy_eval.analysis.clean import extract_answer

    response, _ = extract_answer(text or "")
    m = _TAG_RE.search(response)
    if not m:
        return None
    return m.group(1).lower().replace(" ", "_")


def run_tagging(items: list[dict], get_answer_fn: Callable[..., str], out_path: Path) -> None:
    """Append (item_id, tag) to out_path and (item_id, model_output) to the raw CSV
    next to it, one flushed row per item; resumes by item_id."""
    out_path = Path(out_path)
    raw_path = out_path.with_name(out_path.stem + "_raw" + out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path, encoding="utf-8", newline="") as f:
            done = {r["item_id"] for r in csv.DictReader(f)}
    remaining = [it for it in items if it["id"] not in done]
    print(f"{len(done)} done, {len(remaining)} remaining")
    from performance_eval.run import build_gen_kwargs

    gen_kwargs = build_gen_kwargs(False)  # greedy; model default max_new_tokens
    paths = (out_path, raw_path)
    fresh = [not (p.exists() and p.stat().st_size > 0) for p in paths]
    with open(out_path, "a", encoding="utf-8", newline="") as f, \
            open(raw_path, "a", encoding="utf-8", newline="") as g:
        w = csv.writer(f)
        rw = csv.writer(g)
        if fresh[0]:
            w.writerow(["item_id", "tag"])
        if fresh[1]:
            rw.writerow(["item_id", "model_output"])
        for i, item in enumerate(remaining, 1):
            out = get_answer_fn(build_tag_prompt(item), gen_kwargs=gen_kwargs)
            rw.writerow([item["id"], out])
            g.flush()
            w.writerow([item["id"], parse_tag(out) or UNTAGGED])
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(remaining)}")


def write_handcheck(path: Path, items: list[dict], n: int = 50, seed: int = 7) -> None:
    """Blank-tag sheet of n items sampled from sorted ids. Question text only."""
    path = Path(path)
    by_id = {it["id"]: it for it in items}
    ids = random.Random(seed).sample(sorted(by_id), n)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HANDCHECK_COLUMNS)
        for i in ids:
            w.writerow([i, by_id[i]["question_with_options"], ""])


def read_tags(path: Path, strict: bool = False) -> dict[str, str]:
    """Read item_id -> tag. strict (hand sheet): blank or non-TAGS values are errors."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            tag = (row.get("tag") or "").strip().lower()
            if strict:
                if not tag:
                    raise SystemExit(f"item {row['item_id']!r}: blank hand tag in {path}")
                if tag not in TAGS:
                    raise SystemExit(f"item {row['item_id']!r}: invalid hand tag {tag!r}; expected one of {TAGS}")
            out[row["item_id"]] = tag
    return out


def agreement(model_tags: dict, hand_tags: dict) -> dict:
    ids = sorted(set(model_tags) & set(hand_tags))
    n = len(ids)
    if n == 0:
        return {"n": 0, "percent_agree": 0.0, "cohens_kappa": 0.0}
    p_o = sum(model_tags[i] == hand_tags[i] for i in ids) / n
    cats = {model_tags[i] for i in ids} | {hand_tags[i] for i in ids}
    p_e = sum((sum(model_tags[i] == c for i in ids) / n) * (sum(hand_tags[i] == c for i in ids) / n)
              for c in cats)
    if p_e == 1:
        kappa = 1.0 if p_o == 1 else 0.0
    else:
        kappa = (p_o - p_e) / (1 - p_e)
    return {"n": n, "percent_agree": 100 * p_o, "cohens_kappa": kappa}


def _load_test_items(limit: int | None = None) -> list[dict]:
    from dotenv import load_dotenv

    from performance_eval.data_milu import load_milu_hindi_medicine, split_test_fewshot

    load_dotenv()
    items = load_milu_hindi_medicine(hf_token=os.environ.get("HF_TOKEN"))
    test, _ = split_test_fewshot(items, fewshot_n=10, seed=42)
    return test[:limit] if limit is not None else test


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Blind question-type tagging")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tag")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--limit", type=int, default=None)
    h = sub.add_parser("make-handcheck")
    h.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("agreement")
    a.add_argument("--model", type=Path, required=True)
    a.add_argument("--hand", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "tag":
        import importlib

        get_answer = importlib.import_module("models.himed").get_answer  # lazy: imports torch
        run_tagging(_load_test_items(args.limit), get_answer, args.out)
    elif args.cmd == "make-handcheck":
        write_handcheck(args.out, _load_test_items())
    else:
        r = agreement(read_tags(args.model), read_tags(args.hand, strict=True))
        print(f"n={r['n']}")
        print(f"percent_agree={r['percent_agree']:.1f}")
        print(f"cohens_kappa={r['cohens_kappa']:.3f}")


if __name__ == "__main__":
    main()
