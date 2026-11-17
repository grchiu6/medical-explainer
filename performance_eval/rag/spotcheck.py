"""Retrieval spot-check: sheet for manual relevance labels and the go/no-go gate."""

import argparse
import csv
import os
import random
from pathlib import Path

from performance_eval.prompts import PASSAGE_MAX_CHARS

COLUMNS = ["item_id", "question_with_options",
           "passage_1", "passage_2", "passage_3", "passage_4", "passage_5", "relevance"]
LABELS = ("relevant", "partial", "none")
GATE_MIN_RELEVANT = 10


def sample_spotcheck_ids(test_item_ids: list[str], n: int = 20, seed: int = 42) -> list[str]:
    return random.Random(seed).sample(sorted(test_item_ids), n)


def write_sheet(path: Path, items: list[dict], passages: dict) -> None:
    """One row per item, in the given order. No gold answer or model output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for it in items:
            cells = [f"{p['title']}\n{p['content'][:PASSAGE_MAX_CHARS]}" for p in passages[it["id"]]]
            cells += [""] * (5 - len(cells))
            w.writerow([it["id"], it["question_with_options"], *cells[:5], ""])


def summarize(path: Path, expected_n: int = 20) -> dict:
    counts = {l: 0 for l in LABELS}
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r.get("item_id") for r in rows]
    if len(rows) != expected_n:
        raise ValueError(f"expected {expected_n} rows, found {len(rows)} (truncated or wrong sheet?)")
    if len(set(ids)) != len(ids):
        raise ValueError("item_ids are not unique (duplicate rows in sheet)")
    for row in rows:
        label = (row.get("relevance") or "").strip().lower()
        if label not in counts:
            raise ValueError(
                f"item {row.get('item_id')!r}: invalid relevance label {row.get('relevance')!r}; "
                f"expected one of {LABELS}")
        counts[label] += 1
    return {**counts, "passes_gate": counts["relevant"] >= GATE_MIN_RELEVANT}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--passages", type=Path, required=True)
    mk.add_argument("--out", type=Path, required=True)
    mk.add_argument("--n", type=int, default=20)
    mk.add_argument("--seed", type=int, default=42)
    sm = sub.add_parser("summarize")
    sm.add_argument("sheet", type=Path)
    sm.add_argument("--n", type=int, default=20)
    args = ap.parse_args(argv)

    if args.cmd == "make":
        from dotenv import load_dotenv

        from performance_eval.data_milu import load_milu_hindi_medicine, split_test_fewshot
        from performance_eval.run import load_passages

        load_dotenv()
        items = load_milu_hindi_medicine(hf_token=os.environ.get("HF_TOKEN"))
        test, _ = split_test_fewshot(items)
        by_id = {it["id"]: it for it in test}
        ids = sample_spotcheck_ids(list(by_id), n=args.n, seed=args.seed)
        passages = load_passages(args.passages, ids, allow_extra=True)
        write_sheet(args.out, [by_id[i] for i in ids], passages)
        print(f"wrote {len(ids)} rows to {args.out}")
    else:
        r = summarize(args.sheet, expected_n=args.n)
        print(f"relevant={r['relevant']} partial={r['partial']} none={r['none']}")
        print("GATE: PASS" if r["passes_gate"] else "GATE: FAIL — stop and return to design")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
