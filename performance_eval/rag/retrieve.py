"""Top-k retrieval (treatment) and random-passage control, written as passages JSONL."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from performance_eval.rag.corpus import read_corpus
from performance_eval.rag.index import Encoder, load_bge_m3, load_index
from performance_eval.rag.query import build_query


def retrieve_topk(items, index, passage_ids, encoder: Encoder, k: int = 5):
    """item_id -> [(passage_id, score)] best-first. One encode call, one search."""
    queries = [build_query(it) for it in items]
    scores, idx = index.search(encoder.encode(queries), k)
    return {
        it["id"]: [(passage_ids[j], float(s)) for j, s in zip(idx[i], scores[i])]
        for i, it in enumerate(items)
    }


def sample_random(items, passage_ids, k: int = 5, seed: int = 42):
    """item_id -> [(passage_id, None)]; per-item RNG, so independent of item order."""
    out = {}
    for it in items:
        rng = random.Random(f"{seed}:{it['id']}")
        out[it["id"]] = [(passage_ids[j], None) for j in rng.sample(range(len(passage_ids)), k)]
    return out


def write_passages(path: Path, condition: str, hits: dict, corpus: dict[str, dict],
                   k: int, seed: int | None, index_meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item_id, plist in hits.items():
            passages = []
            for rank, (pid, score) in enumerate(plist, start=1):
                p = corpus[pid]
                passages.append({"rank": rank, "passage_id": pid, "source": p["source"],
                                 "title": p["title"], "content": p["content"], "score": score})
            f.write(json.dumps({"item_id": item_id, "condition": condition,
                                "passages": passages}, ensure_ascii=False) + "\n")
    meta = {"condition": condition, "k": k, "seed": seed,
            "retriever": index_meta["retriever"], "index_meta": index_meta,
            "query_format": "question+options"}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    from dotenv import load_dotenv

    from performance_eval.data_milu import load_milu_hindi_medicine, split_test_fewshot

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    load_dotenv()
    items = load_milu_hindi_medicine(hf_token=os.environ.get("HF_TOKEN"))
    test, _ = split_test_fewshot(items, fewshot_n=10, seed=args.seed)
    index, ids, meta = load_index(args.index_dir)
    corpus = {p["passage_id"]: p for p in read_corpus(args.corpus)}

    hits = retrieve_topk(test, index, ids, load_bge_m3(), k=args.k)
    write_passages(args.out_dir / "passages_rag_k5.jsonl", "rag_k5", hits, corpus,
                   k=args.k, seed=None, index_meta=meta)
    rand = sample_random(test, ids, k=args.k, seed=args.seed)
    write_passages(args.out_dir / "passages_rag_rand_k5.jsonl", "rag_rand_k5", rand, corpus,
                   k=args.k, seed=args.seed, index_meta=meta)
    print(f"Wrote {len(test)} items x 2 conditions to {args.out_dir}")


if __name__ == "__main__":
    main()
