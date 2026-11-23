"""Top-k retrieval (treatment) and random-passage control, written as passages JSONL."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from performance_eval.rag.corpus import read_corpus
from performance_eval.rag.index import Encoder, _sha256_file, load_bge_m3, load_index
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


def english_query_items(test: list[dict], translations_path: Path, exclude_file: Path) -> list[dict]:
    """Test items after exclusions, with question/options replaced by their English translation
    (same ids). Used only to build retrieval queries; the answering prompt stays Hindi."""
    from performance_eval.data_milu import load_english_items
    from performance_eval.exclusions import apply_exclusions, load_exclusions

    return load_english_items(translations_path, apply_exclusions(test, load_exclusions(exclude_file)))


def check_corpus_matches_index(corpus_path: Path, index_meta: dict) -> None:
    """Exit unless the corpus file is the one the index was built from."""
    actual = _sha256_file(Path(corpus_path))
    expected = index_meta.get("corpus_sha256")
    if actual != expected:
        raise SystemExit(f"corpus {corpus_path} (sha256 {actual[:12]}) does not match the corpus the "
                         f"index was built from (sha256 {str(expected)[:12]}); rebuild the index or "
                         "pass the matching --corpus")


def write_passages(path: Path, condition: str, hits: dict, corpus: dict[str, dict],
                   k: int, seed: int | None, index_meta: dict, split_seed: int | None = None,
                   query_language: str = "hi") -> None:
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
    meta = {"condition": condition, "k": k, "seed": seed, "split_seed": split_seed,
            "retriever": index_meta["retriever"], "index_meta": index_meta,
            "query_format": "question+options", "query_language": query_language}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    from dotenv import load_dotenv

    from performance_eval.data_milu import load_milu_hindi_medicine, split_test_fewshot

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=42,
                    help="seed of split_test_fewshot (must match the run.py --seed)")
    ap.add_argument("--seed", type=int, default=42, help="random-control seed only")
    ap.add_argument("--query-translations", type=Path, default=None,
                    help="translate-then-retrieve: build queries from these English translations "
                         "(writes passages_rag_k5_enquery.jsonl only; requires --exclude-file)")
    ap.add_argument("--exclude-file", type=Path, default=None)
    args = ap.parse_args()
    if args.query_translations is not None and args.exclude_file is None:
        ap.error("--query-translations requires --exclude-file (translations cover only the kept items)")

    load_dotenv()
    items = load_milu_hindi_medicine(hf_token=os.environ.get("HF_TOKEN"))
    test, _ = split_test_fewshot(items, fewshot_n=10, seed=args.split_seed)
    index, ids, meta = load_index(args.index_dir)
    check_corpus_matches_index(args.corpus, meta)
    corpus = {p["passage_id"]: p for p in read_corpus(args.corpus)}

    if args.query_translations is not None:
        items = english_query_items(test, args.query_translations, args.exclude_file)
        hits = retrieve_topk(items, index, ids, load_bge_m3(), k=args.k)
        write_passages(args.out_dir / "passages_rag_k5_enquery.jsonl", "rag_k5_enquery", hits, corpus,
                       k=args.k, seed=None, index_meta=meta, split_seed=args.split_seed, query_language="en")
        print(f"Wrote {len(items)} items (English queries) to {args.out_dir}")
        return

    hits = retrieve_topk(test, index, ids, load_bge_m3(), k=args.k)
    write_passages(args.out_dir / "passages_rag_k5.jsonl", "rag_k5", hits, corpus,
                   k=args.k, seed=None, index_meta=meta, split_seed=args.split_seed)
    rand = sample_random(test, ids, k=args.k, seed=args.seed)
    write_passages(args.out_dir / "passages_rag_rand_k5.jsonl", "rag_rand_k5", rand, corpus,
                   k=args.k, seed=args.seed, index_meta=meta,
                   split_seed=args.split_seed)
    print(f"Wrote {len(test)} items x 2 conditions to {args.out_dir}")


if __name__ == "__main__":
    main()
