import json

import pytest

from performance_eval.rag.index import build_index, load_index
from performance_eval.rag.retrieve import retrieve_topk, sample_random, write_passages
from tests.test_rag_index import FakeEncoder, _write_corpus

N = 12
ITEMS = [
    {"id": f"it{i}", "question": f"question {i}",
     "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer_letter": "A"}
    for i in range(6)
]
PIDS = [f"textbooks:{i}" for i in range(N)]


@pytest.fixture
def idx(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    rows = _write_corpus(corpus, N)
    build_index(corpus, tmp_path / "idx", FakeEncoder())
    index, ids, meta = load_index(tmp_path / "idx")
    return index, ids, meta, {r["passage_id"]: r for r in rows}


def test_topk_returns_k_best_first(idx):
    index, ids, _, _ = idx
    hits = retrieve_topk(ITEMS, index, ids, FakeEncoder(), k=5)
    assert list(hits) == [it["id"] for it in ITEMS]
    for v in hits.values():
        assert len(v) == 5
        scores = [s for _, s in v]
        assert scores == sorted(scores, reverse=True)
        assert all(p in ids for p, _ in v)


def test_topk_ignores_answer_letter(idx):
    index, ids, _, _ = idx
    a = retrieve_topk(ITEMS, index, ids, FakeEncoder(), k=5)
    changed = [{**it, "answer_letter": "D"} for it in ITEMS]
    assert retrieve_topk(changed, index, ids, FakeEncoder(), k=5) == a


def test_random_is_deterministic_distinct_and_order_independent():
    a = sample_random(ITEMS, PIDS, k=5, seed=42)
    b = sample_random(list(reversed(ITEMS)), PIDS, k=5, seed=42)
    assert a == b and all(len({p for p, _ in v}) == 5 for v in a.values())
    assert all(s is None for v in a.values() for _, s in v)


def test_write_passages_schema(tmp_path, idx):
    index, ids, meta, corpus = idx
    hits = retrieve_topk(ITEMS, index, ids, FakeEncoder(), k=5)
    path = tmp_path / "passages_rag_k5.jsonl"
    write_passages(path, "rag_k5", hits, corpus, k=5, seed=None, index_meta=meta)
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert [l["item_id"] for l in lines] == [it["id"] for it in ITEMS]
    for l in lines:
        assert l["condition"] == "rag_k5"
        assert [p["rank"] for p in l["passages"]] == [1, 2, 3, 4, 5]
        assert set(l["passages"][0]) == {"rank", "passage_id", "source", "title", "content", "score"}
    m = json.loads(path.with_suffix(".meta.json").read_text())
    assert m["k"] == 5 and m["query_format"] == "question+options"
    assert m["seed"] is None and m["retriever"] == meta["retriever"] and m["index_meta"] == meta


def test_write_random_has_null_scores(tmp_path, idx):
    _, ids, meta, corpus = idx
    hits = sample_random(ITEMS, ids, k=5, seed=42)
    path = tmp_path / "r.jsonl"
    write_passages(path, "rag_rand_k5", hits, corpus, k=5, seed=42, index_meta=meta)
    l = json.loads(path.read_text().splitlines()[0])
    assert all(p["score"] is None for p in l["passages"])
    assert json.loads(path.with_suffix(".meta.json").read_text())["seed"] == 42
