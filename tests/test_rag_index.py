import hashlib
import json

import numpy as np

from performance_eval.rag.index import build_index, load_index

DIM = 8


class FakeEncoder:
    """Deterministic: distinct texts -> distinct L2-normalized dim-8 vectors."""

    def encode(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")
            v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out


def _write_corpus(path, n=3):
    rows = [
        {"passage_id": f"textbooks:{i}", "source": "textbooks", "title": f"T{i}",
         "content": f"content {i}", "contents": f"T{i}. content {i} unique text {i}"}
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return rows


def test_build_then_load_roundtrip(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    rows = _write_corpus(corpus, 3)
    out = tmp_path / "idx"
    build_index(corpus, out, FakeEncoder())
    index, ids, meta = load_index(out)
    assert index.ntotal == 3
    assert ids == [r["passage_id"] for r in rows]
    assert meta["n"] == 3
    assert meta["dim"] == DIM
    assert meta["retriever"] == "BAAI/bge-m3"
    assert meta["corpus_sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()


def test_exact_match_ranks_first(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    rows = _write_corpus(corpus, 5)
    out = tmp_path / "idx"
    build_index(corpus, out, FakeEncoder())
    index, ids, _ = load_index(out)
    enc = FakeEncoder()
    for r in rows:
        _, idx = index.search(enc.encode([r["contents"]]), 3)
        assert ids[idx[0][0]] == r["passage_id"]


def test_small_batches_match_single_batch(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, 7)
    build_index(corpus, tmp_path / "a", FakeEncoder())
    build_index(corpus, tmp_path / "b", FakeEncoder(), batch_size=2)
    ia, ids_a, _ = load_index(tmp_path / "a")
    ib, ids_b, _ = load_index(tmp_path / "b")
    assert ib.ntotal == 7 and ids_a == ids_b
    assert np.allclose(ia.reconstruct_n(0, 7), ib.reconstruct_n(0, 7))
