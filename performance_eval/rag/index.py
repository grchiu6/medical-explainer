"""Embed the RAG corpus with BGE-M3 and build an exact (flat inner-product) FAISS index.

CLI: python -m performance_eval.rag.index --corpus PATH --out-dir DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from performance_eval.rag.corpus import read_corpus

RETRIEVER = "BAAI/bge-m3"


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return float32 array (len(texts), dim) with L2-normalized rows."""


class _BgeM3Encoder:
    def __init__(self, model) -> None:
        self._model = model

    def encode(self, texts: list[str]) -> np.ndarray:
        emb = self._model.encode(
            texts, batch_size=128, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)


def load_bge_m3(device: str | None = None) -> Encoder:
    from sentence_transformers import SentenceTransformer  # lazy: heavy, not always installed

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(RETRIEVER, device=device)
    model.max_seq_length = 512
    if device.startswith("cuda"):
        model.half()
    return _BgeM3Encoder(model)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_index(corpus_path: Path, out_dir: Path, encoder: Encoder,
                *, batch_size: int = 10_000) -> None:
    corpus_path, out_dir = Path(corpus_path), Path(out_dir)
    passages = read_corpus(corpus_path)
    if not passages:
        raise ValueError(f"empty corpus: {corpus_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    index = None
    for start in range(0, len(passages), batch_size):
        texts = [p["contents"] for p in passages[start:start + batch_size]]
        emb = np.ascontiguousarray(encoder.encode(texts), dtype=np.float32)
        if index is None:
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        print(f"embedded {min(start + batch_size, len(passages))}/{len(passages)}", flush=True)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    (out_dir / "passage_ids.json").write_text(
        json.dumps([p["passage_id"] for p in passages]), encoding="utf-8")
    meta = {"retriever": RETRIEVER, "n": index.ntotal, "dim": index.d,
            "corpus_sha256": _sha256_file(corpus_path)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_index(out_dir: Path) -> tuple[faiss.Index, list[str], dict]:
    out_dir = Path(out_dir)
    index = faiss.read_index(str(out_dir / "index.faiss"))
    ids = json.loads((out_dir / "passage_ids.json").read_text(encoding="utf-8"))
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    return index, ids, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    build_index(args.corpus, args.out_dir, load_bge_m3())


if __name__ == "__main__":
    main()
