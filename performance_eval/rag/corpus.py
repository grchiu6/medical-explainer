"""Build the combined Textbooks + StatPearls retrieval corpus (JSONL).

Each passage dict: {"passage_id", "source", "title", "content", "contents"}
with passage_id = "<source>:<original id>". Both sources come from MedRAG.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_FETCH_HINT = "scripts/rag_fetch_statpearls.sh"


def _passage(source: str, row: dict) -> dict:
    return {
        "passage_id": f"{source}:{row['id']}",
        "source": source,
        "title": row["title"],
        "content": row["content"],
        "contents": row["contents"],
    }


def textbook_row_to_passage(row: dict) -> dict:
    return _passage("textbooks", row)


def load_textbooks() -> list[dict]:
    """Download MedRAG/textbooks from the HF hub (network)."""
    from datasets import load_dataset

    ds = load_dataset("MedRAG/textbooks", split="train")
    return [textbook_row_to_passage(row) for row in ds]


def load_statpearls(chunk_dir: Path) -> list[dict]:
    """Read MedRAG's chunked StatPearls *.jsonl files in sorted filename order."""
    chunk_dir = Path(chunk_dir)
    if not chunk_dir.is_dir():
        sys.exit(f"StatPearls chunk dir not found: {chunk_dir}. Run {_FETCH_HINT} first.")
    rows = []
    for path in sorted(chunk_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            rows.extend(_passage("statpearls", json.loads(line)) for line in f if line.strip())
    if not rows:
        sys.exit(f"No StatPearls rows under {chunk_dir}. Run {_FETCH_HINT} first.")
    return rows


def build_corpus(textbooks: list[dict], statpearls: list[dict], out_path: Path) -> int:
    """Write textbooks then statpearls to JSONL; return the row count."""
    passages = list(textbooks) + list(statpearls)
    counts = Counter(p["passage_id"] for p in passages)
    dupes = sorted(i for i, c in counts.items() if c > 1)
    if dupes:
        raise ValueError(f"duplicate passage_ids: {dupes[:5]}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return len(passages)


def read_corpus(path: Path) -> list[dict]:
    """Read the JSONL written by build_corpus, in file order."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--statpearls-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    textbooks = load_textbooks()
    statpearls = load_statpearls(args.statpearls_dir)
    total = build_corpus(textbooks, statpearls, args.out)
    print(f"textbooks: {len(textbooks)}")
    print(f"statpearls: {len(statpearls)}")
    print(f"total: {total} -> {args.out}")


if __name__ == "__main__":
    main()
