#!/bin/bash
#
# Fetch and chunk StatPearls for the RAG corpus (login node, no SBATCH).
# Clones MedRAG into scratch, downloads the NCBI StatPearls tarball, untars it,
# and runs MedRAG's chunker (src/data/statpearls.py).
#
# Usage (on a PACE login node):
#     bash scripts/rag_fetch_statpearls.sh
#
# Then build the corpus:
#     python -m performance_eval.rag.corpus \
#         --statpearls-dir /home/hice1/gchiu6/scratch/rag/MedRAG/corpus/statpearls/chunk \
#         --out /home/hice1/gchiu6/scratch/rag/corpus.jsonl
#
# All outputs live under /home/hice1/gchiu6/scratch/rag/MedRAG (scratch only).
# Re-running is safe: an existing clone is reused; a partial tarball is deleted and re-downloaded.
# Needs the RAG venv (.venv-rag at the repo root): python -m venv .venv-rag &&
#     .venv-rag/bin/pip install -r requirements-rag.txt

set -euo pipefail

SCRATCH="/home/hice1/gchiu6/scratch"
MEDRAG_DIR="$SCRATCH/rag/MedRAG"
TARBALL_URL="https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"
TARBALL="statpearls_NBK430685.tar.gz"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -f "$REPO_ROOT/.venv-rag/bin/activate" ]; then
    echo "ERROR: $REPO_ROOT/.venv-rag is missing. Create it with:"
    echo "       python -m venv .venv-rag && .venv-rag/bin/pip install -r requirements-rag.txt"
    exit 1
fi
source "$REPO_ROOT/.venv-rag/bin/activate"

mkdir -p "$SCRATCH/rag"

if [ ! -d "$MEDRAG_DIR" ]; then
    git clone https://github.com/Teddy-XiongGZ/MedRAG.git "$MEDRAG_DIR"
fi

cd "$MEDRAG_DIR/corpus/statpearls" 2>/dev/null || {
    mkdir -p "$MEDRAG_DIR/corpus/statpearls"
    cd "$MEDRAG_DIR/corpus/statpearls"
}
rm -f "$TARBALL"
wget "$TARBALL_URL"
{
    echo "tarball: $TARBALL_URL"
    echo "sha256: $(sha256sum "$TARBALL" | cut -d' ' -f1)"
    echo "last_modified: $(curl -sI "$TARBALL_URL" | grep -i '^last-modified:' | sed 's/^[^:]*: *//' | tr -d '\r')"
} | tee "$SCRATCH/rag/statpearls_tarball.txt"
tar -xzf "$TARBALL"

cd "$MEDRAG_DIR"
python src/data/statpearls.py

CHUNK_DIR="$MEDRAG_DIR/corpus/statpearls/chunk"
echo ""
echo "Chunk dir: $CHUNK_DIR"
echo "Total lines: $(cat "$CHUNK_DIR"/*.jsonl | wc -l)"
