#!/bin/bash
#
# SLURM job: build the RAG corpus (Textbooks + StatPearls) and its BGE-M3 FAISS index.
#
# Usage:
#     sbatch scripts/rag_build_index.sh [statpearls_chunk_dir]
#     # default chunk dir: $SCRATCH/rag/MedRAG/corpus/statpearls/chunk
#
# Outputs (all under scratch):
#     $SCRATCH/rag/corpus.jsonl
#     $SCRATCH/rag/index_bge_m3/{index.faiss,passage_ids.json,meta.json}
# Logs: $SCRATCH/logs/rag-build-index-*.out / .err

#SBATCH --job-name=rag-build-index
#SBATCH --output=/home/hice1/gchiu6/scratch/logs/rag-build-index-%x-%j.out
#SBATCH --error=/home/hice1/gchiu6/scratch/logs/rag-build-index-%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

SCRATCH="/home/hice1/gchiu6/scratch"
STATPEARLS_DIR="${1:-$SCRATCH/rag/MedRAG/corpus/statpearls/chunk}"
CORPUS="$SCRATCH/rag/corpus.jsonl"
INDEX_DIR="$SCRATCH/rag/index_bge_m3"
LOGS_DIR="$SCRATCH/logs"

mkdir -p "$SCRATCH/rag" "$LOGS_DIR"

echo "=== RAG corpus + index build ==="
echo "StatPearls dir: $STATPEARLS_DIR"
echo "Started:        $(date)"
echo "Node:           $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null \
    || echo "  (nvidia-smi not available)"

if [ ! -d "$STATPEARLS_DIR" ]; then
    echo "ERROR: StatPearls chunk dir not found: $STATPEARLS_DIR"
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"

export HF_HOME="$SCRATCH/hf_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR"

if [ ! -f "$SLURM_SUBMIT_DIR/.venv-rag/bin/activate" ]; then
    echo "ERROR: $SLURM_SUBMIT_DIR/.venv-rag is missing. Create it with:"
    echo "       python -m venv .venv-rag && .venv-rag/bin/pip install -r requirements-rag.txt"
    echo "       (install a CUDA build of torch matching the cluster first)"
    exit 1
fi
source "$SLURM_SUBMIT_DIR/.venv-rag/bin/activate"
echo "Python: $(which python) $(python --version)"
python -c "import faiss, sentence_transformers" || { echo "ERROR: .venv-rag lacks faiss/sentence-transformers; pip install -r requirements-rag.txt"; exit 1; }
python -c "import torch; print('PyTorch:', torch.__version__)"

python -m performance_eval.rag.corpus --statpearls-dir "$STATPEARLS_DIR" --out "$CORPUS"
python -m performance_eval.rag.index --corpus "$CORPUS" --out-dir "$INDEX_DIR"

echo "=== Job complete at $(date) ==="
cat "$INDEX_DIR/meta.json"
