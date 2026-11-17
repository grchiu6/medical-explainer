#!/bin/bash
#
# SLURM job: retrieve top-5 passages (BGE-M3) and random-5 control for the MILU
# Hindi medicine TEST items. Writes passages_rag_k5.jsonl and
# passages_rag_rand_k5.jsonl (+ .meta.json) under the milu_hi_med output dir.
#
# Usage: sbatch scripts/rag_retrieve.sh
# MILU is gated: needs HF_TOKEN or $HF_HOME/token (see performance_milu_run.sh).

#SBATCH --job-name=rag-retrieve
#SBATCH --output=/home/hice1/gchiu6/scratch/logs/rag-retrieve-%x-%j.out
#SBATCH --error=/home/hice1/gchiu6/scratch/logs/rag-retrieve-%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

SCRATCH="/home/hice1/gchiu6/scratch"
OUTPUT_DIR="$SCRATCH/performance_eval_outputs/milu_hi_med"
LOGS_DIR="$SCRATCH/logs"
mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

echo "=== RAG retrieval ==="
echo "Started: $(date)"
echo "Node:    $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null \
    || echo "  (nvidia-smi not available)"

cd "$SLURM_SUBMIT_DIR"

export HF_HOME="$SCRATCH/hf_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR"

# MILU is gated: existence check only; the token is never read or printed.
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HF_HOME/token" ]; then
    echo "ERROR: no Hugging Face credentials found (MILU is a gated dataset)."
    echo "       Run once on PACE: HF_HOME=/home/hice1/gchiu6/scratch/hf_cache huggingface-cli login"
    echo "       (or export HF_TOKEN=...), then resubmit."
    exit 1
fi

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
echo "Python: $(which python) $(python --version)"

python -m performance_eval.rag.retrieve \
    --index-dir "$SCRATCH/rag/index_bge_m3" \
    --corpus "$SCRATCH/rag/corpus.jsonl" \
    --out-dir "$OUTPUT_DIR"

echo ""
echo "=== Job complete at $(date) ==="
wc -l "$OUTPUT_DIR/passages_rag_k5.jsonl" "$OUTPUT_DIR/passages_rag_rand_k5.jsonl"
