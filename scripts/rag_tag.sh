#!/bin/bash
#
# SLURM job: blind question-type tagging of the MILU test items with HiMed-8B.
# The tagger sees only the question text (no answers, no RAG results).
#
# Usage:  sbatch scripts/rag_tag.sh [--limit N]
# Output: /home/hice1/gchiu6/scratch/performance_eval_outputs/milu_hi_med/question_types.csv
#         (+ question_types_raw.csv with the raw model outputs). Re-run to resume.
# Needs HF credentials (MILU is gated); see scripts/performance_milu_run.sh.

# ---------- SLURM resource requests ----------
#SBATCH --job-name=rag-tag
#SBATCH --output=/home/hice1/gchiu6/scratch/logs/rag-tag-%x-%j.out
#SBATCH --error=/home/hice1/gchiu6/scratch/logs/rag-tag-%x-%j.err
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
OUT_CSV="$OUTPUT_DIR/question_types.csv"

echo "=== Question-type tagging ==="
echo "Extra args: $*"
echo "Started:    $(date)"
echo "Node:       $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null \
    || echo "  (nvidia-smi not available)"

cd "$SLURM_SUBMIT_DIR"

export HF_HOME="$SCRATCH/hf_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"
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
python -c "import torch; print('PyTorch:', torch.__version__)"

python -m performance_eval.rag.tagging tag --out "$OUT_CSV" "$@"

echo "=== Job complete at $(date) ==="
echo "Output: $OUT_CSV"
