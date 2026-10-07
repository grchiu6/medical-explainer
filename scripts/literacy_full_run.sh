#!/bin/bash
#
# SLURM FULL RUN script for literacy-level evaluation.
# Runs all 30 topics × 3 levels for ONE model.
# Submit this script TWICE — once for airavata, once for apollo.
# Both jobs can run in parallel if GPUs are available.
#
# Usage:
#     sbatch scripts/literacy_full_run.sh airavata
#     sbatch scripts/literacy_full_run.sh apollo
#
# Output CSVs (one per level):
#     /home/hice1/gchiu6/scratch/literacy_eval_outputs/airavata_low_raw.csv
#     /home/hice1/gchiu6/scratch/literacy_eval_outputs/airavata_mid_raw.csv
#     /home/hice1/gchiu6/scratch/literacy_eval_outputs/airavata_high_raw.csv
#     (same pattern for apollo)
#
# Logs:
#     /home/hice1/gchiu6/scratch/logs/literacy-full-airavata-<job_id>.out
#     /home/hice1/gchiu6/scratch/logs/literacy-full-apollo-<job_id>.out
#
# RESUME SUPPORT:
#     If this job gets killed (time limit, node failure, etc.), just re-run
#     the same sbatch command. The runner skips topics that already have
#     responses in the output CSV. No work is duplicated.
#
# TIME ESTIMATE (rough):
#     Airavata (~7B, bfloat16): ~30-60s per response on a modern GPU
#       → 30 topics × 3 levels = 90 responses → ~45-90 minutes
#     Apollo (~9B, bfloat16): ~40-80s per response
#       → 90 responses → ~60-120 minutes
#     Requested wall time: 4 hours — comfortable buffer for both models.

# ---------- SLURM resource requests ----------
#SBATCH --job-name=literacy-full
#SBATCH --output=/home/hice1/gchiu6/scratch/logs/literacy-full-%x-%j.out
#SBATCH --error=/home/hice1/gchiu6/scratch/logs/literacy-full-%x-%j.err
#SBATCH --time=04:00:00          # 4 hours — generous for 90 responses per model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
# Apollo is ~18 GB in bfloat16 — needs a GPU with at least 24 GB VRAM.
# Uncomment the line below to request a specific GPU type if needed:
# #SBATCH --gres=gpu:A100:1
# #SBATCH --partition=ice-gpu
# #SBATCH --account=your-account-here

# ---------- Validate input ----------
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "ERROR: usage: sbatch $0 <model>"
    echo "       model: airavata | apollo"
    exit 1
fi

MODEL="$1"
if [[ "$MODEL" != "airavata" && "$MODEL" != "apollo" && "$MODEL" != "nanda" && "$MODEL" != "mmed_llama" && "$MODEL" != "himed" ]]; then
    echo "ERROR: model must be 'airavata', 'apollo', 'nanda', 'mmed_llama', or 'himed', got '$MODEL'"
    exit 1
fi

SCRATCH="/home/hice1/gchiu6/scratch"
OUTPUT_DIR="$SCRATCH/literacy_eval_outputs"
LOGS_DIR="$SCRATCH/logs"

mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

echo "=== Literacy-Level Full Evaluation ==="
echo "Model:      $MODEL"
echo "Levels:     all (low, mid, high)"
echo "Topics:     all 30"
echo "Started:    $(date)"
echo "Node:       $(hostname)"
echo "Output dir: $OUTPUT_DIR"
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null \
    || echo "  (nvidia-smi not available)"
echo ""

# ---------- Environment setup ----------
cd "$SLURM_SUBMIT_DIR"

# Keep caches off the home filesystem to avoid quota errors
export HF_HOME="$SCRATCH/hf_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"

mkdir -p "$HF_HOME"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$PIP_CACHE_DIR"

echo "HF_HOME: $HF_HOME"
echo "TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
echo "PIP_CACHE_DIR: $PIP_CACHE_DIR"

# Activate the project virtual environment
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

echo "Python: $(which python)"
python --version

echo "PyTorch:"
python -c "import torch; print(torch.__version__)"

echo ""

# ---------- FULL RUN: curated topics — all 30 × all 3 levels ----------
# --level all runs low → mid → high sequentially.
# The model is loaded once and reused across all levels.
echo "=== Running $MODEL — curated topics (all levels) ==="

python -m literacy_eval.run_eval \
    --model "$MODEL" \
    --level all \
    --topic-source curated \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Running $MODEL — HiMed topics (all levels) ==="
# 50 MMLU-Pro-Health questions from HiMed-West_Bench, converted to
# open-ended explanation topics. Outputs land in {model}_{level}_himed_raw.csv

wantpython -m literacy_eval.run_eval \
    --model "$MODEL" \
    --level all \
    --topic-source himed \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Job complete at $(date) ==="
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR/${MODEL}_"*_raw.csv 2>/dev/null || echo "  (no output files found — check for errors above)"
echo ""
echo "Row counts:"
for f in "$OUTPUT_DIR/${MODEL}_"*_raw.csv; do
    [ -f "$f" ] && echo "  $(basename "$f"): $(( $(wc -l < "$f") - 1 )) rows"
done

