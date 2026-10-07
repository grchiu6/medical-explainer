#!/bin/bash
#
# SLURM SMOKE TEST script for literacy-level evaluation — HiMed topics ONLY.
# Runs ONE HiMed question at the 'low' level for a SINGLE model, so you can
# verify the HiMed pipeline works end-to-end before committing to the full
# 50-question run.
#
# Submit once per model — jobs run IN PARALLEL across separate GPUs, which
# is what makes this fast for testing all 5 models (vs. one job looping
# through models sequentially and loading each one after another):
#     sbatch scripts/literacy_himed_smoke_test.sh airavata
#     sbatch scripts/literacy_himed_smoke_test.sh apollo
#     sbatch scripts/literacy_himed_smoke_test.sh nanda
#     sbatch scripts/literacy_himed_smoke_test.sh mmed_llama
#     sbatch scripts/literacy_himed_smoke_test.sh himed
#
# After they finish, inspect the output CSVs:
#     cat /home/hice1/gchiu6/scratch/literacy_eval_outputs/airavata_low_himed_raw.csv
#
# Logs:
#     /home/hice1/gchiu6/scratch/logs/literacy-himed-smoke-<model>-<job_id>.out

# ---------- SLURM resource requests ----------
# Kept small/short since this is a single-model, single-question check.
#SBATCH --job-name=literacy-himed-smoke
#SBATCH --output=/home/hice1/gchiu6/scratch/logs/literacy-himed-smoke-%x-%j.out
#SBATCH --error=/home/hice1/gchiu6/scratch/logs/literacy-himed-smoke-%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
# Nanda is ~20GB in bfloat16 — needs a GPU with at least 24GB VRAM.
# #SBATCH --gres=gpu:A100:1
# #SBATCH --partition=ice-gpu
# #SBATCH --account=your-account-here

# ---------- Validate input ----------
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "ERROR: usage: sbatch $0 <model>"
    echo "       model: airavata | apollo | nanda | mmed_llama | himed"
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

echo "=== Literacy-Level HiMed Smoke Test ==="
echo "Model:      $MODEL"
echo "Level:      low"
echo "Topics:     1 (sampled from HiMed)"
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

export HF_HOME="$SCRATCH/hf_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"

mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR"

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
echo "Python: $(which python) $(python --version)"
python -c "import torch; print('PyTorch:', torch.__version__)"
echo ""

# ---------- SMOKE TEST: 1 HiMed topic, low level ----------
echo "=== $MODEL smoke test (HiMed, 1 topic, low level) ==="
python -m literacy_eval.run_eval \
    --model "$MODEL" \
    --level low \
    --topic-source himed \
    --sample 1 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Output:"
cat "$OUTPUT_DIR/${MODEL}_low_himed_raw.csv"
echo ""
echo "=== Smoke test complete at $(date) ==="
