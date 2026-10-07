#!/bin/bash
#
# SLURM SMOKE TEST script for literacy-level evaluation.
# Runs ONE topic at ONE level per model so you can verify the pipeline
# works end-to-end on the cluster before committing to the full run.
#
# Usage:
#     sbatch scripts/literacy_smoke_test.sh
#
# After it finishes, inspect the output CSVs:
#     cat /home/hice1/gchiu6/scratch/literacy_eval_outputs/airavata_low_raw.csv
#     cat /home/hice1/gchiu6/scratch/literacy_eval_outputs/apollo_low_raw.csv
#
# Logs:
#     /home/hice1/gchiu6/scratch/logs/literacy-smoke-<job_id>.out
#     /home/hice1/gchiu6/scratch/logs/literacy-smoke-<job_id>.err

# ---------- SLURM resource requests ----------
#SBATCH --job-name=literacy-smoke
#SBATCH --output=/home/hice1/gchiu6/scratch/logs/literacy-smoke-%j.out
#SBATCH --error=/home/hice1/gchiu6/scratch/logs/literacy-smoke-%j.err
#SBATCH --time=01:00:00          # 1 hour — plenty for loading 2 models + 1 response each
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
# Uncomment + edit the lines below if your cluster requires a partition/account:
# #SBATCH --partition=ice-gpu
# #SBATCH --account=your-account-here

set -euo pipefail

SCRATCH="/home/hice1/gchiu6/scratch"
OUTPUT_DIR="$SCRATCH/literacy_eval_outputs"
LOGS_DIR="$SCRATCH/logs"

mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

echo "=== Literacy-Level Smoke Test ==="
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

# Point HuggingFace cache at scratch — model weights are ~14-18 GB each
# and will not fit in $HOME quota. This is the most important env var.
export HF_HOME="$SCRATCH/hf_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"

mkdir -p "$HF_HOME"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$PIP_CACHE_DIR"
echo "HF_HOME (model weight cache): $HF_HOME"

# Load Python (adjust if your cluster uses a different module name)
# Activate the project virtualenv
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
echo "Python: $(which python) $(python --version)"
python -c "import torch; print('PyTorch:', torch.__version__)"
echo ""

# ---------- SMOKE TEST: Airavata — 1 topic, low level ----------
echo "=== [1/2] Airavata smoke test (topic: diabetes, level: low) ==="
python -m literacy_eval.run_eval \
    --model airavata \
    --level low \
    --topic diabetes \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Airavata output:"
cat "$OUTPUT_DIR/airavata_low_raw.csv"
echo ""

# ---------- SMOKE TEST: Apollo — 1 topic, low level ----------
echo "=== [2/2] Apollo smoke test (topic: diabetes, level: low) ==="
python -m literacy_eval.run_eval \
    --model apollo \
    --level low \
    --topic diabetes \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Apollo output:"
cat "$OUTPUT_DIR/apollo_low_raw.csv"
echo ""

echo "=== Smoke test complete at $(date) ==="
echo "Outputs saved to: $OUTPUT_DIR"
