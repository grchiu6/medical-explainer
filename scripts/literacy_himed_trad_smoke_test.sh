#!/bin/bash
#
# SLURM SMOKE TEST for literacy-level evaluation — HiMed-Trad track.
# Runs ONE HiMed-Trad question at ALL THREE levels for a SINGLE model, so
# you can verify the end-to-end pipeline (reference JSON → prompt → model →
# CSV) before committing to the full 60-question run.
#
# Submit once per model:
#     sbatch scripts/literacy_himed_trad_smoke_test.sh airavata
#     sbatch scripts/literacy_himed_trad_smoke_test.sh apollo
#     sbatch scripts/literacy_himed_trad_smoke_test.sh himed
#     sbatch scripts/literacy_himed_trad_smoke_test.sh nanda
#     sbatch scripts/literacy_himed_trad_smoke_test.sh mmed_llama
#
# After they finish, inspect the output CSVs:
#     cat /home/hice1/gchiu6/scratch/literacy_eval_outputs/apollo_low_himed_trad_raw.csv
#
# Logs:
#     /home/hice1/gchiu6/scratch/logs/lit-trad-smoke-<model>-<job_id>.out

# ---------- SLURM resource requests ----------
#SBATCH --job-name=lit-trad-smoke
#SBATCH --output=/home/hice1/gchiu6/scratch/logs/lit-trad-smoke-%x-%j.out
#SBATCH --error=/home/hice1/gchiu6/scratch/logs/lit-trad-smoke-%x-%j.err
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
    echo "       model: airavata | apollo | himed | nanda | mmed_llama"
    exit 1
fi

MODEL="$1"
if [[ "$MODEL" != "airavata" && "$MODEL" != "apollo" && "$MODEL" != "himed" && "$MODEL" != "nanda" && "$MODEL" != "mmed_llama" ]]; then
    echo "ERROR: model must be 'airavata', 'apollo', 'himed', 'nanda', or 'mmed_llama', got '$MODEL'"
    exit 1
fi

SCRATCH="/home/hice1/gchiu6/scratch"
OUTPUT_DIR="$SCRATCH/literacy_eval_outputs"
LOGS_DIR="$SCRATCH/logs"

mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"

echo "=== Literacy-Level HiMed-Trad Smoke Test ==="
echo "Model:      $MODEL"
echo "Levels:     all (low, mid, high)"
echo "Topics:     1 (first item from himed_trad_reference.json)"
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

# ---------- SMOKE TEST: 1 HiMed-Trad topic, all levels ----------
# --sample 1 picks the first item in the reference JSON (alphabetically by id).
# Running all 3 levels checks that register-switching works correctly.
echo "=== $MODEL smoke test (HiMed-Trad, 1 topic, all levels) ==="
python -m literacy_eval.run_eval \
    --model "$MODEL" \
    --level all \
    --topic-source himed_trad \
    --sample 1 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Output:"
for LEVEL in low mid high; do
    f="$OUTPUT_DIR/${MODEL}_${LEVEL}_himed_trad_raw.csv"
    if [ -f "$f" ]; then
        echo "--- $LEVEL ---"
        cat "$f"
    fi
done
echo ""
echo "=== Smoke test complete at $(date) ==="
