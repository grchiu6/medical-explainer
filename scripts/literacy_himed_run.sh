#!/bin/bash
#
# SLURM FULL RUN script for literacy-level evaluation — HiMed topics ONLY.
# Runs all 50 HiMed-derived topics x 3 levels for ONE model.
# Curated topics are NOT run here (use scripts/literacy_full_run.sh for that).
#
# Submit once per model — jobs run in parallel across separate GPUs:
#     sbatch scripts/literacy_himed_run.sh airavata
#     sbatch scripts/literacy_himed_run.sh apollo
#     sbatch scripts/literacy_himed_run.sh nanda
#     sbatch scripts/literacy_himed_run.sh mmed_llama
#     sbatch scripts/literacy_himed_run.sh himed
#
# Output CSVs (one per level):
#     /home/hice1/gchiu6/scratch/literacy_eval_outputs/{model}_low_himed_raw.csv
#     /home/hice1/gchiu6/scratch/literacy_eval_outputs/{model}_mid_himed_raw.csv
#     /home/hice1/gchiu6/scratch/literacy_eval_outputs/{model}_high_himed_raw.csv
#
# Logs:
#     /home/hice1/gchiu6/scratch/logs/literacy-himed-<model>-<job_id>.out
#
# RESUME SUPPORT:
#     If this job gets killed, just re-run the same sbatch command. The
#     runner skips topics that already have responses in the output CSV.
#
# TIME ESTIMATE (rough, 50 topics x 3 levels = 150 responses per model):
#     Airavata/Apollo (~7-9B): ~30-80s per response -> ~75-200 min
#     Nanda/HiMed/MMed-Llama (~8-10B): similar range
#     Requested wall time: 3 hours — comfortable buffer.

# ---------- SLURM resource requests ----------
#SBATCH --job-name=literacy-himed
#SBATCH --output=/home/hice1/gchiu6/scratch/logs/literacy-himed-%x-%j.out
#SBATCH --error=/home/hice1/gchiu6/scratch/logs/literacy-himed-%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
# Nanda is ~20GB in bfloat16 — needs a GPU with at least 24GB VRAM.
# Uncomment the line below to request a specific GPU type if needed:
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

echo "=== Literacy-Level HiMed-Only Evaluation ==="
echo "Model:      $MODEL"
echo "Levels:     all (low, mid, high)"
echo "Topics:     all 50 (HiMed-derived, curated topics skipped)"
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

echo "HF_HOME: $HF_HOME"

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

echo "Python: $(which python)"
python --version
echo "PyTorch:"
python -c "import torch; print(torch.__version__)"
echo ""

# ---------- HiMed topics only — all levels, all 50 questions ----------
echo "=== Running $MODEL — HiMed topics (all levels) ==="

python -m literacy_eval.run_eval \
    --model "$MODEL" \
    --level all \
    --topic-source himed \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Job complete at $(date) ==="
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR/${MODEL}_"*_himed_raw.csv 2>/dev/null || echo "  (no output files found — check for errors above)"
echo ""
echo "Row counts:"
for f in "$OUTPUT_DIR/${MODEL}_"*_himed_raw.csv; do
    [ -f "$f" ] && echo "  $(basename "$f"): $(( $(wc -l < "$f") - 1 )) rows"
done
