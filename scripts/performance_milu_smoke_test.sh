#!/bin/bash
#
# SLURM SMOKE TEST for the MILU Hindi Health-and-Medicine MCQ performance track.
# Runs only 2 items (--limit 2) for ONE model + condition, to verify the
# end-to-end path (HF gate/token -> model load -> prompt -> generation -> CSV)
# before committing to a full run.
#
# Usage:
#     sbatch scripts/performance_milu_smoke_test.sh <model> <condition> [extra run.py flags...]
#     sbatch scripts/performance_milu_smoke_test.sh himed c0_baseline
#     sbatch scripts/performance_milu_smoke_test.sh nanda c2_cot --use-cot --max-new-tokens 768
#
# HUGGING FACE AUTH (MILU is gated): before the first run, do ONE of
#     HF_HOME=/home/hice1/gchiu6/scratch/hf_cache huggingface-cli login   # once, on PACE
#     export HF_TOKEN=...          # in your shell before sbatch
# The script only checks that credentials exist; it never prints the token.
#
# Output CSV: /home/hice1/gchiu6/scratch/performance_eval_outputs/milu_hi_med_smoke/<model>_<condition>_raw.csv
# Logs:       /home/hice1/gchiu6/scratch/logs/perf-milu-smoke-*.out / .err

# ---------- SLURM resource requests ----------
#SBATCH --job-name=perf-milu-smoke
#SBATCH --output=/home/hice1/gchiu6/scratch/logs/perf-milu-smoke-%x-%j.out
#SBATCH --error=/home/hice1/gchiu6/scratch/logs/perf-milu-smoke-%x-%j.err
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

# ---------- Validate input (before any cluster-specific work) ----------
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "ERROR: usage: sbatch $0 <model> <condition> [extra run.py flags...]"
    echo "       model:     himed | nanda"
    echo "       condition: label used in the output filename, e.g. c0_baseline"
    exit 1
fi

MODEL="$1"
CONDITION="$2"
if [[ "$MODEL" != "himed" && "$MODEL" != "nanda" ]]; then
    echo "ERROR: model must be 'himed' or 'nanda', got '$MODEL'"
    exit 1
fi
if [[ "$CONDITION" == --* ]]; then
    echo "ERROR: condition label expected as 2nd argument, got flag '$CONDITION'"
    exit 1
fi
shift 2   # remaining args are forwarded verbatim to performance_eval.run

SCRATCH="/home/hice1/gchiu6/scratch"
OUTPUT_DIR="$SCRATCH/performance_eval_outputs/milu_hi_med_smoke"
LOGS_DIR="$SCRATCH/logs"

mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"
OUT_CSV="$OUTPUT_DIR/${MODEL}_${CONDITION}_raw.csv"

echo "=== MILU Hindi Med MCQ Smoke Test ==="
echo "Model:      $MODEL"
echo "Condition:  $CONDITION"
echo "Extra args: $*"
echo "Items:      2 (--limit 2)"
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

# MILU is gated: fail fast if no HF credentials. Existence check only; the
# token is never read or printed.
# Only HF_TOKEN or $HF_HOME/token count: HF_HOME is overridden above, so
# ~/.cache/huggingface/token would NOT be read by the job.
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HF_HOME/token" ]; then
    echo "ERROR: no Hugging Face credentials found (MILU is a gated dataset)."
    echo "       Run once on PACE: HF_HOME=/home/hice1/gchiu6/scratch/hf_cache huggingface-cli login"
    echo "       (or export HF_TOKEN=...), then resubmit."
    exit 1
fi

source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
echo "Python: $(which python) $(python --version)"
python -c "import torch; print('PyTorch:', torch.__version__)"
echo ""

# ---------- SMOKE TEST ----------
# Always regenerate: delete any earlier smoke CSV so resume cannot skip the model.
rm -f "$OUT_CSV"
python -m performance_eval.run \
    --model "$MODEL" \
    --condition "$CONDITION" \
    --limit 2 \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo ""
echo "Output ($OUT_CSV):"
cat "$OUT_CSV"
echo ""
echo "=== Smoke test complete at $(date) ==="
