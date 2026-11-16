#!/bin/bash
#
# SLURM FULL RUN for the MILU Hindi Health-and-Medicine MCQ performance track.
# Runs all test items for ONE model + condition (no --limit).
#
# Usage:
#     sbatch scripts/performance_milu_run.sh <model> <condition> [extra run.py flags...]
#
# Ladder example (copy-paste; one job per model per condition):
#     # C0 baseline
#     sbatch scripts/performance_milu_run.sh himed c0_baseline
#     sbatch scripts/performance_milu_run.sh nanda c0_baseline
#     # C2 chain-of-thought. 768 is a STARTING POINT: check it against the
#     # output lengths seen in the smoke test / C0 before relying on it.
#     sbatch scripts/performance_milu_run.sh himed c2_cot --use-cot --max-new-tokens 768
#     sbatch scripts/performance_milu_run.sh nanda c2_cot --use-cot --max-new-tokens 768
#
# Then compare conditions:
#     python -m performance_eval.analyze --model himed --ladder c0_baseline:first_letter c2_cot:after_marker --raw-dir /home/hice1/gchiu6/scratch/performance_eval_outputs/milu_hi_med
#
# HUGGING FACE AUTH (MILU is gated): before the first run, do ONE of
#     HF_HOME=/home/hice1/gchiu6/scratch/hf_cache huggingface-cli login   # once, on PACE
#     export HF_TOKEN=...          # in your shell before sbatch
# The script only checks that credentials exist; it never prints the token.
#
# Output CSV: /home/hice1/gchiu6/scratch/performance_eval_outputs/milu_hi_med/<model>_<condition>_raw.csv
# Logs:       /home/hice1/gchiu6/scratch/logs/perf-milu-*.out / .err
#
# RESUME SUPPORT: if killed, re-run the same sbatch command; the runner skips
# item_ids already present in the output CSV.
# WARNING: Resume skips item_ids already in the CSV. If you change ANY flag for a condition, use a new --condition label or delete that CSV first.
#
# Requested wall time: 4 hours; adjust after the smoke test gives real timing.

# ---------- SLURM resource requests ----------
#SBATCH --job-name=perf-milu
#SBATCH --output=/home/hice1/gchiu6/scratch/logs/perf-milu-%x-%j.out
#SBATCH --error=/home/hice1/gchiu6/scratch/logs/perf-milu-%x-%j.err
#SBATCH --time=04:00:00
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
OUTPUT_DIR="$SCRATCH/performance_eval_outputs/milu_hi_med"
LOGS_DIR="$SCRATCH/logs"

mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"
OUT_CSV="$OUTPUT_DIR/${MODEL}_${CONDITION}_raw.csv"

echo "=== MILU Hindi Med MCQ Full Run ==="
echo "Model:      $MODEL"
echo "Condition:  $CONDITION"
echo "Extra args: $*"
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

# ---------- FULL RUN ----------
python -m performance_eval.run \
    --model "$MODEL" \
    --condition "$CONDITION" \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo ""
echo "=== Job complete at $(date) ==="
echo "Output: $OUT_CSV"
# csv-aware count (responses may contain newlines, so wc -l would overcount)
python - "$OUT_CSV" <<'PY'
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    print("Rows:", sum(1 for _ in csv.DictReader(f)))
PY
