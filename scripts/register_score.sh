#!/bin/bash
#
# SLURM job: score the outside validation sets with ALREADY-TRAINED register classifiers (no retraining).
# Runs every trained model it finds (register_runs/<corpus>/<encoder>_seed42/best.pt) on every file in
# ~/scratch/register_eval/*.jsonl, and writes <file>_scores.jsonl next to each model's metrics.json.
#
# EVERYTHING WRITTEN BY THIS JOB GOES TO SCRATCH (/home/hice1/gchiu6/scratch).
#
# USAGE (on PACE, in the repo, after git pull and after uploading the files to ~/scratch/register_eval/):
#     sbatch scripts/register_score.sh
#
# TIME: ~2,000 short texts x 4 models: a few minutes. GPU: NVIDIA only, never mi210.

#SBATCH --job-name=register-score
#SBATCH --output=/home/hice1/gchiu6/scratch/logs/register-%x-%j.out
#SBATCH --error=/home/hice1/gchiu6/scratch/logs/register-%x-%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
# If l40s nodes are busy, change the line above to  gpu:a40:1  (or a100 / h100).

set -euo pipefail

SCRATCH="/home/hice1/gchiu6/scratch"
EVAL_DIR="$SCRATCH/register_eval"
RUNS_DIR="$SCRATCH/register_runs"

compgen -G "$EVAL_DIR/*.jsonl" > /dev/null || { echo "ERROR: no files in $EVAL_DIR/*.jsonl"; exit 1; }

export HF_HOME="$SCRATCH/hf_cache"
export TORCH_HOME="$SCRATCH/torch_cache"
export XDG_CACHE_HOME="$SCRATCH/xdg_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR" "$SCRATCH/logs"

echo "=== Scoring validation sets: $(ls "$EVAL_DIR"/*.jsonl | xargs -n1 basename | tr '\n' ' ') ==="
echo "Started: $(date)   Node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not available)"

cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

found=0
for ckpt in "$RUNS_DIR"/*/*_seed42/best.pt; do
    [ -f "$ckpt" ] || continue
    run_dir="$(dirname "$ckpt")"
    encoder="$(basename "$run_dir" | sed 's/_seed42$//')"
    echo "--- $run_dir ($encoder)"
    python -m literacy_eval.register.score --encoder "$encoder" --checkpoint "$ckpt" \
        --inputs "$EVAL_DIR"/*.jsonl --out-dir "$run_dir"
    found=$((found + 1))
done
[ "$found" -gt 0 ] || { echo "ERROR: no best.pt found under $RUNS_DIR (were they deleted?)"; exit 1; }

echo "=== Done at $(date): scored with $found models ==="
