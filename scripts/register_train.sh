#!/bin/bash
#
# SLURM job: fine-tune the Hindi register classifier (5th / 10th / professional) and score the LLM outputs.
#
# EVERYTHING WRITTEN BY THIS JOB GOES TO SCRATCH (/home/hice1/gchiu6/scratch), the only place with space:
# model downloads, torch/triton/pip caches, temp files, logs, checkpoints and results. The repo itself (code +
# .venv) stays where you cloned it.
#
# USAGE (one job per encoder x corpus; 4 jobs in total, they run in parallel on separate GPUs):
#     sbatch --job-name=reg-muril-c10   scripts/register_train.sh muril     register_corpus
#     sbatch --job-name=reg-indic-c10   scripts/register_train.sh indicbert register_corpus
#     sbatch --job-name=reg-muril-c910  scripts/register_train.sh muril     register_corpus_9_10
#     sbatch --job-name=reg-indic-c910  scripts/register_train.sh indicbert register_corpus_9_10
#   register_corpus      = 10th grade is NCERT Class 10 biology only
#   register_corpus_9_10 = 10th grade is Class 10 + Class 9 biology (more data per class)
#
# BEFORE SUBMITTING (once): see "Running the training job" in literacy_eval/docs/2026-10-12-register-classifier-plan.md
#   - $SCRATCH/register_corpus/       train.jsonl val.jsonl test.jsonl llm_windows.jsonl
#   - $SCRATCH/register_corpus_9_10/  train.jsonl val.jsonl test.jsonl llm_windows.jsonl
#   - optional: $SCRATCH/register_eval/*.jsonl  (extra files to score, e.g. readmepp_hi.jsonl, with a "text" field)
#
# OUTPUT: /home/hice1/gchiu6/scratch/register_runs/<corpus>/<encoder>_seed42/
#     metrics.json              val history + test accuracy / macro-F1 / confusion matrix
#     test_scores.jsonl         per-passage scores on the held-out test split
#     llm_window_scores.jsonl   per-window scores for the usable LLM outputs   <- copy back to the Mac
#     *_scores.jsonl            one per extra file in register_eval/
#     best.pt                   model weights (~1 GB; leave on scratch, delete when done)
#
# TIME: a few hundred ~100-word passages x 5 epochs on one GPU is a few minutes (1 hour requested).
# GPU: NVIDIA only (l40s, a40, a100, h100, v100, rtx_6000). NOT mi210: AMD, the CUDA build of PyTorch won't run.

#SBATCH --job-name=register-train
#SBATCH --output=/home/hice1/gchiu6/scratch/logs/register-%x-%j.out
#SBATCH --error=/home/hice1/gchiu6/scratch/logs/register-%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
# If l40s nodes are busy, change the line above to  gpu:a40:1  (or a100 / h100).
# NOTE: the log directory in --output above must already exist when you submit:  mkdir -p ~/scratch/logs

set -euo pipefail

if [ "$#" -ne 2 ] || [[ "$1" != "muril" && "$1" != "indicbert" ]] \
    || [[ "$2" != "register_corpus" && "$2" != "register_corpus_9_10" ]]; then
    echo "ERROR: usage: sbatch $0 <muril|indicbert> <register_corpus|register_corpus_9_10>"
    exit 1
fi
ENCODER="$1"
CORPUS_NAME="$2"

SCRATCH="/home/hice1/gchiu6/scratch"
CORPUS_DIR="$SCRATCH/$CORPUS_NAME"
EVAL_DIR="$SCRATCH/register_eval"
OUT_DIR="$SCRATCH/register_runs/$CORPUS_NAME"
mkdir -p "$OUT_DIR" "$SCRATCH/logs"

for f in train val test llm_windows; do
    [ -f "$CORPUS_DIR/$f.jsonl" ] || { echo "ERROR: missing $CORPUS_DIR/$f.jsonl (see BEFORE SUBMITTING)"; exit 1; }
done

# ---------- Send every cache and temp file to scratch ----------
export HF_HOME="$SCRATCH/hf_cache"            # Hugging Face model downloads
export TORCH_HOME="$SCRATCH/torch_cache"
export XDG_CACHE_HOME="$SCRATCH/xdg_cache"    # anything that falls back to ~/.cache
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

echo "=== Register classifier: $ENCODER on $CORPUS_NAME ==="
echo "Started: $(date)   Node: $(hostname)"
echo "Output:  $OUT_DIR"
echo "Scratch free space:"; df -h "$SCRATCH" | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "(nvidia-smi not available)"

cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

EXTRA=()
if compgen -G "$EVAL_DIR/*.jsonl" > /dev/null; then
    EXTRA=(--extra-score "$EVAL_DIR"/*.jsonl)
fi

python -m literacy_eval.register.train \
    --encoder "$ENCODER" \
    --corpus-dir "$CORPUS_DIR" \
    --out-dir "$OUT_DIR" \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo "=== Done at $(date) ==="
ls -lh "$OUT_DIR/${ENCODER}_seed42/"
