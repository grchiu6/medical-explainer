#!/bin/bash
#
# SLURM job: fine-tune the Hindi register classifier (5th / 10th / professional) and score the LLM outputs.
#
# BEFORE SUBMITTING (once):
#   1. On your Mac, build the corpus and the LLM windows:
#        python -m literacy_eval.register.build_corpus
#        python -m literacy_eval.register.prepare_outputs
#   2. Copy the corpus to PACE scratch (from the repo root on your Mac):
#        scp data/register_corpus/{train,val,test,llm_windows}.jsonl \
#            gchiu6@login-ice.pace.gatech.edu:/home/hice1/gchiu6/scratch/register_corpus/
#      (make the folder first: ssh in and run  mkdir -p ~/scratch/register_corpus)
#   3. git pull on PACE so literacy_eval/register/train.py is there.
#
# SUBMIT once per encoder (they run in parallel on separate GPUs):
#     sbatch scripts/register_train.sh muril
#     sbatch scripts/register_train.sh indicbert
#
# OUTPUT (per encoder): /home/hice1/gchiu6/scratch/register_runs/<encoder>_seed42/
#     metrics.json              val history + test accuracy / macro-F1 / confusion matrix
#     test_scores.jsonl         per-passage scores on the held-out test split
#     llm_window_scores.jsonl   per-window scores for the usable LLM outputs   <- copy back to the Mac
#     best.pt                   model weights (large; leave on PACE)
#
# TIME: a few thousand ~100-word passages x 5 epochs on one L40S/A40 is ~5-15 minutes. 1 hour requested.
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

set -euo pipefail

if [ "$#" -ne 1 ] || [[ "$1" != "muril" && "$1" != "indicbert" ]]; then
    echo "ERROR: usage: sbatch $0 <muril|indicbert>"
    exit 1
fi
ENCODER="$1"

SCRATCH="/home/hice1/gchiu6/scratch"
CORPUS_DIR="$SCRATCH/register_corpus"
OUT_DIR="$SCRATCH/register_runs"
mkdir -p "$OUT_DIR" "$SCRATCH/logs"

for f in train val test llm_windows; do
    [ -f "$CORPUS_DIR/$f.jsonl" ] || { echo "ERROR: missing $CORPUS_DIR/$f.jsonl (see BEFORE SUBMITTING)"; exit 1; }
done

echo "=== Register classifier: $ENCODER ==="
echo "Started: $(date)   Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "(nvidia-smi not available)"

cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$SCRATCH/hf_cache"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

EXTRA=()
[ -f "$CORPUS_DIR/readmepp_hi.jsonl" ] && EXTRA=(--extra-score "$CORPUS_DIR/readmepp_hi.jsonl")

python -m literacy_eval.register.train \
    --encoder "$ENCODER" \
    --corpus-dir "$CORPUS_DIR" \
    --out-dir "$OUT_DIR" \
    "${EXTRA[@]}"

echo "=== Done at $(date) ==="
ls -lh "$OUT_DIR/${ENCODER}_seed42/"
