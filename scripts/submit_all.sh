#!/bin/bash
#
# Submit ALL 6 local-model evaluation jobs at once.
#
# Usage (from project root):
#     bash scripts/submit_all.sh              # full runs (1052+ questions each)
#     bash scripts/submit_all.sh 5            # smoke test with 5 questions per job
#
# All 6 jobs queue independently — SLURM runs them in parallel as GPUs free up.
# Total wall-clock: ~2-4 hours if cluster is uncongested, longer if busy.

set -euo pipefail

SAMPLE_SIZE="${1:-}"   # optional first arg

MODELS=("airavata" "apollo")
DATASETS=("indicqa" "himed_trad" "himed_west")

echo "Submitting 6 jobs (2 models x 3 datasets)..."
echo "Sample size: ${SAMPLE_SIZE:-<all questions>}"
echo

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        if [ -n "$SAMPLE_SIZE" ]; then
            sbatch scripts/run_local_model.sh "$model" "$dataset" "$SAMPLE_SIZE"
        else
            sbatch scripts/run_local_model.sh "$model" "$dataset"
        fi
    done
done

echo
echo "All jobs submitted. Watch them with:"
echo "    squeue -u \$USER"
