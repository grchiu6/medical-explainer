#!/bin/bash
#
# SLURM submission script for running a LOCAL model (Airavata or Apollo)
# on the GPU cluster against a specific dataset.
#
# Usage (from the project root):
#     sbatch scripts/run_local_model.sh <model> <dataset> [sample_size]
#
# Examples:
#     sbatch scripts/run_local_model.sh airavata himed_trad
#     sbatch scripts/run_local_model.sh apollo indicqa 50
#
# Output goes to logs/<model>-<dataset>-<job_id>.{out,err}
#
# What this script does:
#   1. Asks SLURM for 1 GPU + 4 CPU cores + 48 GB RAM for up to 6 hours
#   2. Activates the project's virtual environment
#   3. Runs evaluate.py with the given model + dataset
#
# Scoring is intentionally NOT in this script — run that separately
# on the laptop after the raw CSV is downloaded:
#     python score_mcq.py --model airavata --dataset himed_trad   (for MCQ)
#     python score.py --model airavata                             (for IndicQA)

# ---------- SLURM resource requests ----------
# These #SBATCH lines must come BEFORE any executable code. SLURM reads
# them as comments but interprets them as resource requests.

#SBATCH --job-name=hindi-eval               # how this job appears in `squeue`
#SBATCH --output=logs/%x-%j.out             # stdout: %x = job name, %j = job ID
#SBATCH --error=logs/%x-%j.err              # stderr: same naming
#SBATCH --time=06:00:00                     # max wall-clock time (HH:MM:SS)
#SBATCH --nodes=1                           # always 1 node (we're not distributed)
#SBATCH --ntasks=1                          # one process
#SBATCH --cpus-per-task=4                   # CPU cores for data loading / tokenization
#SBATCH --mem=48G                           # RAM (model weights + activations)
#SBATCH --gres=gpu:1                        # generic resource: 1 GPU of any type
# -- If you need a SPECIFIC GPU type, replace the line above with e.g.:
# #SBATCH --gres=gpu:V100:1
# #SBATCH --gres=gpu:A100:1
# -- ICE PACE may also require a partition. Uncomment + edit if so:
# #SBATCH --partition=ice-gpu
# #SBATCH --account=your-account-here

# ---------- Validate inputs ----------
set -euo pipefail   # -e: stop on error  -u: error on unset var  -o pipefail: catch pipe errors

if [ "$#" -lt 2 ]; then
    echo "ERROR: usage: sbatch $0 <model> <dataset> [sample_size]"
    echo "       model:   airavata | apollo | gemini"
    echo "       dataset: indicqa | himed_trad | himed_west"
    exit 1
fi

MODEL="$1"
DATASET="$2"
SAMPLE_SIZE="${3:-}"   # optional 3rd arg, defaults to empty

echo "=== Job started at $(date) ==="
echo "Model:       $MODEL"
echo "Dataset:     $DATASET"
echo "Sample size: ${SAMPLE_SIZE:-<all>}"
echo "Node:        $(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "  (nvidia-smi not available)"

# ---------- Environment setup ----------
# Move to the project root (the directory you submitted from).
cd "$SLURM_SUBMIT_DIR"

# Point HuggingFace at scratch space so model downloads don't blow $HOME quota.
# Airavata + Apollo together are ~32 GB. $HOME usually has a tight quota.
# Scratch is large and quota-free.
export HF_HOME="/home/hice1/gchiu6/scratch/hf_cache"
mkdir -p "$HF_HOME"
echo "HF_HOME (model cache): $HF_HOME"

# Load Python module (cluster-specific — replace with your cluster's command).
# Common alternatives:
#   module load anaconda3/2024.02
#   module load python/3.12
module load anaconda3 || echo "WARNING: 'module load anaconda3' failed — continuing"

# Activate the project virtualenv.
source venv/bin/activate

# ---------- Run the evaluation ----------
echo "=== Running evaluate.py ==="

if [ -n "$SAMPLE_SIZE" ]; then
    python evaluate.py --model "$MODEL" --dataset "$DATASET" --sample "$SAMPLE_SIZE"
else
    python evaluate.py --model "$MODEL" --dataset "$DATASET"
fi

echo "=== Job finished at $(date) ==="
