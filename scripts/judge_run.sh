#!/bin/bash
#
# SLURM job: run the M-Prometheus-14B judge (vLLM, greedy) on ~/scratch/judge/jobs.jsonl.
# Appends to ~/scratch/judge/results.jsonl and resumes where it stopped, so after a timeout just resubmit.
#
# EVERYTHING WRITTEN BY THIS JOB GOES TO SCRATCH (/home/hice1/gchiu6/scratch), including the vLLM venv
# and the ~30 GB model download.
#
# USAGE:
#   (Mac)  python -m literacy_eval.judge.jobs --out data/judge/jobs.jsonl
#          ssh gchiu6@login-ice.pace.gatech.edu 'mkdir -p ~/scratch/judge'     # GT VPN if off campus
#          scp data/judge/jobs.jsonl gchiu6@login-ice.pace.gatech.edu:~/scratch/judge/jobs.jsonl
#   (PACE, in the repo, after git pull)
#          sbatch scripts/judge_run.sh                       # all jobs
#          sbatch scripts/judge_run.sh --match controlled:   # extra args go to literacy_eval.judge.run
#   (Mac)  scp gchiu6@login-ice.pace.gatech.edu:~/scratch/judge/results.jsonl data/judge/results.jsonl
#          python -m literacy_eval.judge.collect
#
# TIME: ~3,500 prompts of 1-6k characters on one L40S: roughly 1-3 h. GPU: NVIDIA only, never mi210.

#SBATCH --job-name=judge
#SBATCH --output=/home/hice1/gchiu6/scratch/logs/judge-%x-%j.out
#SBATCH --error=/home/hice1/gchiu6/scratch/logs/judge-%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
# If l40s nodes are busy, change the line above to  gpu:a100:1  (or h100 / a40).

set -euo pipefail

SCRATCH="/home/hice1/gchiu6/scratch"
JUDGE_DIR="$SCRATCH/judge"
VENV="$SCRATCH/venv-judge"

[ -f "$JUDGE_DIR/jobs.jsonl" ] || { echo "ERROR: $JUDGE_DIR/jobs.jsonl not found (scp it from the Mac first)"; exit 1; }

export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$SCRATCH/torch_cache"
export XDG_CACHE_HOME="$SCRATCH/xdg_cache"
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"
export VLLM_CACHE_ROOT="$SCRATCH/vllm_cache"
export TMPDIR="$SCRATCH/tmp"
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$PIP_CACHE_DIR" \
         "$VLLM_CACHE_ROOT" "$TMPDIR" "$SCRATCH/logs" "$JUDGE_DIR"

echo "=== Judge run: $(wc -l < "$JUDGE_DIR/jobs.jsonl") jobs, args: $* ==="
echo "Started: $(date)   Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not available)"

# vLLM needs Python >= 3.10, but the system python3 on ICE nodes is 3.9. Load a newer Python module:
#     sbatch --export=ALL,PY_MODULE=python/3.12.5 scripts/judge_run.sh     (names: `module avail python`)
# The module is loaded in the job too, because a venv's python links to the module's shared libraries.
# TIP: build the venv once on a login node (same module load, then the venv/pip lines below), so a broken
# install shows up before waiting in the GPU queue. After the pilot, pin the printed vllm version
# (VLLM_SPEC="vllm==X.Y.Z") so the full run uses the same build.
PY_MODULE="${PY_MODULE:-python/3.12.5}"
VLLM_SPEC="${VLLM_SPEC:-vllm==0.32.0}"   # installed and imported on ICE L40S, 2026-10-27
if [ -n "$PY_MODULE" ]; then
    module load "$PY_MODULE"
fi
if ! "$VENV/bin/python" -c "import vllm" 2>/dev/null; then
    python3 -c 'import sys; assert sys.version_info >= (3, 10)' 2>/dev/null || {
        echo "ERROR: python3 is $(python3 --version 2>&1); vLLM needs >= 3.10."
        echo "       Pass a newer module: sbatch --export=ALL,PY_MODULE=<name from 'module avail python'> scripts/judge_run.sh"
        exit 1; }
    echo "--- (re)building $VENV with $(python3 --version): no importable vllm (first run, or an earlier install failed)"
    rm -rf "$VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install "$VLLM_SPEC" pandas huggingface_hub
fi
"$VENV/bin/python" --version
# (not `pip show | head`: head closes the pipe early and, under pipefail, that kills the job)
"$VENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)"

cd "$SLURM_SUBMIT_DIR"
"$VENV/bin/python" -m literacy_eval.judge.run \
    --jobs "$JUDGE_DIR/jobs.jsonl" --out "$JUDGE_DIR/results.jsonl" "$@"

echo "=== Done at $(date): $(wc -l < "$JUDGE_DIR/results.jsonl") result rows in $JUDGE_DIR/results.jsonl ==="
