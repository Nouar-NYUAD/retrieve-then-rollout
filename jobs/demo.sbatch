#!/bin/bash
# HalfCheetah demo: runs lines <from>..<to> of a task file SEQUENTIALLY (each = cheetah_mpc.py args) on one GPU.
# Usage: sbatch demo.sbatch <taskfile> <from> <to>
#SBATCH --job-name=icwm_demo
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn263,cn016
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
module load miniconda/3-4.11.0
source activate /scratch/$USER/conda-envs/tabpfn-mujoco
export TABPFN_MODEL_CACHE_DIR=/scratch/$USER/tabpfn_models PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 MUJOCO_GL=egl
cd /scratch/$USER/icwm/cheetah
SNAP=".code_${SLURM_JOB_ID}"; mkdir -p "$SNAP"; cp cheetah_mpc.py pilot.py "$SNAP"/
echo "$(date) node $(hostname): $1 lines $2-$3"; nvidia-smi -L
( while sleep 600; do echo "gpu $(date +%H:%M) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; done ) &
mon=$!
sed -n "$2,$3p" "$1" | while read -r line; do
  [ -z "$line" ] && continue
  echo "$(date) START $line"
  python -W ignore "$SNAP/cheetah_mpc.py" $line 2>&1 | grep --line-buffered -v -i warn
done
kill $mon 2>/dev/null
echo "$(date) finished"
