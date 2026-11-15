#!/bin/bash
# Prediction-comparison videos (real motion | TabPFN + retrieval | TabPFN recent window), seed 0.
#SBATCH --job-name=icwm_predvid
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn263,cn016
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
module load miniconda/3-4.11.0
source activate /scratch/$USER/conda-envs/tabpfn-mujoco
export TABPFN_MODEL_CACHE_DIR=/scratch/$USER/tabpfn_models PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 MUJOCO_GL=egl
cd /scratch/$USER/icwm/cheetah
SNAP=".code_${SLURM_JOB_ID}"; mkdir -p "$SNAP"; cp predict_video.py study.py pilot.py "$SNAP"/
echo "$(date) node $(hostname)"; nvidia-smi -L
( while sleep 600; do echo "gpu $(date +%H:%M) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; done ) &
mon=$!
python -W ignore "$SNAP/predict_video.py" --env hopper --seed 0 --steps 250 --fps 25 2>&1 | grep --line-buffered -v -i warn
python -W ignore "$SNAP/predict_video.py" --env walker2d --seed 0 --steps 250 --fps 25 2>&1 | grep --line-buffered -v -i warn
python -W ignore "$SNAP/predict_video.py" --env halfcheetah --seed 0 --steps 150 --fps 20 2>&1 | grep --line-buffered -v -i warn
kill $mon 2>/dev/null
echo "$(date) finished"
