#!/bin/bash
# Pool-size scaling of retrieval vs full-pool context (one process per GPU: clean time / memory numbers).
# Usage: sbatch scale.sbatch <n_train>
#SBATCH --job-name=icwm_scale
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn263,cn016
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
module load miniconda/3-4.11.0
source activate /scratch/$USER/conda-envs/tabpfn
export TABPFN_MODEL_CACHE_DIR=/scratch/$USER/tabpfn_models PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
cd /scratch/$USER/icwm/scaling
SNAP=".code_${SLURM_JOB_ID}"; mkdir -p "$SNAP"; cp scaling.py study.py pilot.py "$SNAP"/
echo "$(date) node $(hostname) n_train $1"; nvidia-smi -L
( while sleep 600; do echo "gpu $(date +%H:%M) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; done ) &
mon=$!
python -W ignore "$SNAP/scaling.py" --system hopper --n_train $1 --seed_ids 0 1 2 --horizon 50 --out scale_hopper_nt$1.json 2>&1 | grep --line-buffered -v -i warn
kill $mon 2>/dev/null
echo "$(date) finished"
