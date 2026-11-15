#!/bin/bash
# Retrieval-vs-recent context pilot for one system.
# Usage (from /scratch/$USER/icwm/pilot): sbatch pilot.sbatch <lorenz|pendulum> <seed>
#SBATCH --job-name=icwm_pilot
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
module load miniconda/3-4.11.0
source activate /scratch/$USER/conda-envs/tabpfn
export TABPFN_MODEL_CACHE_DIR=/scratch/$USER/tabpfn_models
export OMP_NUM_THREADS=8 PYTHONWARNINGS=ignore
cd /scratch/$USER/icwm/pilot
echo "$(date) node $(hostname): $1 seed $2"; nvidia-smi -L
python -c "import tabpfn; print('tabpfn', tabpfn.__version__)"
python pilot.py --system "$1" --seed_ids "$2" --tabpfn_version v3.5 --out "results_$1_s$2.json"
echo "$(date) finished"
