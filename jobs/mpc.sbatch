#!/bin/bash
# Cart-pole MPC pilot episode with TabPFN for one context strategy and seed.
# Usage: sbatch mpc.sbatch <retrieval|recent|random> <seed>
#SBATCH --job-name=icwm_mpc
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
module load miniconda/3-4.11.0
source activate /scratch/$USER/conda-envs/tabpfn
export TABPFN_MODEL_CACHE_DIR=/scratch/$USER/tabpfn_models
export OMP_NUM_THREADS=8 PYTHONWARNINGS=ignore
cd /scratch/$USER/icwm/pilot
echo "$(date) node $(hostname): $1 seed $2"; nvidia-smi -L
python mpc.py --task cartpole --models tabpfn --strategies "$1" --seed_ids "$2" --tabpfn_fit_mode fit_preprocessors --out "mpc_cartpole_tabpfn_$1_s$2.json"
echo "$(date) finished"
