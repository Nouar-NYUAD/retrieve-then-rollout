#!/bin/bash
# Timing check: TabPFN MPC on cart-pole, short episode, cached vs default fit mode.
#SBATCH --job-name=icwm_mpct
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
module load miniconda/3-4.11.0
source activate /scratch/$USER/conda-envs/tabpfn
export TABPFN_MODEL_CACHE_DIR=/scratch/$USER/tabpfn_models
export OMP_NUM_THREADS=8 PYTHONWARNINGS=ignore
cd /scratch/$USER/icwm/pilot
echo "$(date) node $(hostname)"; nvidia-smi -L
for fm in fit_with_cache fit_preprocessors; do
  python mpc.py --task cartpole --models tabpfn --strategies retrieval --seed_ids 0 --T_ep 5 --tabpfn_fit_mode $fm --out timing_$fm.json
done
echo "$(date) finished"
