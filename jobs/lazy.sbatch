#!/bin/bash
# Lazy-retrieval + DynaPFN test. Usage: sbatch lazy.sbatch <system> <retrieve_every>
#SBATCH --job-name=icwm_lazy
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
module load miniconda/3-4.11.0
source activate /scratch/$USER/conda-envs/tabpfn
export TABPFN_MODEL_CACHE_DIR=/scratch/$USER/tabpfn_models
export OMP_NUM_THREADS=8 PYTHONWARNINGS=ignore
cd /scratch/$USER/icwm/pilot
echo "$(date) node $(hostname): $1 k $2"; nvidia-smi -L
T0=600; [ "$1" = hopper ] && T0=100
MODELS="tabpfn"; [ "$2" = 1 ] && MODELS="tabpfn dynapfn"
STRATS="retrieval"; [ "$2" = 1 ] && STRATS="retrieval recent"
python study.py --system "$1" --t0_min $T0 --n_train 5 --models $MODELS --strategies $STRATS --H 100 --seed_ids 0 1 --n_starts 2 --horizon 100 --report_h 1 10 25 50 100 --retrieve_every "$2" --out "lazy_$1_k$2.json"
echo "$(date) finished"
