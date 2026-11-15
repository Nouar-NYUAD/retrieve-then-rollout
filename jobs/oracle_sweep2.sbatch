#!/bin/bash
# Hopper oracle-planner sweep 2: larger CEM budget (candidates x iterations) at long horizons.
#SBATCH --job-name=icwm_hopsweep2
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cn263,cn016
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
module load miniconda/3-4.11.0
source activate /scratch/$USER/conda-envs/tabpfn-mujoco
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
cd /scratch/$USER/icwm/cheetah
SNAP=".code_${SLURM_JOB_ID}"; mkdir -p "$SNAP"; cp cheetah_mpc_dev.py "$SNAP"/cheetah_mpc.py; cp pilot.py "$SNAP"/
for cfg in "40 1000 5" "60 1000 5" "60 512 5" "100 1000 5"; do
  set -- $cfg
  python -W ignore "$SNAP/cheetah_mpc.py" --env hopper --controller oracle --strategy - --T 400 --P $1 --n_cand $2 \
     --n_iter $3 --n_elite 50 --fall_penalty 100 --out sweep2_hop_P$1_c$2.json 2>&1 | grep "^\[hopper" \
     | sed "s/^/P=$1 cand=$2 iter=$3 /" &
done
wait
echo "$(date) finished"
