#!/bin/bash
# Hopper oracle-planner sweep: planning horizon x fall penalty (simulator only, ~20 min).
#SBATCH --job-name=icwm_hopsweep
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=cn263,cn016
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
module load miniconda/3-4.11.0
source activate /scratch/$USER/conda-envs/tabpfn-mujoco
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
cd /scratch/$USER/icwm/cheetah
SNAP=".code_${SLURM_JOB_ID}"; mkdir -p "$SNAP"; cp cheetah_mpc_dev.py "$SNAP"/cheetah_mpc.py; cp pilot.py "$SNAP"/
for cfg in "15 100" "40 0" "40 100" "60 100" "80 100" "40 1000"; do
  set -- $cfg
  python -W ignore "$SNAP/cheetah_mpc.py" --env hopper --controller oracle --strategy - --T 400 --P $1 --fall_penalty $2 \
     --out sweep_hop_P$1_f$2.json 2>&1 | grep "^\[hopper" | sed "s/^/P=$1 penalty=$2 /" &
done
wait
echo "$(date) finished"
