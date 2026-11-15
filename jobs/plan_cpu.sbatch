#!/bin/bash
# CPU planning baselines (oracle + local linear with every context strategy).
#SBATCH --job-name=icwm_planbase
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
module load miniconda/3-4.11.0
source activate /scratch/$USER/conda-envs/tabpfn
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=2
cd /scratch/$USER/icwm/plan
SNAP=".code_${SLURM_JOB_ID}"; mkdir -p "$SNAP"; cp mpc.py pilot.py "$SNAP"/; echo "snapshot $(md5sum $SNAP/mpc.py)"
python "$SNAP/mpc.py" --task cartpole --models oracle locallin --strategies random recent retrieval trajretrieval hybrid --seed_ids 0 1 2 3 4 --out planbase_cartpole.json &
python "$SNAP/mpc.py" --task acrobot --models oracle locallin --strategies random recent retrieval trajretrieval hybrid --seed_ids 0 1 2 --P 40 --T_ep 250 --out planbase_acrobot.json &
wait; echo "$(date) finished"
