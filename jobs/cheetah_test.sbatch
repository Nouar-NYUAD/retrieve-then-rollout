#!/bin/bash
# GPU checks for the HalfCheetah demo: equality of parallel workers, speed settings, offscreen video.
#SBATCH --job-name=icwm_chtest
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn263
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
module load miniconda/3-4.11.0
source activate /scratch/$USER/conda-envs/tabpfn-mujoco
export TABPFN_MODEL_CACHE_DIR=/scratch/$USER/tabpfn_models PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 MUJOCO_GL=egl
cd /scratch/$USER/icwm/cheetah
SNAP=".code_${SLURM_JOB_ID}"; mkdir -p "$SNAP"; cp cheetah_mpc.py cheetah_test.py pilot.py "$SNAP"/
echo "$(date) node $(hostname)"; nvidia-smi -L
python -W ignore "$SNAP/cheetah_test.py" --out test_equality.json
for cfg in "v3.5 1 1" "v3.5 6 1" "v3.5 17 1" "v3.5-fast 17 1" "v3.5-fast 17 2"; do
  set -- $cfg
  python -W ignore "$SNAP/cheetah_mpc.py" --controller tabpfn --strategy trajretrieval --T 4 \
     --tabpfn_version $1 --workers $2 --replan_every $3 --out "speed_$1_w$2_r$3.json" 2>&1 | grep -E "cheetah s0|Traceback|Error"
done
python -W ignore "$SNAP/cheetah_mpc.py" --controller oracle --strategy - --T 40 --video_dir videos --out video_test.json 2>&1 | grep -E "cheetah s0|Traceback|Error"
ls -la videos
echo "$(date) finished"
