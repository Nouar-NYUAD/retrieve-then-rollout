#!/bin/bash
# Diagnostics on a CPU node. Usage: sbatch diag_cpu.sbatch <d1|d2|d3>
#SBATCH --job-name=icwm_diag
#SBATCH --partition=compute
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
module load miniconda/3-4.11.0
source activate /scratch/$USER/conda-envs/tabpfn
export TABPFN_MODEL_CACHE_DIR=/scratch/$USER/tabpfn_models OMP_NUM_THREADS=28 PYTHONWARNINGS=ignore
cd /scratch/$USER/icwm/gp
SNAP=".code_${SLURM_JOB_ID}"; mkdir -p "$SNAP"; cp study.py pilot.py diag.py "$SNAP"/
echo "$(date) part $1, snapshot $SNAP"
python -W ignore "$SNAP/diag.py" --parts "$1" --d2_queries 30 --out "diag_$1.json"
echo "$(date) finished"
