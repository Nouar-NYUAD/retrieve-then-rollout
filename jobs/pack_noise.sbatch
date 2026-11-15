#!/bin/bash
# Packed GPU job: runs every line of a task file as its own study.py process, all sharing ONE GPU,
# so the GPU stays busy (Jubail terminates GPU jobs with minimal usage). Each line = study.py args.
# Usage: sbatch --job-name=<name> pack.sbatch <taskfile> <line_from> <line_to>
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
module load miniconda/3-4.11.0
source activate /scratch/$USER/conda-envs/tabpfn
export TABPFN_MODEL_CACHE_DIR=/scratch/$USER/tabpfn_models PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
cd /scratch/$USER/icwm/noise
echo "$(date) node $(hostname): $1 lines $2-$3"; nvidia-smi -L
mkdir -p logs/tasks
# code snapshot for THIS job, so later edits of the shared files cannot change a running job
SNAP=".code_${SLURM_JOB_ID}"; mkdir -p "$SNAP"; cp study.py pilot.py "$SNAP"/
echo "code snapshot $SNAP: $(grep -o 'CODE_VERSION = "[^"]*"' $SNAP/study.py)"
pids=()
while read -r line; do
  [ -z "$line" ] && continue
  tag=$(echo "$line" | grep -o -- "--out [^ ]*" | cut -d" " -f2 | sed "s/.json//")
  python "$SNAP/study.py" $line > "logs/tasks/${tag}.log" 2>&1 &
  pids+=($!); echo "started $tag (pid $!)"
done < <(sed -n "$2,$3p" "$1")
( while sleep 600; do echo "gpu $(date +%H:%M) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; done ) &
mon=$!
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
kill $mon 2>/dev/null
echo "$(date) finished, failed processes: $fail"
