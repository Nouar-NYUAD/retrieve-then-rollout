#!/bin/bash
# Packed GPU job: runs every line of a task file as its own study.py process, all sharing ONE GPU,
# so the GPU stays busy (Jubail terminates GPU jobs with minimal usage). Each line = study.py args.
# Usage: sbatch --job-name=<name> pack.sbatch <taskfile> <line_from> <line_to>
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80g
#SBATCH --exclude=cn263,cn016
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
module load miniconda/3-4.11.0
source activate /scratch/$USER/conda-envs/tabpfn
export TABPFN_MODEL_CACHE_DIR=/scratch/$USER/tabpfn_models PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
cd /scratch/$USER/icwm/sliding
echo "$(date) node $(hostname): $1 lines $2-$3"; nvidia-smi -L
mkdir -p logs/tasks
logs=()
# code snapshot for THIS job, so later edits of the shared files cannot change a running job
SNAP=".code_${SLURM_JOB_ID}"; mkdir -p "$SNAP"; cp study.py pilot.py "$SNAP"/
echo "code snapshot $SNAP: $(grep -o 'CODE_VERSION = "[^"]*"' $SNAP/study.py)"
pids=()
while read -r line; do
  [ -z "$line" ] && continue
  tag=$(echo "$line" | grep -o -- "--out [^ ]*" | cut -d" " -f2 | sed "s/.json//")
  python "$SNAP/study.py" $line > "logs/tasks/${tag}.log" 2>&1 &
  pids+=($!); logs+=("logs/tasks/${tag}.log"); echo "started $tag (pid $!)"
done < <(sed -n "$2,$3p" "$1")
( while sleep 600; do echo "gpu $(date +%H:%M) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; done ) &
mon=$!
# stall watchdog: a task whose log has not changed for STALL_SEC seconds is killed and reported
# (a hung task would otherwise idle the GPU until the cluster cancels the whole job)
STALL_SEC=${STALL_SEC:-14400}; WD_EVERY=${WD_EVERY:-300}
( while sleep "$WD_EVERY"; do for i in "${!pids[@]}"; do p=${pids[$i]}; lg=${logs[$i]}
    if kill -0 "$p" 2>/dev/null; then age=$(( $(date +%s) - $(stat -c %Y "$lg") ))
      if [ "$age" -gt "$STALL_SEC" ]; then echo "STALLED $(basename "$lg" .log) (pid $p, log unchanged ${age}s) - killed"; kill "$p"; fi; fi
  done; done ) &
wd=$!
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
kill $mon $wd 2>/dev/null
echo "$(date) finished, failed processes: $fail"
