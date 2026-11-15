#!/bin/bash
# CPU version of pack.sbatch (for CPU-only models such as the GP). Usage: sbatch --job-name=<n> pack_cpu.sbatch <taskfile> <from> <to>
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
module load miniconda/3-4.11.0
source activate /scratch/$USER/conda-envs/tabpfn
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd /scratch/$USER/icwm/pilot
echo "$(date) node $(hostname): $1 lines $2-$3"
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
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "$(date) finished, failed processes: $fail"
