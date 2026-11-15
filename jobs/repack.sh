#!/bin/bash
# Cancel pending jobs whose packing exceeds GPU memory, splicing each out of its afterany chain
# (successor -> predecessor) so the concurrency limit is preserved. Usage: bash repack.sh JOB...
for J in "$@"; do
  st=$(squeue -h -j $J -o %T 2>/dev/null)
  [ "$st" != "PENDING" ] && { echo "skip $J (state '$st')"; continue; }
  pred=$(squeue -h -j $J -o %E | grep -o 'afterany:[0-9]*' | cut -d: -f2)
  for S in $(squeue -h -u $USER -t PD -o "%i %E" | awk -v j="afterany:$J" 'index($2, j) {print $1}'); do
    if [ -n "$pred" ]; then scontrol update JobId=$S Dependency=afterany:$pred && echo "  $S now after $pred"; else scontrol update JobId=$S Dependency=afterany:${FALLBACK:?set FALLBACK} && echo "  $S now after fallback $FALLBACK"; fi
  done
  scancel $J && echo "cancelled $J (was after ${pred:-none})"
done
