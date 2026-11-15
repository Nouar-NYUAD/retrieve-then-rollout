#!/bin/bash
# Refresh the task logs of RUNNING pilot jobs whose watchdog uses the old 90-min threshold, so slow
# (not hung) configurations are not killed as stalled. True hangs are still caught by Jubail's
# 2-hour idle-GPU rule. Prints the number of logs touched.
cd /scratch/naa9497/icwm/pilot/logs
n=0
for j in $(squeue -h -u naa9497 -t R -o "%i %j" | awk '$2 ~ /^icwm_(d4rl|l96|full)$/ {print $1}'); do
  f=$(ls icwm_*_$j.out 2>/dev/null | head -1); [ -n "$f" ] || continue
  grep -q "stall watchdog\|STALL_SEC" /var/spool/slurmd/job$j/slurm_script 2>/dev/null || true
  for t in $(grep -o "^started [^ ]*" "$f" | cut -d" " -f2); do touch "tasks/$t.log" && n=$((n+1)); done
done
echo "$(date +%H:%M) touched $n task logs"
