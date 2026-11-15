#!/bin/bash
# Job watcher (runs on the Jubail login node, called every 10 min). Prints ONLY changes / problems.
source /etc/profile >/dev/null 2>&1
ST=/scratch/naa9497/icwm/.watch; mkdir -p $ST
cur=$(squeue -u naa9497 -h -o "%i %j %T" | sort)
prev=$(cat $ST/jobs 2>/dev/null)
echo "$cur" > $ST/jobs
# jobs that left the queue -> final state + failures reported in their log
for id in $(comm -23 <(echo "$prev" | awk '{print $1}' | sort) <(echo "$cur" | awk '{print $1}' | sort)); do
  s=$(sacct -j $id -X -n -o JobName%16,State%12,Elapsed | tr -s " ")
  lg=$(ls /scratch/naa9497/icwm/*/logs/*_$id.out 2>/dev/null | head -1)
  extra=$(grep -h -o -E "failed processes: [0-9]+|^STALLED [^ ]+|CANCELLED AT [^ ]+|Traceback|Error[A-Za-z]*" $lg 2>/dev/null | sort | uniq -c | tr -s " " | tr "\n" ";")
  echo "ENDED $id $s | $extra"
done
# PENDING -> RUNNING transitions
for id in $(echo "$cur" | awk '$3=="RUNNING"{print $1}'); do
  echo "$prev" | grep -q "^$id .* PENDING" && echo "STARTED $id $(echo "$cur" | grep "^$id" | awk '{print $2}')"
done
# running jobs whose GPU has been at 0 % for the last 6 log samples (~1 h): cluster kills after 2 h
for id in $(echo "$cur" | awk '$3=="RUNNING"{print $1}'); do
  lg=$(ls /scratch/naa9497/icwm/*/logs/*_$id.out 2>/dev/null | head -1); [ -z "$lg" ] && continue
  n=$(grep "^gpu" $lg | tail -6 | grep -c " 0 %,")
  [ "$n" -ge 6 ] && echo "GPU-IDLE $id ($(basename $lg)) last 6 samples 0 %"
done
# new crashes in per-task logs (count of files with a Traceback / OOM changed)
for d in cheetah pilot camels2 noise abl2 plan; do
  c=$(grep -l "Traceback\|OutOfMemory" /scratch/naa9497/icwm/$d/logs/tasks/*.log 2>/dev/null | sort)
  p=$(cat $ST/crash_$d 2>/dev/null); echo "$c" > $ST/crash_$d
  new=$(comm -13 <(echo "$p") <(echo "$c") | grep -v '^$' | xargs -r -n1 basename | tr "\n" " ")
  [ -n "$new" ] && [ -n "$p$c" ] && echo "CRASH in $d: $new"
done
exit 0
