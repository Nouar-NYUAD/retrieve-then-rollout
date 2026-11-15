#!/bin/bash
# Enforce "at most ~N GPU jobs at once" with Slurm dependencies: the running GPU jobs of the
# study are chain heads; every pending GPU job is appended round-robin to a chain with
# afterany:<previous job in that chain>, in priority order. Usage: bash chain_jobs.sh <N>
N=${1:-10}
mapfile -t heads < <(squeue -h -u $USER -t R -p nvidia -o "%i %j" | awk '$2 ~ /^icwm_full/ {print $1}' | head -n $N)
order="icwm_d4rl icwm_cam icwm_noise icwm_ver icwm_l96"
pend=()
for n in $order; do
  while read -r j; do [ -n "$j" ] && pend+=("$j"); done < <(squeue -h -u $USER -t PD -n $n -o "%i" | sort -n)
done
echo "chain heads (${#heads[@]}): ${heads[*]}"; echo "pending GPU jobs: ${#pend[@]}"
declare -a tail_of; for k in "${!heads[@]}"; do tail_of[$k]=${heads[$k]}; done
k=0
for j in "${pend[@]}"; do
  c=$((k % ${#heads[@]})); prev=${tail_of[$c]}
  scontrol update JobId=$j Dependency=afterany:$prev && echo "chain $c: $prev -> $j"
  tail_of[$c]=$j; k=$((k+1))
done
