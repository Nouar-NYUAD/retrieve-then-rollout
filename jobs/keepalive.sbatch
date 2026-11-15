#!/bin/bash
# Keep-alive for packed jobs submitted with the old 90-min stall watchdog: touches the task logs of
# running icwm d4rl/l96/full jobs every 20 min so slow (not hung) configurations are not killed.
#SBATCH --job-name=icwm_keepalive
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
while true; do bash /scratch/$USER/icwm/pilot/keepalive.sh; n=$(squeue -h -u $USER -t R,PD -o %j | grep -cE "^icwm_(d4rl|l96|full)$"); [ "$n" = "0" ] && { echo "no target jobs left"; break; }; sleep 1200; done
