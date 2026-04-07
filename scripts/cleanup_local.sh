#!/bin/bash
# Cleanup local experiment files on a specific node after all jobs are done.
#
# Usage (via srun on target node):
#   srun -p rtx3090 -w <node> --time=00:05:00 -n1 --cpus-per-task=1 --mem=1G \
#       bash /data_fast/home/sihun/diffprotein/dplm/cleanup_local.sh
#
# Or to clean all idle nodes at once:
#   for node in nutella orange tomato udon watermelon; do
#       srun -p rtx3090 -w $node --time=00:05:00 -n1 --cpus-per-task=1 --mem=1G \
#           bash /data_fast/home/sihun/diffprotein/dplm/cleanup_local.sh &
#   done; wait

LOCAL_BASE="/data_large/unsynced_store/sihun/diffprotein/dplm"

if [ ! -d "$LOCAL_BASE" ]; then
    echo "[$(hostname)] Nothing to clean: $LOCAL_BASE does not exist"
    exit 0
fi

echo "[$(hostname)] Cleaning up $LOCAL_BASE ..."
du -sh "$LOCAL_BASE" 2>/dev/null

rm -rf "$LOCAL_BASE"
echo "[$(hostname)] Done. Removed $LOCAL_BASE"
