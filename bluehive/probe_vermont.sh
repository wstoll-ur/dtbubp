#!/bin/bash
# Read-only check of the Vermont nodes (outside Slurm) for running CP2K there.
#   bash bluehive/probe_vermont.sh "bhx0131 bhx0132 bhx0133"
# Per node: cores, memory, load and who is using it, shared file systems (is /scratch visible?), apptainer,
# modules, Intel MPI. Writes bluehive/reports/probe_vermont.txt and pushes it.
cd "$(git rev-parse --show-toplevel)" || exit 1
NODES=${1:?give the node list, e.g. \"bhx0131 bhx0132 bhx0133\"}
REPO=$PWD
OUT=bluehive/reports/probe_vermont.txt
{
echo "# probe_vermont $(date -Is) from $(hostname); repo $REPO; SCRATCH=$SCRATCH"
for h in $(echo $NODES | tr ',' ' '); do
  echo; echo "################ $h"
  ssh -o ConnectTimeout=10 -o BatchMode=yes $h bash -l -s <<REMOTE 2>&1
echo "host \$(hostname)  nproc \$(nproc)"; lscpu | grep -E "Model name|^CPU\(s\)|Socket|Core|NUMA node\(s\)"
free -g | head -2; uptime
echo "-- top cpu users"; ps -eo user,pcpu,etime,comm --sort=-pcpu | head -6
echo "-- file systems"; df -h /home/$USER /scratch/$USER /gpfs/fs2 /software 2>&1 | grep -v "^Filesystem"
ls -d $REPO >/dev/null 2>&1 && echo "repo visible: $REPO" || echo "repo NOT visible: $REPO"
touch $REPO/.write_test_\$(hostname) 2>/dev/null && { echo "repo writable"; rm -f $REPO/.write_test_\$(hostname); } || echo "repo NOT writable"
ls /scratch/$USER/containers 2>&1 | head -3
echo "-- tools"; type module 2>&1 | head -1
for x in apptainer singularity mpirun; do echo "\$x: \$(which \$x 2>&1)"; done
module avail apptainer 2>&1 | grep -i apptainer | head -3
module load apptainer/1.4.1 2>&1 | head -2; echo "after module load: \$(which apptainer 2>&1)"
ls /tmp | head -0; df -h /tmp | tail -1
REMOTE
done
} > $OUT 2>&1
cat $OUT
git add $OUT && git commit -qm "Vermont nodes probe $(date +%F_%H%M)" && git push -q && echo pushed
