#!/bin/bash
# Run dtbubp job scripts on Vermont nodes (outside Slurm), like vermont_crystal.sh: nohup over ssh.
# The job scripts written by dtbubp (`--dry-run` writes them without sbatch) are plain bash; #SBATCH lines are
# comments. Array jobs are started once per node with SLURM_ARRAY_TASK_ID = 0, 1, ...
#
#   bash bluehive/vermont.sh run  <job_script.sh> "<node0> [node1 ...]"   # task i on node i
#   bash bluehive/vermont.sh status                                       # what runs where (from the run registry)
#   bash bluehive/vermont.sh kill <tag>                                   # stop one launch (all its nodes)
#
# Needs: the repo on a file system the nodes see (probe_vermont.sh: "repo visible/writable").
# Every launch is recorded in bluehive/vermont_runs/<tag>.txt (node, pid, script, task id, log file).
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
REG=$PWD/bluehive/vermont_runs
mkdir -p $REG
cmd=${1:-}; shift || true

case "$cmd" in
run)
  script=$(realpath "$1"); nodes=$(echo "$2" | tr ',' ' ')
  dir=$(dirname "$script"); name=$(basename "$script" .sh)
  tag=${name}_$(date +%Y%m%d_%H%M%S)
  i=0
  for h in $nodes; do
    log=$dir/${name}_vermont_${h}_task$i.log
    pid=$(ssh -o BatchMode=yes $h "cd $dir && SLURM_ARRAY_TASK_ID=$i SLURM_JOB_ID=$tag setsid nohup bash -l $script > $log 2>&1 < /dev/null & echo \$!")
    echo "$h $pid $script task=$i log=$log" | tee -a $REG/$tag.txt
    i=$((i + 1))
  done
  echo "registered as $tag  (stop: bash bluehive/vermont.sh kill $tag)"
  ;;
status)
  for f in $REG/*.txt; do
    [ -f "$f" ] || continue
    echo "== $(basename $f .txt)"
    while read h pid script task log; do
      alive=$(ssh -o BatchMode=yes -o ConnectTimeout=5 $h "ps -p $pid >/dev/null && echo running || echo finished")
      load=$(ssh -o BatchMode=yes -o ConnectTimeout=5 $h "cut -d' ' -f1 /proc/loadavg")
      echo "  $h pid $pid $task: $alive (load $load)  ${log#log=}"
    done < $f
  done
  ;;
kill)
  f=$REG/$1.txt; [ -f "$f" ] || { echo "no run $1"; exit 1; }
  while read h pid script task log; do
    echo "stopping $1 on $h (pid $pid)"
    # the job script's process group, then any CP2K/mpirun left on that node by this user
    ssh -o BatchMode=yes $h "pkill -TERM -g $pid 2>/dev/null; sleep 2; \
         pkill -u $USER -f cp2k.psmp; pkill -u $USER -f 'mpirun -np'; true"
  done < $f
  mv $f $f.killed
  ;;
*) sed -n 2,12p "$0"; exit 1 ;;
esac
