#!/bin/bash
# Run dtbubp job scripts on the BlueHive Vermont nodes (no Slurm, NO shared file system with bluehive3).
# Like vermont_crystal.sh: copy the job folder to the node, start it with nohup over ssh, copy results back.
#
#   bash bluehive/vermont.sh run  <job_script.sh> <node>[,node2,...] [task_id] [VAR=val ...]  # copy job folder -> node(s), start on the first
#   bash bluehive/vermont.sh pull [tag]                              # copy outputs back (all runs, or one)
#   bash bluehive/vermont.sh watch [minutes]                         # pull every N min (default 30) until nothing runs
#   bash bluehive/vermont.sh status                                  # running/finished + load per node
#   bash bluehive/vermont.sh kill <tag>                              # stop a run (then pull)
#   bash bluehive/vermont.sh stopall <node>[,node]                   # stop ALL our CP2K jobs on node(s), registered or not
# run refuses to start on a node that already runs our cp2k.psmp (FORCE=1 to override).
#
# Job scripts are the ones dtbubp writes with --dry-run (plain bash; #SBATCH lines are comments).
# On the node the repo root  <repo>  is replaced by  /home/$USER/dtbubp_runs  (node-local disk).
# Runs are registered in bluehive/vermont_runs/<tag>.txt.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
REPO=$PWD
RROOT=/home/$USER/dtbubp_runs
REG=$REPO/bluehive/vermont_runs
mkdir -p $REG
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
cmd=${1:-}; shift || true

remote_of() { echo "$RROOT/${1#$REPO/}"; }

pull_one() {   # $1 = registry file
  read h pid script task ldir rdir < $1
  rsync -a --update --exclude '*.wfn*' --exclude '*.bak-*' -e "$SSH" $h:$rdir/ $ldir/ \
    && echo "pulled $(basename $1 .txt) from $h -> ${ldir#$REPO/}"
}

case "$cmd" in
run)
  # <node> may be a comma list (multi-node MPI run): the folder is copied to every node (no shared file
  # system; OpenMPI needs the working directory everywhere), the job starts on the first, results live there.
  [ -f "${1:-}" ] || { echo "job script not found: ${1:-} (run the dtbubp --dry-run step first)"; exit 1; }
  [ -n "${2:-}" ] || { echo "give a node"; exit 1; }
  script=$(realpath "$1"); nodes=$(echo $2 | tr ',' ' '); h=$(echo $nodes | awk '{print $1}'); task=${3:-0}
  shift 3 2>/dev/null || shift $#; envs="$*"     # optional VAR=value ... (e.g. DTB_LANES=1 DTB_RANKS=16)
  ldir=$(dirname "$script"); rdir=$(remote_of "$ldir"); name=$(basename "$script" .sh)
  tag=${name}_${h}_t${task}_$(date +%m%d_%H%M%S)
  # refuse to start a second CP2K job on a node that already runs ours (FORCE=1 overrides)
  for n in $nodes; do
    busy=$($SSH $n "pgrep -u $USER -c cp2k.psmp" < /dev/null)
    if [ "${busy:-0}" -gt 0 ] && [ "${FORCE:-0}" != 1 ]; then
      echo "$n already runs $busy cp2k.psmp processes of $USER -> not starting (stop them first: bash bluehive/vermont.sh stopall $n)"; exit 1
    fi
  done
  for n in $nodes; do
    $SSH $n "mkdir -p $rdir" || exit 1
    # inputs (and any finished outputs, so finished frames are skipped); job script with node-local paths.
    # A labelling folder (chunk_<task>.txt present) copies only that chunk's frame folders, not all 2200.
    if [ -f $ldir/chunk_$task.txt ]; then
      list=$(mktemp); { echo chunk_$task.txt; cat $ldir/chunk_$task.txt; } > $list
      echo "copying chunk $task ($(($(wc -l < $list) - 1)) frame folders) to $n:$rdir ..."
      rsync -a -r --files-from=$list --exclude '*_vermont_*.log' -e "$SSH" $ldir/ $n:$rdir/ || { rm -f $list; exit 1; }
      rm -f $list
    else
      echo "copying $(du -sh $ldir | cut -f1) to $n:$rdir ..."
      rsync -a --exclude '*_vermont_*.log' -e "$SSH" $ldir/ $n:$rdir/ || exit 1
    fi
    echo "  copied"
    sed -e "s#$REPO#$RROOT#g" -e "s#/scratch/$USER/DtBuDp/DtBuDp#$RROOT#g" -e "s#/gpfs/fs2$RROOT#$RROOT#g" $script \
      | $SSH $n "cat > $rdir/$name.vermont.sh"
  done
  log=$rdir/${name}_vermont_t$task.log
  pid=$($SSH $h "cd $rdir && SLURM_ARRAY_TASK_ID=$task SLURM_JOB_ID=$tag $envs setsid nohup bash -l $name.vermont.sh > $log 2>&1 < /dev/null & echo \$!")
  echo "$h $pid $name task=$task $ldir $rdir" > $REG/$tag.txt
  echo "started $tag: $h pid $pid, node dir $rdir, log $log"
  ;;
pull)
  for f in $REG/${1:-*}.txt; do [ -f "$f" ] && pull_one $f; done
  ;;
status)
  for f in $REG/*.txt; do
    [ -f "$f" ] || continue
    read h pid script task ldir rdir < $f
    st=$($SSH $h "ps -p $pid >/dev/null && echo RUNNING || echo finished; cut -d' ' -f1-3 /proc/loadavg" | tr '\n' ' ')
    echo "$(basename $f .txt): $st  (${ldir#$REPO/})"
  done
  ;;
watch)
  m=${1:-30}
  while :; do
    for f in $REG/*.txt; do [ -f "$f" ] && pull_one $f; done
    alive=0
    for f in $REG/*.txt; do
      [ -f "$f" ] || continue; read h pid rest < $f
      $SSH $h "ps -p $pid >/dev/null" && alive=$((alive + 1))
    done
    echo "$(date '+%F %T') $alive run(s) still going"
    [ $alive -eq 0 ] && break
    sleep $((m * 60))
  done
  ;;
kill)
  f=$REG/$1.txt; [ -f "$f" ] || { echo "no run $1"; exit 1; }
  read h pid rest < $f
  $SSH $h "pkill -TERM -g $pid; sleep 3; pkill -KILL -g $pid; true"
  pull_one $f; mv $f $f.killed; echo "stopped $1"
  ;;
stopall)
  # stop EVERY job of ours started by vermont.sh on the given node(s), registered or not (job scripts, mpirun, CP2K)
  for h in $(echo $1 | tr ',' ' '); do
    echo "== $h: before: $($SSH $h "pgrep -u $USER -c cp2k.psmp" < /dev/null) cp2k.psmp"
    $SSH $h "pkill -u $USER -f '\.vermont\.sh'; pkill -u $USER -x mpirun; pkill -u $USER -x cp2k.psmp; sleep 3; \
             pkill -9 -u $USER -x cp2k.psmp; true" < /dev/null
    echo "   after: $($SSH $h "pgrep -u $USER -c cp2k.psmp" < /dev/null) cp2k.psmp"
    for f in $REG/*.txt; do [ -f "$f" ] && read rh rest < $f && [ "$rh" = "$h" ] && mv $f $f.stopped; done
  done
  ;;
*) sed -n 2,16p "$0"; exit 1 ;;
esac
