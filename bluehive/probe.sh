#!/bin/bash
# BlueHive discovery for the DtBuDp CP2K campaign. Read-only: installs nothing, submits nothing.
# Run from the repo root on a BlueHive login node:   bash bluehive/probe.sh
# Writes bluehive/reports/probe.txt, then commits and pushes it so Claude can read it.
cd "$(git rev-parse --show-toplevel)" || exit 1
OUT=bluehive/reports/probe.txt
mkdir -p bluehive/reports
{
echo "# probe $(date -Is) on $(hostname) as $USER"
echo; echo "## OS / CPU"; uname -a; lscpu | grep -E "Model name|^CPU\(s\)|Socket|Core|Thread|NUMA node\(s\)"; free -g | head -2
echo; echo "## Slurm partitions (partition avail timelimit nodes cpus mem gres state)"
sinfo -o "%P %a %l %D %c %m %G %t" 2>&1 | sort -u | head -60
echo; echo "## My associations (account partition qos)"
sacctmgr -nP show assoc user=$USER format=account,partition,qos,maxwall 2>&1 | head -30
echo; echo "## Free/idle nodes per partition"
sinfo -t idle -o "%P %D %c %m %N" 2>&1 | head -20
echo; echo "## My jobs"; squeue -u $USER 2>&1 | head -20
echo; echo "## Storage"; echo "HOME=$HOME  PWD=$PWD"; df -h $HOME /scratch/$USER 2>&1 | tail -n +1; quota -s 2>&1 | head -10
echo; echo "## CP2K modules"
module -t avail 2>&1 | grep -i cp2k
module spider cp2k 2>&1 | grep -iE "cp2k/|versions" | head -20
echo; echo "## each CP2K module: executables, version, data files"
for m in $(module -t avail 2>&1 | grep -i '^cp2k'); do
  echo "### $m"
  ( module purge >/dev/null 2>&1; module load $m 2>&1 | head -5
    module list 2>&1 | tail -n +2
    for x in cp2k.popt cp2k.psmp cp2k.sopt cp2k.ssmp cp2k; do p=$(which $x 2>/dev/null) && echo "exe $x -> $p"; done
    exe=$(which cp2k.psmp 2>/dev/null || which cp2k.popt 2>/dev/null)
    [ -n "$exe" ] && OMP_NUM_THREADS=1 timeout 60 $exe --version 2>&1 | head -8
    echo "CP2K_DATA_DIR=$CP2K_DATA_DIR"
    for d in "$CP2K_DATA_DIR" $(dirname $(dirname "$exe" 2>/dev/null) 2>/dev/null)/share/cp2k/data $(dirname $(dirname "$exe" 2>/dev/null) 2>/dev/null)/data; do
      [ -f "$d/BASIS_MOLOPT" ] || continue
      echo "data dir: $d"
      grep -E "^ *(C|H) +(TZV2P-MOLOPT-GTH|DZVP-MOLOPT-SR-GTH)" $d/BASIS_MOLOPT
      grep -E "^ *(C|H) +GTH-PBE-q(1|4)" $d/POTENTIAL | head
      ls -la $d/dftd3.dat 2>&1
      break
    done
    which mpirun srun 2>&1; mpirun --version 2>&1 | head -2 )
done
echo; echo "## other ways to get CP2K"
for x in apptainer singularity conda mamba micromamba spack; do p=$(which $x 2>/dev/null) && echo "$x -> $p"; done
module -t avail 2>&1 | grep -iE "^(apptainer|singularity|anaconda|miniconda|miniforge|conda|python|openmpi|intel|impi|gcc)" | head -40
echo; echo "## python for dtbubp (needs numpy; TOML via python>=3.11 or tomli)"
for py in python3 python; do p=$(which $py 2>/dev/null) && echo "$py -> $p: $($py -c 'import sys;print(sys.version.split()[0])' 2>&1) numpy=$($py -c 'import numpy;print(numpy.__version__)' 2>&1 | tail -1)"; done
echo; echo "## srun MPI plugins"; srun --mpi=list 2>&1 | head -10
} > $OUT 2>&1
echo "wrote $OUT ($(wc -l < $OUT) lines)"
git add $OUT && git commit -qm "BlueHive probe $(date +%F_%H%M)" && git push -q && echo "pushed"
