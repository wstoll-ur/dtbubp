#!/bin/bash
# Get CP2K on BlueHive WITHOUT installing anything into your environments: the official CP2K container
# (same version, 2024.1, as every Leonardo DFT run of this project) pulled as one .sif file into scratch.
#   bash bluehive/setup_cp2k.sh            (login node; needs network to Docker Hub; ~1-2 GB, a few minutes)
# With TEST_NODE=<vermont node> it also runs a 1-minute test there over ssh. Reports are pushed.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
TAG=${CP2K_TAG:-2024.1_openmpi_generic_psmp}
DIR=/scratch/$USER/containers
SIF=$DIR/cp2k_$TAG.sif
R=bluehive/reports/setup_cp2k.txt
mkdir -p $DIR bluehive/reports
export APPTAINER_CACHEDIR=/scratch/$USER/.apptainer_cache     # keep the layer cache out of $HOME
module load apptainer/1.4.1
{
echo "# setup_cp2k $(date -Is) on $(hostname)"
if [ ! -s $SIF ]; then apptainer pull $SIF docker://cp2k/cp2k:$TAG 2>&1 | tail -5; fi
ls -la $SIF
echo "## version";   apptainer exec --cleanenv $SIF cp2k.psmp --version 2>&1 | head -6
echo "## mpirun";    apptainer exec --cleanenv $SIF mpirun --version 2>&1 | head -2
echo "## data dir"
apptainer exec --cleanenv $SIF bash -c 'for d in $CP2K_DATA_DIR /opt/cp2k/data /opt/cp2k/share/cp2k/data; do
  [ -f $d/BASIS_MOLOPT ] || continue; echo "data: $d"
  grep -E "^ *(C|H) +(TZV2P-MOLOPT-GTH|DZVP-MOLOPT-SR-GTH)( |$)" $d/BASIS_MOLOPT
  grep -E "^ *(C|H) +GTH-PBE-q(1|4)( |$)" $d/POTENTIAL; ls $d/dftd3.dat; break; done'
} > $R 2>&1
cat $R
git add $R && git commit -qm "BlueHive CP2K container setup" && git push -q && echo "pushed setup report"
# ---- 1-minute test on a Vermont node (no Slurm): two concurrent 4-rank runs pinned to disjoint cores,
# i.e. the packing used for the labels. Usage: TEST_NODE=bhx0131 bash bluehive/setup_cp2k.sh
[ -n "${TEST_NODE:-}" ] || { echo "set TEST_NODE=<vermont node> to run the container test there"; exit 0; }
T=$PWD/bluehive/cp2k_test
ssh -o BatchMode=yes $TEST_NODE bash -l -s > bluehive/reports/cp2k_test.txt 2>&1 <<REMOTE
echo "# cp2k container test on \$(hostname) \$(date -Is)"
module load apptainer/1.4.1
cd $T
for i in 0 1; do
  mkdir -p run\$i && cp ch4.inp run\$i/
  ( cd run\$i && apptainer exec --cleanenv --env OMP_NUM_THREADS=1 $SIF \
      mpirun -np 4 --bind-to core --cpu-set \$((4*i))-\$((4*i+3)) cp2k.psmp -i ch4.inp -o ch4.out > run.log 2>&1 ) &
done
wait
for i in 0 1; do echo "== run\$i"; grep -E "ENERGY\| Total|PROGRAM ENDED|Analytical stress|ABORT" run\$i/ch4.out; tail -4 run\$i/run.log; done
REMOTE
cat bluehive/reports/cp2k_test.txt
git add bluehive/reports/cp2k_test.txt && git commit -qm "CP2K container test on $TEST_NODE" && git push -q && echo pushed
