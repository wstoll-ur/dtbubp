#!/bin/bash
# Vermont nodes: no shared file system, no Slurm, own module tree with cp2k/2025.1.
# Inspect that module on one node and run the 1-minute methane test there (inputs copied over by scp,
# run in /home/$USER/dtbubp_test on the node, outputs copied back). Read-only for your environments.
#   bash bluehive/test_node_cp2k.sh bhx0124
cd "$(git rev-parse --show-toplevel)" || exit 1
H=${1:?node}
OUT=bluehive/reports/test_node_cp2k_$H.txt
RD=/home/$USER/dtbubp_test
ssh -o BatchMode=yes $H "mkdir -p $RD" && scp -q bluehive/cp2k_test/ch4.inp $H:$RD/
ssh -o BatchMode=yes $H bash -l -s > $OUT 2>&1 <<REMOTE
echo "# cp2k module test on \$(hostname) \$(date -Is)"; cat /etc/redhat-release
df -h /home/$USER /local_scratch | tail -2
echo "## module show"; module show cp2k/2025.1 2>&1
module load cp2k/2025.1 2>&1; echo "## loaded:"; module list 2>&1
for x in cp2k.psmp cp2k.popt cp2k.ssmp cp2k.sopt cp2k mpirun mpiexec; do echo "\$x -> \$(which \$x 2>&1)"; done
mpirun --version 2>&1 | head -3
EXE=\$(which cp2k.psmp 2>/dev/null || which cp2k.popt)
OMP_NUM_THREADS=1 \$EXE --version 2>&1 | head -8
echo "CP2K_DATA_DIR=\$CP2K_DATA_DIR"
for d in \$CP2K_DATA_DIR \$(dirname \$(dirname \$EXE))/share/cp2k/data \$(dirname \$(dirname \$EXE))/data; do
  [ -f \$d/BASIS_MOLOPT ] || continue; echo "data: \$d"
  grep -E "^ *(C|H) +(TZV2P-MOLOPT-GTH|DZVP-MOLOPT-SR-GTH)( |\$)" \$d/BASIS_MOLOPT
  grep -E "^ *(C|H) +GTH-PBE-q(1|4)( |\$)" \$d/POTENTIAL; ls \$d/dftd3.dat; break; done
echo "## ldd"; ldd \$EXE | grep -iE "mpi|blas|lapack|scalapack|mkl|xc|fftw" | head -12
echo "## test: 2 concurrent 4-rank runs"
cd $RD
for i in 0 1; do mkdir -p run\$i; cp ch4.inp run\$i/; done
export OMP_NUM_THREADS=1
( cd run0 && mpirun -np 4 \$EXE -i ch4.inp -o ch4.out > run.log 2>&1 < /dev/null ) &
( cd run1 && mpirun -np 4 \$EXE -i ch4.inp -o ch4.out > run.log 2>&1 < /dev/null ) &
wait
for i in 0 1; do echo "== run\$i"; grep -E "ENERGY\| Total|PROGRAM ENDED|STRESS\| Analytical|ABORT|CP2K +1 " run\$i/ch4.out; tail -5 run\$i/run.log; done
REMOTE
cat $OUT
git add $OUT && git commit -qm "CP2K module test on $H" && git push -q && echo pushed
