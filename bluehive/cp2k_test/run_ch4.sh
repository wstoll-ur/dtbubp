#!/bin/bash -l
# Runs ON a Vermont node (copied there by build_cp2k.sh test/deploy). Methane PBE-D3(BJ)/TZV2P:
# two concurrent 4-rank runs pinned to disjoint cores (= the label packing) + one serial run.
echo "# CP2K test on $(hostname) $(date -Is)"
source $HOME/Claude/cp2k-2024.1.env
export OMP_NUM_THREADS=1
echo "cp2k: $(which cp2k.psmp)"; echo "mpirun: $(which mpirun)"
ldd $(which cp2k.psmp) | grep "not found" && echo "MISSING LIBRARIES"
cd $HOME/dtbubp_test || exit 1
for i in 0 1 s; do rm -rf trun$i; mkdir -p trun$i; cp ch4.inp trun$i/; done
( cd trun0 && mpirun -np 4 --bind-to core --cpu-set 0-3 cp2k.psmp -i ch4.inp -o ch4.out > run.log 2>&1 < /dev/null ) &
( cd trun1 && mpirun -np 4 --bind-to core --cpu-set 4-7 cp2k.psmp -i ch4.inp -o ch4.out > run.log 2>&1 < /dev/null ) &
wait
( cd truns && cp2k.psmp -i ch4.inp -o ch4.out > run.log 2>&1 < /dev/null )
for i in 0 1 s; do
  echo "== trun$i"
  grep -E "CP2K\| version string|Data directory path|Total number of message passing|ENERGY\| Total|PROGRAM ENDED|ABORT|^ CP2K +1 " trun$i/ch4.out
  grep -A4 "STRESS| Analytical stress tensor" trun$i/ch4.out | head -5
  tail -3 trun$i/run.log
done
