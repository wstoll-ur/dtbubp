#!/bin/bash
# Can the three Vermont nodes run ONE MPI job together? Read-only checks + a tiny MPI run + a timing test.
#   bash bluehive/multinode.sh check   "bhx0123 bhx0124 bhx0125"
# Checks: node-to-node ssh without password (OpenMPI starts the remote ranks with ssh from the first node),
# network interfaces/InfiniBand, a 3-node 'hostname' with our OpenMPI, then the methane test on 3x4 ranks.
#   bash bluehive/multinode.sh timing  "bhx0123 bhx0124 bhx0125"
# timing: the REAL level-12 system (92 atoms, PBE-D3(BJ)+C9/DZVP-SR, the bench deck) as one ENERGY_FORCE with
# analytical stress = one cell-opt step: 24 ranks on the first node alone, then 72 ranks on all three.
# Needs cellopt/12_pbe_d3bj_c9_dzvpsr/{input.inp,coord.xyz} (bench-setup --dry-run).
cd "$(git rev-parse --show-toplevel)" || exit 1
MODE=${1:-check}
NODES=${2:-"bhx0123 bhx0124 bhx0125"}
H=$(echo $NODES | awk '{print $1}')
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
OMPI=/home/$USER/Claude/cp2k-2024.1/tools/toolchain/install/openmpi-4.1.5
OUT=bluehive/reports/multinode_check.txt
HL=$(echo $NODES | sed 's/ /:4,/g'):4
if [ "$MODE" = timing ]; then
  L=Calculations/01_dft_benchmark/cellopt/12_pbe_d3bj_c9_dzvpsr
  [ -f $L/input.inp ] || { echo "run bench-setup --only 12_pbe_d3bj_c9_dzvpsr --dry-run first"; exit 1; }
  OUT=bluehive/reports/multinode_timing.txt
  T=/home/$USER/dtbubp_test/timing12
  N=$(echo $NODES | wc -w); H24=$(echo $NODES | sed 's/ /:24,/g'):24
  for h in $NODES; do
    $SSH $h "mkdir -p $T/one $T/all" < /dev/null
    for d in one all; do
      sed -e 's/RUN_TYPE CELL_OPT/RUN_TYPE ENERGY_FORCE/' $L/input.inp | $SSH $h "cat > $T/$d/input.inp"
      scp -q $L/coord.xyz $h:$T/$d/
    done
  done
  MP="mpirun --prefix $OMPI --map-by core --bind-to core -x PATH -x LD_LIBRARY_PATH -x OMP_NUM_THREADS --mca btl_tcp_if_include 192.168.18.0/24 --mca oob_tcp_if_include 192.168.18.0/24"
  {
  echo "# level-12 single point timing $(date -Is): 24 ranks on $H vs $((24*N)) ranks on $NODES"
  $SSH $H "bash -l -c 'source /home/$USER/Claude/cp2k-2024.1.env; export OMP_NUM_THREADS=1
    cd $T/one && $MP -np 24 --host $H:24 cp2k.psmp -i input.inp -o output.out > run.log 2>&1 < /dev/null
    cd $T/all && $MP -np $((24*N)) --host $H24 cp2k.psmp -i input.inp -o output.out > run.log 2>&1 < /dev/null'" < /dev/null
  for d in one all; do
    echo "== $d"
    $SSH $H "cd $T/$d; grep -E 'Total number of message passing|SCF run converged in|Total FORCE_EVAL|Analytical stress|PROGRAM ENDED|^ CP2K +1 |ABORT' output.out; tail -3 run.log" < /dev/null
  done
  } > $OUT 2>&1
  cat $OUT
  git add $OUT && git commit -qm "multinode timing (level 12)" && git push -q && echo pushed
  exit 0
fi
{
echo "# multinode check $(date -Is); first node $H; nodes $NODES"
for h in $NODES; do
  echo "## $h: interfaces / InfiniBand"
  $SSH $h "ip -br addr 2>/dev/null | grep -v '^lo' ; ls /sys/class/infiniband 2>/dev/null && echo IB_PRESENT || echo no_IB; \
           for i in \$(ls /sys/class/net | grep -v lo); do echo \"\$i speed \$(cat /sys/class/net/\$i/speed 2>/dev/null) Mb/s\"; done" < /dev/null
done
echo "## ssh from $H to the others (must print hostnames without asking for a password)"
for h in $NODES; do $SSH $H "ssh -o BatchMode=yes -o ConnectTimeout=10 $h hostname" < /dev/null; done
echo "## 3-node hostname with our OpenMPI"
$SSH $H "export PATH=$OMPI/bin:\$PATH; timeout 60 mpirun --prefix $OMPI -np 3 --host $(echo $NODES | tr ' ' ',') hostname" < /dev/null
echo "## methane test, 12 ranks on 3 nodes"
for h in $NODES; do $SSH $h "mkdir -p /home/$USER/dtbubp_test/multi" < /dev/null && scp -q bluehive/cp2k_test/ch4.inp $h:/home/$USER/dtbubp_test/multi/; done
$SSH $H "bash -l -c 'source /home/$USER/Claude/cp2k-2024.1.env; export OMP_NUM_THREADS=1; cd /home/$USER/dtbubp_test/multi && \
         time timeout 600 mpirun --prefix $OMPI -np 12 --host $HL --map-by core --bind-to core -x PATH -x LD_LIBRARY_PATH -x OMP_NUM_THREADS \
         --mca btl_tcp_if_include 192.168.18.0/24 --mca oob_tcp_if_include 192.168.18.0/24 cp2k.psmp -i ch4.inp -o ch4.out > run.log 2>&1 < /dev/null'; \
         cd /home/$USER/dtbubp_test/multi; grep -E 'Total number of message passing|Total FORCE_EVAL|PROGRAM ENDED|^ CP2K +1 ' ch4.out; echo '-- ABORT section:'; grep -B3 -A10 ABORT ch4.out | head -30; echo '-- end of ch4.out:'; tail -25 ch4.out; echo '-- run.log:'; tail -15 run.log" < /dev/null
} > $OUT 2>&1
cat $OUT
git add $OUT && git commit -qm "multinode check" && git push -q && echo pushed
