#!/bin/bash
# Can the three Vermont nodes run ONE MPI job together? Read-only checks + a tiny MPI run + a timing test.
#   bash bluehive/multinode.sh check   "bhx0123 bhx0124 bhx0125"
# Checks: node-to-node ssh without password (OpenMPI starts the remote ranks with ssh from the first node),
# network interfaces/InfiniBand, a 3-node 'hostname' with our OpenMPI, then the methane test on 3x4 ranks.
cd "$(git rev-parse --show-toplevel)" || exit 1
NODES=${2:-"bhx0123 bhx0124 bhx0125"}
H=$(echo $NODES | awk '{print $1}')
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
OMPI=/home/$USER/Claude/cp2k-2024.1/tools/toolchain/install/openmpi-4.1.5
OUT=bluehive/reports/multinode_check.txt
HL=$(echo $NODES | sed 's/ /:4,/g'):4
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
         cd /home/$USER/dtbubp_test/multi; grep -E 'Total number of message passing|ENERGY\\| Total|PROGRAM ENDED|^ CP2K +1 ' ch4.out; echo '-- ABORT section:'; grep -B3 -A10 ABORT ch4.out | head -30; echo '-- end of ch4.out:'; tail -25 ch4.out; echo '-- run.log:'; tail -15 run.log" < /dev/null
} > $OUT 2>&1
cat $OUT
git add $OUT && git commit -qm "multinode check" && git push -q && echo pushed
