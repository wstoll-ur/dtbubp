#!/bin/bash
# CP2K for the Broadwell Vermont nodes (RHEL 7.9, glibc 2.17): conda-forge build of CP2K 2024.1 (the version of
# all Leonardo runs) into a NEW, separate env named Claude: /home/$USER/Claude/cp2k-2024.1. Your own
# environments (tera, base, ...) are not touched. The node-local /home on each Vermont node gets a copy at the
# same path, so the env works there unchanged.
#   bash bluehive/build_conda_cp2k.sh "bhx0123 bhx0124 bhx0125"
cd "$(git rev-parse --show-toplevel)" || exit 1
NODES=${1:?node list}
ENV=/home/$USER/Claude/cp2k-2024.1
OUT=bluehive/reports/build_conda_cp2k.txt
{
echo "# $(date -Is) $(hostname)"
module load miniforge3/25.3.0-3
export CONDA_PKGS_DIRS=/scratch/$USER/.conda_pkgs_claude      # package cache out of $HOME
export CONDA_OVERRIDE_GLIBC=2.17                              # only builds that run on RHEL 7 nodes
if [ ! -x $ENV/bin/cp2k.psmp ]; then
  conda create -y -p $ENV -c conda-forge --override-channels "cp2k=2024.1=*openmpi*" 2>&1 | tail -25
fi
ls $ENV/bin | grep -E "cp2k|mpirun"
du -sh $ENV
conda list -p $ENV 2>/dev/null | grep -E "^(cp2k|openmpi|libxc|libint|openblas|libblas|fftw|scalapack|dbcsr|libxsmm) "
for h in $NODES; do
  echo "## $h"
  ssh -o BatchMode=yes $h "mkdir -p /home/$USER/Claude" && rsync -a --delete $ENV/ $h:$ENV/ && echo "copied to $h"
done
H=$(echo $NODES | awk '{print $1}')
echo "## methane test on $H (2 concurrent 4-rank runs)"
ssh -o BatchMode=yes $H "mkdir -p /home/$USER/dtbubp_test" && scp -q bluehive/cp2k_test/ch4.inp $H:/home/$USER/dtbubp_test/
ssh -o BatchMode=yes $H bash -s <<REMOTE
export PATH=$ENV/bin:\$PATH OMP_NUM_THREADS=1
cd /home/$USER/dtbubp_test
for i in 0 1; do mkdir -p crun\$i && cp ch4.inp crun\$i/; done
( cd crun0 && mpirun -np 4 --bind-to core --cpu-set 0-3 cp2k.psmp -i ch4.inp -o ch4.out > run.log 2>&1 ) &
( cd crun1 && mpirun -np 4 --bind-to core --cpu-set 4-7 cp2k.psmp -i ch4.inp -o ch4.out > run.log 2>&1 ) &
wait
for i in 0 1; do echo "== crun\$i"; grep -E "ENERGY\| Total|PROGRAM ENDED|STRESS\| Analytical|ABORT|CP2K +1 " crun\$i/ch4.out; tail -6 crun\$i/run.log; done
grep -m1 "CP2K| version" crun0/ch4.out; grep -m1 "Data directory" crun0/ch4.out
REMOTE
} > $OUT 2>&1
cat $OUT
git add $OUT && git commit -qm "conda CP2K 2024.1 build + Vermont test" && git push -q && echo pushed
