#!/bin/bash
# Compile CP2K 2024.1 (the version of every Leonardo run) for the Broadwell Vermont nodes (RHEL 7.9, AVX2).
# Built from source with the CP2K toolchain (GCC + OpenMPI 4.1.5 + OpenBLAS + ScaLAPACK + FFTW + libxc +
# libint + libxsmm only (optional COSMA/ELPA/SIRIUS/HDF5/... off: not needed for GPW labels), all built locally; no libgrpp = ECP integrals, not needed with GTH) into  /home/$USER/Claude/cp2k-2024.1  on the node's own disk.
# Nothing is installed into your existing environments.
#
#   bash bluehive/build_cp2k.sh fetch            # bluehive3: download source + toolchain packages (nodes may lack internet)
#   bash bluehive/build_cp2k.sh start bhx0123    # copy to the node, start the build there (nohup, ~2-3 h)
#   bash bluehive/build_cp2k.sh log   bhx0123    # tail the build log (and push it to the repo)
#   bash bluehive/build_cp2k.sh test bhx0123     # methane test on the build node, nothing copied
#   bash bluehive/build_cp2k.sh deploy bhx0123 "bhx0124 bhx0125"   # copy the finished build to the other nodes + methane test on all
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
REPO=$PWD
SRC=/scratch/$USER/cp2k_build_downloads
DEST=/home/$USER/Claude
CP2K=$DEST/cp2k-2024.1
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
R=bluehive/reports
cmd=${1:-}; shift || true

case "$cmd" in
fetch)
  mkdir -p $SRC/pkgs && cd $SRC
  [ -s cp2k-2024.1.tar.bz2 ] || wget -q https://github.com/cp2k/cp2k/releases/download/v2024.1/cp2k-2024.1.tar.bz2
  # the Vermont nodes have no bzip2: repack as gzip here
  [ -s cp2k-2024.1.tar.gz ] || { bzip2 -dc cp2k-2024.1.tar.bz2 | gzip -1 > cp2k-2024.1.tar.gz; }
  cd pkgs
  for f in cmake-3.28.1-linux-x86_64.sh openmpi-4.1.5.tar.gz OpenBLAS-0.3.25.tar.gz fftw-3.3.10.tar.gz \
           libxc-6.2.2.tar.gz libint-v2.6.0-cp2k-lmax-5.tgz libxsmm-1.17.tar.gz \
           scalapack-2.2.1.tgz; do
    [ -s $f ] || wget -q https://www.cp2k.org/static/downloads/$f || echo "FAILED $f"
  done
  ls -la $SRC $SRC/pkgs
  ;;
start)
  H=$1
  $SSH $H "mkdir -p $DEST" && rsync -a --exclude '*.bz2' $SRC/ $H:$DEST/downloads/ || exit 1
  cat > /tmp/build_cp2k_remote_$USER.sh <<REMOTE
#!/bin/bash -l
# no set -e: the node's module function returns non-zero on harmless warnings (unalias sudo)
echo "build start \$(date) on \$(hostname), \$(nproc) cores"
cd $DEST
echo "tools on this node:"; for x in gzip bzip2 unzip make patch perl python3 wget m4 git; do printf "  %-8s %s\\n" \$x "\$(command -v \$x || echo MISSING)"; done
for x in gzip make patch perl; do command -v \$x >/dev/null || { echo "MISSING required tool \$x"; exit 1; }; done
[ -d cp2k-2024.1 ] || tar xzf downloads/cp2k-2024.1.tar.gz || { echo 'untar failed'; exit 1; }
cd cp2k-2024.1/tools/toolchain || exit 1
mkdir -p build && cp -n $DEST/downloads/pkgs/* build/
module purge || true
echo "gcc modules:"; module avail gcc 2>&1 | grep -o "gcc/[0-9][^ ]*" | sort -u
# prefer GCC 12/13 (CP2K 2024.1 and its libraries predate GCC 14); fall back to 11, then 14
G=\$(module avail gcc 2>&1 | grep -o "gcc/1[23][^ ]*" | sort -V | tail -1)
[ -n "\$G" ] || G=\$(module avail gcc 2>&1 | grep -o "gcc/11[^ ]*" | sort -V | tail -1)
[ -n "\$G" ] || G=gcc/14.2.0/b1
module load \$G; echo "using \$G: \$(gcc --version | head -1)"
# libint's Fortran interface is generated with the system 'python' (2.7): do NOT load the python3 module before
# the toolchain (it breaks python 2 -> libint_f.mod missing; 1st build attempt). A libint install without the
# Fortran module is removed so the toolchain rebuilds it.
# python3 for the toolchain's arch-file generation and CP2K's fypp, WITHOUT the python3 module's environment
# (which breaks the system python 2 that libint needs): a wrapper script in $DEST/bin.
mkdir -p $DEST/bin
printf '#!/bin/sh\nLD_LIBRARY_PATH=/software/python3/3.7.1/lib:\$LD_LIBRARY_PATH exec /software/python3/3.7.1/bin/python3 "\$@"\n' > $DEST/bin/python3
chmod +x $DEST/bin/python3
export PATH=$DEST/bin:\$PATH
echo "python3 wrapper: \$(python3 --version 2>&1); python: \$(python --version 2>&1)"
LI=install/libint-v2.6.0-cp2k-lmax-5
if [ -d \$LI ] && [ -z "\$(find \$LI -name 'libint_f.mod' 2>/dev/null)" ]; then echo "removing incomplete \$LI"; rm -rf \$LI; fi
./install_cp2k_toolchain.sh -j \$(nproc) --target-cpu=haswell --mpi-mode=openmpi --with-gcc=system \\
    --with-openmpi=install --with-openblas=install --with-cmake=install --with-libgrpp=no \\
    --with-cosma=no --with-elpa=no --with-sirius=no --with-hdf5=no --with-gsl=no --with-spglib=no \\
    --with-libvdwxc=no --with-spfft=no --with-spla=no --with-libvori=no --with-plumed=no --with-quip=no \\
    --with-pexsi=no --with-superlu=no --with-ptscotch=no --with-libtorch=no \\
    || { echo 'TOOLCHAIN FAILED'; exit 1; }
[ -n "\$(find \$LI -name 'libint_f.mod')" ] || { echo 'libint Fortran module missing (see build/libint-*/make.log)'; exit 1; }
cp install/arch/local.psmp ../../arch/ || exit 1
source install/setup
echo "python3: \$(command -v python3)"
cd ../..
make -j \$(nproc) ARCH=local VERSION=psmp || { echo 'CP2K MAKE FAILED'; exit 1; }
[ -x exe/local/cp2k.psmp ] || { echo 'no cp2k.psmp'; exit 1; }
ls -la exe/local/
echo "module load \$G" > $DEST/cp2k-2024.1.env
echo "source $CP2K/tools/toolchain/install/setup" >> $DEST/cp2k-2024.1.env
echo "export PATH=$CP2K/exe/local:\\\$PATH" >> $DEST/cp2k-2024.1.env
echo "export CP2K_DATA_DIR=$CP2K/data" >> $DEST/cp2k-2024.1.env
echo "BUILD OK \$(date)"
REMOTE
  scp -q /tmp/build_cp2k_remote_$USER.sh $H:$DEST/build_cp2k_remote.sh
  $SSH $H "cd $DEST && setsid nohup bash -l build_cp2k_remote.sh > build.log 2>&1 < /dev/null & echo started pid \$!"
  echo "log: bash bluehive/build_cp2k.sh log $H"
  ;;
log)
  H=$1
  $SSH $H "tail -n 40 $DEST/build.log; echo; ls $CP2K/exe/local 2>/dev/null; grep -h -iE 'error|failed' $DEST/build.log | tail -15" > $R/build_cp2k_log.txt 2>&1
  cat $R/build_cp2k_log.txt
  git add $R/build_cp2k_log.txt && git commit -qm "CP2K build log ($H)" && git push -q && echo pushed
  ;;
test)
  # methane PBE-D3(BJ)/TZV2P on one node: two concurrent 4-rank runs on disjoint cores (the label packing),
  # plus a 1-rank run to check the serial path. No copying. Usage: bash bluehive/build_cp2k.sh test bhx0123
  H=$1; OUT=$R/build_cp2k_test_$H.txt
  $SSH $H "mkdir -p /home/$USER/dtbubp_test" && scp -q bluehive/cp2k_test/ch4.inp $H:/home/$USER/dtbubp_test/
  $SSH $H bash -l -s > $OUT 2>&1 <<TEST
echo "# CP2K test on \$(hostname) \$(date -Is)"
source $DEST/cp2k-2024.1.env; export OMP_NUM_THREADS=1
echo "cp2k: \$(which cp2k.psmp)  mpirun: \$(which mpirun)"; ldd \$(which cp2k.psmp) | grep "not found"
cd /home/$USER/dtbubp_test
for i in 0 1 s; do rm -rf trun\$i; mkdir -p trun\$i && cp ch4.inp trun\$i/; done
( cd trun0 && mpirun -np 4 --bind-to core --cpu-set 0-3 cp2k.psmp -i ch4.inp -o ch4.out > run.log 2>&1 < /dev/null ) &
( cd trun1 && mpirun -np 4 --bind-to core --cpu-set 4-7 cp2k.psmp -i ch4.inp -o ch4.out > run.log 2>&1 < /dev/null ) &
wait
( cd truns && cp2k.psmp -i ch4.inp -o ch4.out > run.log 2>&1 < /dev/null )
for i in 0 1 s; do echo "== trun\$i"; grep -E "CP2K\| version string|Data directory path|Total number of message passing|ENERGY\| Total|PROGRAM ENDED|ABORT|CP2K +1 " trun\$i/ch4.out; grep -A4 "STRESS| Analytical stress tensor" trun\$i/ch4.out | head -5; tail -3 trun\$i/run.log; done
TEST
  cat $OUT
  git add $OUT && git commit -qm "CP2K build test on $H" && git push -q && echo pushed
  ;;
deploy)
  H=$1; OTHERS=$2
  OUT=$R/build_cp2k_deploy.txt
  {
  # node -> bluehive3 scratch -> other nodes (the nodes may not have ssh keys for each other)
  STAGE=/scratch/$USER/cp2k_build_copy; mkdir -p $STAGE
  rsync -a --exclude 'tools/toolchain/build' --exclude 'obj' --exclude 'lib/local' -e "$SSH" $H:$CP2K $H:$DEST/cp2k-2024.1.env $STAGE/ && echo "copied from $H"
  for h in $OTHERS; do
    echo "## copy -> $h"
    $SSH $h "mkdir -p $DEST" && rsync -a -e "$SSH" $STAGE/ $h:$DEST/ && echo ok
  done
  for h in $H $OTHERS; do
    echo "## methane test on $h"
    $SSH $h "mkdir -p /home/$USER/dtbubp_test" && scp -q bluehive/cp2k_test/ch4.inp $h:/home/$USER/dtbubp_test/
    $SSH $h bash -l -s <<TEST
source $DEST/cp2k-2024.1.env; export OMP_NUM_THREADS=1
cd /home/$USER/dtbubp_test
for i in 0 1; do mkdir -p brun\$i && cp ch4.inp brun\$i/; done
( cd brun0 && mpirun -np 4 --bind-to core --cpu-set 0-3 cp2k.psmp -i ch4.inp -o ch4.out > run.log 2>&1 < /dev/null ) &
( cd brun1 && mpirun -np 4 --bind-to core --cpu-set 4-7 cp2k.psmp -i ch4.inp -o ch4.out > run.log 2>&1 < /dev/null ) &
wait
for i in 0 1; do echo "== brun\$i"; grep -E "ENERGY\| Total|PROGRAM ENDED|Analytical stress|ABORT|CP2K +1 " brun\$i/ch4.out; tail -4 brun\$i/run.log; done
TEST
  done
  } > $OUT 2>&1
  cat $OUT
  git add $OUT && git commit -qm "CP2K build deployed + tested" && git push -q && echo pushed
  ;;
*) sed -n 2,11p "$0"; exit 1 ;;
esac
