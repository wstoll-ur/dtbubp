#!/bin/bash
# Does /software/cp2k/2025.1 (the build that dies with SIGILL on the Broadwell Vermont nodes) run on the
# Slurm 'standard' nodes (newer Xeons)? 15-minute test job; waits for it, then pushes the report.
#   bash bluehive/test_slurm_cp2k.sh [partition]
cd "$(git rev-parse --show-toplevel)" || exit 1
P=${1:-standard}; T=$PWD/bluehive/cp2k_test; OUT=bluehive/reports/test_slurm_cp2k_$P.txt
ls -la /software/cp2k/2025.1/bin/cp2k.psmp || { echo "/software/cp2k/2025.1 not visible from bluehive3"; }
cat > $T/slurm_test.sh <<JOB
#!/bin/bash
#SBATCH --job-name=dtb_cp2ktest --partition=$P --nodes=1 --ntasks=8 --cpus-per-task=1 --mem=16G --time=00:15:00
#SBATCH --output=$PWD/$OUT
echo "# \$(hostname) \$(date -Is)"; lscpu | grep -E "Model name|^CPU\(s\)"; grep -o -w -E "avx2|avx512f" /proc/cpuinfo | sort | uniq -c
export PATH=/software/cp2k/2025.1/bin:/software/cp2k/src/tools/toolchain/install/openmpi-5.0.7/bin:\$PATH
export LD_LIBRARY_PATH=/software/cp2k/src/tools/toolchain/install/openmpi-5.0.7/lib:\${LD_LIBRARY_PATH:-}
module load gcc/14.2.0 2>&1 | head -2
ldd /software/cp2k/2025.1/bin/cp2k.psmp | grep "not found"
cd $T && for i in 0 1; do mkdir -p srun\$i && cp ch4.inp srun\$i/; done
export OMP_NUM_THREADS=1
( cd srun0 && env -u SLURM_JOBID mpirun -np 4 --bind-to core --cpu-set 0-3 cp2k.psmp -i ch4.inp -o ch4.out > run.log 2>&1 ) &
( cd srun1 && env -u SLURM_JOBID mpirun -np 4 --bind-to core --cpu-set 4-7 cp2k.psmp -i ch4.inp -o ch4.out > run.log 2>&1 ) &
wait
for i in 0 1; do echo "== run\$i"; grep -E "ENERGY\| Total|PROGRAM ENDED|STRESS\| Analytical|ABORT|CP2K +1 " srun\$i/ch4.out; tail -6 srun\$i/run.log; done
grep -m1 -A3 "CP2K| version" srun0/ch4.out; grep -m1 "CP2K| Data directory" srun0/ch4.out
JOB
sbatch --wait $T/slurm_test.sh
cat $OUT
git add $OUT && git commit -qm "CP2K 2025.1 test on Slurm $P" && git push -q && echo pushed
