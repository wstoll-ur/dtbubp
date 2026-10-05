#!/bin/bash
# Commit + push small result files from BlueHive so Claude can read them (never DFT outputs).
#   bash bluehive/push_reports.sh "message"
cd "$(git rev-parse --show-toplevel)" || exit 1
export PYTHONPATH=$PWD/code:$PYTHONPATH
R=bluehive/reports; mkdir -p $R
C=Calculations
for d in $C/02_dft_dataset_r*/; do
  r=$(basename $d | sed 's/02_dft_dataset_//; s/_.*//')          # r1, r2, ...
  [ -d $d/label ] && python3 -m dtbubp.cli --config $d/config.toml status > $R/${r}_status.txt 2>&1
  cp $d/LOG.md $R/${r}_LOG.md 2>/dev/null
  cp $d/state.json $R/${r}_state.json 2>/dev/null
done
cp $C/01_dft_benchmark/cellopt/bench_summary.txt $R/bench_summary.txt 2>/dev/null
cp $C/01_dft_benchmark/LOG.md $R/bench_LOG.md 2>/dev/null
squeue -u $USER > $R/squeue.txt 2>&1
for f in $C/02_dft_dataset_r*/smoketest/*/output.out; do
  [ -f "$f" ] && { echo "== $f"; grep -E "PROGRAM (STARTED|ENDED)|SCF run converged|NOT converged|ABORT|ERROR|CP2K   *1 " "$f" | tail -8; tail -30 "$f"; } 
done > $R/smoketest_tails.txt 2>&1
git add $R && git commit -qm "${1:-BlueHive reports $(date +%F_%H%M)}" && git push -q && echo pushed
