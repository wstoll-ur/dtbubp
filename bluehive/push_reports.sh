#!/bin/bash
# Commit + push small result files from BlueHive so Claude can read them (never DFT outputs).
#   bash bluehive/push_reports.sh "message"
cd "$(git rev-parse --show-toplevel)" || exit 1
export PYTHONPATH=$PWD/code:$PYTHONPATH
R=bluehive/reports; mkdir -p $R
C=Calculations
for cfg in $C/02_dft_dataset_r1_pbe-d3bj_tzv2p/config.toml; do
  [ -d "$(dirname $cfg)/label" ] && python3 -m dtbubp.cli --config $cfg status > $R/r1_status.txt 2>&1
done
cp $C/02_dft_dataset_r1_pbe-d3bj_tzv2p/LOG.md $R/r1_LOG.md 2>/dev/null
cp $C/02_dft_dataset_r1_pbe-d3bj_tzv2p/state.json $R/r1_state.json 2>/dev/null
cp $C/01_dft_benchmark/cellopt/bench_summary.txt $R/bench_summary.txt 2>/dev/null
cp $C/01_dft_benchmark/LOG.md $R/bench_LOG.md 2>/dev/null
squeue -u $USER > $R/squeue.txt 2>&1
for f in $C/02_dft_dataset_r1_pbe-d3bj_tzv2p/smoketest/*/output.out; do
  [ -f "$f" ] && { echo "== $f"; grep -E "PROGRAM (STARTED|ENDED)|SCF run converged|NOT converged|ABORT|ERROR|CP2K   *1 " "$f" | tail -8; tail -30 "$f"; } 
done > $R/smoketest_tails.txt 2>&1
git add $R && git commit -qm "${1:-BlueHive reports $(date +%F_%H%M)}" && git push -q && echo pushed
