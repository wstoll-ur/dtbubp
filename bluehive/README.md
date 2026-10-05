# DtBuDp on BlueHive (University of Rochester CIRC)

Moved here from Leonardo on 2026-10-05: Leonardo CPU hours are no longer available (its GPUs still are, so MACE
training stays there). The project travels through the private GitHub repo `wstoll-ur/dtbubp`:

```
Box (Mac)  --git push-->  GitHub  --git pull-->  BlueHive  --(bluehive/push_reports.sh)-->  GitHub  -->  Box
```

Only code, configs, notes, the 2200 selected frames (`02_dft_dataset/frame_selection/frames.extxyz`) and small
reports are in git (see `.gitignore`). Per-frame DFT inputs/outputs stay on BlueHive.

## 0. One-time setup

```bash
# deploy key (write access) added to the GitHub repo; ~/.ssh/config has "Host github-dtbubp"
cd /scratch/$USER            # or wherever there is space (probe.sh reports quotas)
git clone github-dtbubp:wstoll-ur/dtbubp.git DtBuDp
cd DtBuDp
git config user.name  "Will Stoll (BlueHive)"
git config user.email william.stoll@rochester.edu
bash bluehive/probe.sh       # read-only discovery -> bluehive/reports/probe.txt, pushed
```

## 1. Common environment for the commands below

```bash
cd /scratch/$USER/DtBuDp && git pull
export PYTHONPATH=$PWD/code:$PYTHONPATH
alias dtb='python3 -m dtbubp.cli'
R1=Calculations/02_dft_dataset_r1_pbe-d3bj_tzv2p/config.toml
BENCH=Calculations/01_dft_benchmark/config_bluehive.toml
```
dtbubp needs only numpy. Configs are TOML; on a python without a TOML parser (< 3.11, no tomli) it reads the
`.<config>.json` copy committed next to each config (refused if it does not match the TOML).

## 2. Bench level 12 (PBE-D3(BJ)+C9 / DZVP-MOLOPT-SR), one node

```bash
dtb --config $BENCH bench-setup --only 12_pbe_d3bj_c9_dzvpsr --dry-run   # inspect cellopt/12_*/input.inp + bench_job.sh
dtb --config $BENCH bench-setup --only 12_pbe_d3bj_c9_dzvpsr
dtb --config $BENCH bench-analyze          # any time; table includes the Leonardo levels
```

## 3. r1 labels: the same 2200 frames at PBE-D3(BJ) / TZV2P-MOLOPT (bench level 06)

```bash
dtb --config $R1 smoketest --dry-run       # inspect smoketest/*/input.inp + smoke_job.sh
dtb --config $R1 smoketest                 # one node: label deck / numerical stress / finite-field polarizability
dtb --config $R1 smoketest --check         # PASS -> timing per frame -> set chunks/frames_per_node/time_limit
dtb --config $R1 label                     # array job, one node per chunk; re-run = continue (skips finished)
dtb --config $R1 status
dtb --config $R1 collect                   # -> parsed/{labels,train,valid,test}.xyz (copy to Leonardo for training)
```

## 4. Reporting back

```bash
bash bluehive/push_reports.sh "what happened"   # status, LOG.md, bench summary, squeue, smoke-test tails
```
