# Fixed-temperature NPT tests of potential r0 (CP2K-MACE), compared with experiment

Six independent runs: 298, 260, 220, 180, 120, 80 K. Each: 2×2×2 supercell (736 atoms, 16 molecules),
NPT_F (flexible triclinic cell), 1 bar, CSVR thermostat (100 fs), barostat 1 ps, Δt 0.5 fs, 100 ps
(first 20 ps discarded). Start = the fix-deform frame nearest each temperature (thermalised H positions,
volume already near experiment). A watchdog stops a run whose volume leaves 0.80–1.25 × the start.

**How to read the comparison:** the experimental room-T phase (P-1, Z = 2) only exists above ~210 K.
At 180, 120 and 80 K the real crystal has transformed (Z = 12, then the low-T phase), which a 2×2×2
supercell of the room-T cell cannot do. Those runs show the room-T structure held metastably; compare
them with the experimental trend, not as a pass/fail. The 298 / 260 / 220 K runs are the real test.

## Run (Leonardo)

```bash
cd $SCRATCH/DtBuDp/Calculations/04_npt_ramp_78-300K/fixedT_r0
module load python/3.11.7 && source ~/cp2k_torch_env/bin/activate
dtbn() { PYTHONPATH=$SCRATCH/DtBuDp/code python -m dtbubp.cli --config $PWD/config.toml "$@"; }

ls -l ../../03_mace_finetune/models/potential_r0/*-cp2k.pth      # the exported model must exist
dtbn npt-setup --dry-run      # writes T*/md.inp, start.xyz, npt_job.sh
dtbn npt-setup                # submits a 6-task GPU array
```

## Check early (first ~15 min)

```bash
tail -5 dtb_npt_*_0.out                       # the ldd line must show cp2k-mace/install/lib64
tail -3 T298/md-1.cell T298/md-1.ener         # volume (last column) and temperature moving sensibly
grep -m1 -i "step.*time" T298/md.out; grep "MD| Step number" T298/md.out | tail -1
```

## Analyse (any time; uses what has been written so far)

```bash
dtbn npt-analyze              # npt_summary.txt / .json / .png
```

A run killed at the 24 h limit continues from its restart file: just `sbatch npt_job.sh` again
(finished temperatures exit immediately).
