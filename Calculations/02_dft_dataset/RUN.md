# Campaign r0 — DFT labels for the fix-deform frames, then MACE fine-tuning

Everything here is driven by `config.toml` (this folder) and the `dtbubp` code in `../../code`.
Each command appends to `LOG.md` (this folder) automatically; `state.json` keeps the numbers.

## Done locally (2026-10-04)

`select` was run on the Mac (it needs the 750 MB fix-deform dump):
`frame_selection/frames.extxyz` = **2000 fix-deform frames + 200 strained cells**
(train 1772 / valid 224 / test 204), plot in `frame_selection/selection.png`.

## On Leonardo

### 0. Copy (from the Mac, keep the relative layout)

The jobs find each other by relative paths, so Leonardo must have exactly
`$SCRATCH/DtBuDp/code` and `$SCRATCH/DtBuDp/Calculations/02_dft_dataset`.
`$SCRATCH` must be expanded **on Leonardo**, not on the Mac (it is empty on the Mac).
macOS ships `openrsync`, which is fragile against Leonardo's rsync; the tar pipe below avoids it.

```bash
# on the Mac, from Box/Documents/Manuscripts/
tar czf - DtBuDp/code DtBuDp/Calculations/02_dft_dataset \
  | ssh wstoll00@login.leonardo.cineca.it 'cd $SCRATCH && tar xzf - && ls DtBuDp DtBuDp/Calculations'
```

Later updates of the code only (rsync with the path spelled out; `brew install rsync` if openrsync fails again):

```bash
rsync -av DtBuDp/code/ wstoll00@login.leonardo.cineca.it:/leonardo_scratch/large/userexternal/wstoll00/DtBuDp/code/
```

Foundation models (on Leonardo, once):

```bash
mkdir -p DtBuDp/Calculations/03_mace_finetune/foundation && cd $_
cp <9MA>/cp2k/npt_al/old_data/MACE-OFF23_small.model .                                   # same file train1-3 used
```

Shorthand used below (from `DtBuDp/Calculations/02_dft_dataset`):

```bash
module load python/3.11.7 && source ~/cp2k_torch_env/bin/activate   # python 3.11 + numpy; nothing installed
dtb() { PYTHONPATH=$PWD/../../code python -m dtbubp.cli --config config.toml "$@"; }
```

### 1. Smoke test (one frame, one DCGP node, ~1-3 h)

```bash
dtb smoketest            # 3 CP2K runs side by side on one node
dtb smoketest --check    # after it finishes
```

It must report: all five labels present (energy, forces, stress, dipole, polarizability),
analytical vs numerical stress within 0.05 GPa, and analytical vs finite-field polarizability within 2 %.
It also prints the wall time of one label run: use it to adjust `[dft] frames_per_node / chunks` (§3).

If LINRES fails or the polarizability check fails with the NN10 smoothing, set `xc_smoothing = false`
in `config.toml`, rerun the smoke test, and note it in `PROJECT_LOG.md`.

### 2. Label (2200 frames)

```bash
dtb label --dry-run      # look at label/label_job.sh and one label/fNNNN/input.inp
dtb label                # submits the array job (30 tasks x 1 node, 4 frames at a time per node)
dtb status               # finished / started, wall time per frame
dtb label                # re-run any time: finished frames are skipped, the rest is resubmitted
```

Budget: 2200 frames x (label wall time) / 4 per node. At 30 min per frame ≈ 275 node-hours.

### 3. Collect

```bash
dtb collect              # -> parsed/labels.xyz, train.xyz, valid.xyz, test.xyz, labels_overview.png
```

Read `parsed/summary.txt`: the mean DFT static pressure over the fix-deform volumes tells us directly
how far BLYP-D3 is from MACE-OFF23 (which sits at about −0.7 GPa at the experimental cell).

### 4. Train

The dielectric model (dipole + polarizability) is trained **from scratch** (`AtomicDielectricMACE`):
MACE-MDP cannot be fine-tuned with released mace-torch (see PROJECT_LOG, 2026-10-04).

```bash
dtb train both --dry-run
dtb train both
```

The potential job ends by evaluating on the test set (`eval_potential.json/png`) and exporting
`<model>-cp2k.pth` for CP2K-MACE NPT. The dielectric job writes `eval_dielectric.json/png`.
