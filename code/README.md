# dtbubp

DFT labels and MACE fine-tuning for crystalline 4,4′-di-*tert*-butylbiphenyl (DtBuBP).
Part of the DtBuDp project; the full record of what was done and why is `../PROJECT_LOG.md`.

**What it does (campaign r0):** takes frames of the MACE-OFF23 fix-deform cooling run (300 → 53 K),
labels them with CP2K — energy, forces, stress, Berry-phase dipole and periodic polarizability in one
run per frame — and fine-tunes two MACE models:

| model | targets | start | used for |
|---|---|---|---|
| potential | E, F, stress | MACE-OFF23-small | NPT in CP2K-MACE (exported with `tools/create_cp2k_model.py`) |
| dielectric | dipole, polarizability | from scratch (`AtomicDielectricMACE`; MACE-MDP cannot be fine-tuned in released MACE) | α(t) along the NPT trajectories → low-frequency Raman |

## Commands

```
python -m dtbubp.cli --config CONFIG select            # local (where the 750 MB dump is): frames.extxyz
python -m dtbubp.cli --config CONFIG smoketest         # Leonardo: one frame, 3 CP2K checks (one node)
python -m dtbubp.cli --config CONFIG smoketest --check
python -m dtbubp.cli --config CONFIG label             # all frames, packed array job (resubmit = continue)
python -m dtbubp.cli --config CONFIG status
python -m dtbubp.cli --config CONFIG collect           # -> parsed/{labels,train,valid,test}.xyz
python -m dtbubp.cli --config CONFIG train both        # potential + dielectric (GPU jobs, evaluate themselves)
```
`--dry-run` writes every input and job script without submitting. Nothing needs installing:
put this folder on `PYTHONPATH` (the training jobs do it themselves). Needs numpy (+ tomli on python 3.10);
`evaluate` needs ase and mace-torch.

## Level of theory (from 9MA, unchanged)

CP2K 2024.1 module, BLYP-D3 (zero damping), DZVP-MOLOPT-SR-GTH, GTH-BLYP-q4/q1, 600/60 Ry,
NN10 XC smoothing, EPS_SCF 1e-8, EPS_DEFAULT 1e-16, Γ point. Added: `&PRINT &MOMENTS PERIODIC T`
(Berry-phase dipole) and `&PROPERTIES &LINRES &POLAR` (DFPT polarizability, Berry-phase dipole operator).
LINRES runs post-SCF inside `ENERGY_FORCE` (CP2K 2024.1 `qs_energies_properties`), so one run gives all labels.

## Label conventions (`parsed/*.xyz`)

| key | unit | note |
|---|---|---|
| `REF_energy` | eV | |
| `REF_forces` | eV/Å | per atom |
| `REF_stress` | eV/Å³ | ASE sign (CP2K's printed stress negated), Voigt xx yy zz yz xz xy |
| `REF_dipole` | e·Å | Berry phase, folded to the branch nearest zero (molecules are made whole first) |
| `REF_polarizability` | e·Å²/V | 3×3 row-major, symmetrised; MACE-MDP / SPICE-α units (so MDP can serve as a baseline). ×14.3996 → Å³ |
| `polarizability_A3` | Å³ | same tensor, for reading |
| `pressure_GPa` | GPa | DFT static pressure, CP2K sign (positive = wants to expand) |
| `split`, `T_target`, `step`, `time_ns`, `config_type` | | provenance |

Train/valid/test are assigned by 0.2 ns time blocks of the ramp (not random frames).

## Layout

```
dtbubp/          units, io (LAMMPS dump, extxyz), geometry (molecules, strains, dipole folding),
                 cp2k (decks + parsers), select, label, train, evaluate, slurm, project, cli
templates/       label.inp (the CP2K deck)
tools/           create_cp2k_model.py (Will's MACE -> CP2K exporter, copied from 9MA/cp2k)
tests/           pytest, synthetic CP2K outputs: python -m pytest -q tests
config_template.toml
```

Never commit trajectories, DFT outputs, models or CIFs (`.gitignore`).
