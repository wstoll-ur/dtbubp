# DtBuDp — 4,4′-di-tert-butylbiphenyl (C₂₀H₂₆)

Trimorphic molecular crystal. Goal: fine-tune a MACE potential (energies, forces, stresses) and a MACE dielectric model (dipoles, polarizabilities) on DFT data, run NPT from 78 to 300 K, improve the model by active learning, and compute the low-frequency Raman spectrum from the polarizability along the trajectory.

**Read [`PROJECT_LOG.md`](PROJECT_LOG.md) first** — it is the detailed lab notebook (what exists, what was found, the plan, the open decisions, and a dated changelog).

## Layout

| Folder | Contents |
|---|---|
| `Data/crystal_structures/` | Experimental CIFs (298 K now; 160 K and low-T to be added) |
| `Calculations/00_reference_structures/` | Simulation cells built from the CIFs |
| `Calculations/01_dft_benchmark/` | CP2K: convergence, functional/dispersion, basis/GTH, torsions, polarizability, Γ-point Raman |
| `Calculations/02_dft_dataset/` | Frame selection, DFT inputs/outputs, parsed training data |
| `Calculations/03_mace_finetune/` | Datasets, configs, models, validation |
| `Calculations/04_npt_ramp_78-300K/` | Production NPT heating/cooling |
| `Calculations/05_active_learning/` | AL iterations |
| `Calculations/06_raman/` | Polarizability time series and spectra |
| `Calculations/legacy/` | Pre-existing runs (OPLS-AA, MACE-OFF23-small fix-deform, UMA fix-deform), untouched |
| `code/` | Future GitHub repository |
| `Figures/ Tables/ Presentations/ Videos/ Structure_fitting/` | Manuscript material |

Electronic structure: CP2K only. MD: CP2K-MACE (FIST + MACE), the same build as the sister project `../9MA` (whose `cp2k/npt_al` stress active-learning loop is the template here).
Compute: CINECA Leonardo. Software environment: a dedicated virtual environment named `Claude`.
