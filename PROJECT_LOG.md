# DtBuBP — MACE fine-tuning, NPT (78 → 300 K) and low-frequency Raman

**Project lab notebook.** This file is the single source of truth for what has been done, why, with which settings, and where the files are. Every calculation, decision, and file move gets an entry here. Newest log entries go at the bottom of §9 (Changelog).

| | |
|---|---|
| System | 4,4′-di-*tert*-butylbiphenyl (DtBuBP; folder name "DtBuDp"), C₂₀H₂₆ |
| Owner | Will Stoll (william.stoll@rochester.edu) |
| Root folder | `Box/Documents/Manuscripts/DtBuDp/` |
| Electronic structure | **CP2K only** (decision 2026-10-04 — no Gaussian) |
| MLIP / MD engine | MACE (fine-tuned), run inside **CP2K (FIST + MACE via libtorch)** for NPT |
| Compute | CINECA **Leonardo** A100 GPUs (MACE MD/training); **BlueHive** (U. Rochester CIRC) **Vermont nodes bhx0123–0125** (outside Slurm) for DFT from 2026-10-05, with CP2K 2024.1 compiled there — Leonardo CPU hours ended. Code/config travel via the private GitHub repo `wstoll-ur/dtbubp` (§6.5, `bluehive/README.md`) |
| Sister project | `Box/Documents/Manuscripts/9MA/` — same toolchain, already debugged (§4.4) |
| Log started | 2026-10-04 |
| Status | r0 labels + potential r0 done; NPT collapse traced to BLYP-D3/DZVP (EOS); benchmark r1 → PBE-D3(BJ)/TZV2P (−6 %). **2026-10-05: moved DFT to BlueHive; CP2K 2024.1 built on the Vermont nodes; next: bench level 12 (1 vs 3 nodes timing), then r1 relabel of the 2200 frames.** |

---

## Table of contents

1. [Goal and overall strategy](#1-goal-and-overall-strategy)
2. [The system: what we know experimentally](#2-the-system-what-we-know-experimentally)
3. [Folder organization (and every file move made)](#3-folder-organization)
4. [Existing work, audited (this folder and the 9MA project)](#4-existing-work-audited)
5. [The plan, stage by stage](#5-the-plan-stage-by-stage)
6. [Computing environment (Leonardo) and conventions](#6-computing-environment-and-conventions)
7. [Issues found during the audit](#7-issues-found-during-the-audit)
8. [Open questions and decisions needed](#8-open-questions-and-decisions-needed)
9. [Changelog](#9-changelog)
10. [References](#10-references)

---

## 1. Goal and overall strategy

**Scientific goal.** Simulate crystalline DtBuBP under NPT conditions from 78 K (liquid N₂) to 300 K with a machine-learned interatomic potential accurate enough to (i) reproduce the experimental V(T) curve, including the phase transitions of this trimorphic system, (ii) describe *tert*-butyl rotor dynamics and the biphenyl twist correctly, and (iii) produce a low-frequency (lattice/phonon region, ≲ 200 cm⁻¹) Raman spectrum from the time correlation of the cell polarizability evaluated along the trajectory.

**Why fine-tuning is needed.** MACE-OFF23-small gives a crystal that is far too dense. Two independent runs show it (§4.2, §4.4): in LAMMPS at the experimental volume the average pressure is ≈ −0.7 GPa, and in CP2K-MACE NPT at 298 K the cell collapses from 860 to ≈ 764 Å³ (−11 %) within 2 ps. MACE-OFF was never trained on periodic stresses. The cure, already demonstrated on 9MA, is to fine-tune on **DFT energies, forces and stresses** from cells spread over a range of volumes.

**Pipeline (stage numbers = folders in `Calculations/`):**

```
00 reference structures   experimental CIFs → simulation cells (H positions relaxed by DFT)
        │
01 DFT benchmark (CP2K)   numerical settings, functional + dispersion, basis + GTH pseudopotential,
        │                 torsions, polarizability/dipole, Γ-point phonons + Raman vs experiment
02 DFT dataset            frames from the existing fixdeform trajectory + strained cells +
        │                 all three polymorphs → CP2K labels: E, F, σ, μ (Berry phase), α (LINRES/POLAR)
03 MACE fine-tune         (a) potential: E, F, σ   (b) dielectric model: μ, α
        │
04 NPT 78 → 300 K         CP2K-MACE NPT_F (heating and cooling)
        │
05 Active learning        adapted from 9MA `npt_al`: NPT → select over volume/T → DFT (E,F,σ,α) → retrain
        │                 (loop back to 02/03 until converged)
06 Raman                  α(t) from model (b) on saved frames → low-frequency Raman spectrum
```

---

## 2. The system: what we know experimentally

### 2.1 Molecule

- 4,4′-di-*tert*-butylbiphenyl, C₂₀H₂₆, M = 266.43 g mol⁻¹, **46 atoms** per molecule (20 C, 26 H). Only C and H.
- Internal degrees of freedom that matter here:
  - **Biphenyl ring–ring twist** φ_bp (dihedral C_ortho–C_bridge–C_bridge′–C_ortho′). In the 298 K structure (major disorder component): **φ_bp = −39.6°**. The OPLS validation notes quote 32–42° at 160 K.
  - **Two *tert*-butyl rotors** (C_ortho–C_ipso–C_quat–C_methyl; threefold → wells every 120°). At 298 K (major component): methyl dihedrals {−167.2, −50.4, 72.1}° and {−160.4, 80.9, −43.7}°.

### 2.2 Crystal structures (trimorphism)

`Trimorphism.pdf` is currently a **0-byte file** (not synced from Box; §8).

| Phase | T (K) | Space group | Z | a, b, c (Å) | α, β, γ (°) | V (Å³) | V/molecule (Å³) | File |
|---|---|---|---|---|---|---|---|---|
| Room-T | 298 | P-1 | 2 | 8.2340(3), 10.1638(3), 11.8166(4) | 66.646(3), 83.975(3), 71.361(3) | 860.01(6) | 430.0 | `Data/crystal_structures/298k.cif` ✅ |
| Intermediate | 160 | P-1 | 12 | 8.1301, 23.1269, 26.4805 | 92.164, 91.157, 95.384 | 4952.06 | 412.7 | **missing** (`160k.cif`; values from `validate_160.in`) |
| Low-T | < ~150 | ? | ? | ? | ? | ? | ~401–408 | **missing** |

**298 K structure details** (from the CIF): SHELXL-2019/3, Cu Kα, R1(gt) = 0.0491, wR2(ref) = 0.1728, Z′ = 1. **Both *tert*-butyl groups are two-site rotationally disordered**: 0.547/0.453 (C9/C12/C14 vs C10/C11/C13) and 0.727/0.273 (C21/C24/C26 vs C22/C23/C25). H atoms are riding (refinement flags R/GR), so X-ray C–H lengths are short (~0.93–0.96 Å) — see §7 #13.

### 2.3 Experimental V(T) curve

From `Calculations/legacy/mace-off23-small_fixdeform/volume_vs_temperature_ONCOOLING.png` (on cooling; values read off the plot, approximate):

| T (K) | 298 | 280 | 260 | 240 | 220 | 215 | 210 | 205 | 200 | 190 | 180 | 170 | 160 | 150 | 140 | 120 | 100 | 90 | 80 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V/molecule (Å³) | 430.0 | 427.7 | 425.1 | 422.6 | 420.2 | 419.7 | 418.1 | 417.8 | 416.9 | 416.1 | 414.7 | 413.7 | 412.7 | 408.0 | 407.2 | 405.5 | 404.4 | 402.8 | 400.9 |

- 298 → 220 K: roughly linear, dV/dT ≈ 0.13 Å³ K⁻¹ per molecule (≈ 0.25–0.27 per Z = 2 cell → the `DVDT = 0.27` of the fixdeform inputs).
- **Small kink at ~210–215 K** (ΔV ≈ 1.5 Å³/molecule) — possibly the Z = 2 → Z = 12 transition; to confirm against the manuscript.
- **Clear step between 160 and 150 K** (ΔV ≈ −4.7 Å³/molecule, ≈ −1.1 %) — into the low-T phase.
- The raw VT-XRD table (cell parameters with esds at every T) should go into `Data/` once located.

---

## 3. Folder organization

### 3.1 Current tree

```
DtBuDp/
├── README.md                     ← short orientation (points here)
├── PROJECT_LOG.md                ← this file
├── Trimorphism.pdf               ← manuscript draft (0 bytes locally — re-sync from Box)
├── Data/
│   └── crystal_structures/
│       └── 298k.cif              ← experimental 298 K structure (+ embedded hkl/res)
├── Calculations/
│   ├── 00_reference_structures/
│   │   ├── 298K_P-1_Z2/          ← 92-atom cell built from 298k.cif
│   │   │   ├── 300_cell.xyz
│   │   │   ├── coord_300cell.lmp (LAMMPS data file, atomic style; = "coord.lmp" in the run inputs)
│   │   │   └── from_xyz_to_lmp_2.tcl (VMD/TopoTools script that wrote it)
│   │   ├── 160K_Z12/             ← empty — waiting for 160k.cif
│   │   └── lowT_phase/           ← empty — waiting for the low-T CIF
│   ├── 01_dft_benchmark/         (all CP2K; see §5.1)
│   │   ├── 00_structure_prep/        (DFT relaxation of H positions at the experimental cells)
│   │   ├── A_numerical_convergence/  (cutoff, rel_cutoff, EPS, XC smoothing, analytical vs numerical stress)
│   │   ├── B_functional_dispersion/  (functionals × dispersion: 0 K cells, polymorph energies, P at V_exp)
│   │   ├── C_basis_pseudopotential/  (MOLOPT/UZH basis sets × GTH pseudopotentials; BSSE)
│   │   ├── D_torsions_isolated_molecule/ (biphenyl twist + tBu rotation, gas phase and in crystal)
│   │   ├── E_polarizability_dipole/  (LINRES/POLAR and Berry-phase dipole: convergence, functional)
│   │   ├── F_gamma_phonons_raman/    (Γ-point frequencies + Raman intensities vs experiment)
│   │   └── results/                  (summary tables/figures for the decision)
│   ├── 02_dft_dataset/           ← campaign r0 (labels): config.toml, RUN.md, LOG.md, state.json
│   │   ├── frame_selection/      (frames.extxyz = 2200 frames, selection.json, selection.png, molecules.json)
│   │   ├── label/                (created on Leonardo: one folder per frame + array job)
│   │   ├── smoketest/            (created on Leonardo)
│   │   ├── parsed/               (labels.xyz, train/valid/test.xyz after `collect`)
│   │   └── inputs/  outputs/     (empty, unused — the code writes label/; delete when convenient)
│   ├── 03_mace_finetune/
│   │   ├── foundation/           (MACE-OFF23_small.model, MACE-MDP.model — to copy/download on Leonardo)
│   │   ├── models/               (potential_r0/, dielectric_r0/ written by the training jobs)
│   │   ├── datasets/  configs/  validation/   (empty)
│   ├── 04_npt_ramp_78-300K/
│   ├── 05_active_learning/
│   ├── 06_raman/
│   └── legacy/                   ← everything that existed before this log, unchanged
│       ├── opls-aa_classical/
│       ├── mace-off23-small_fixdeform/
│       └── uma_fixdeform/
├── code/                         ← the `dtbubp` package (future GitHub repo; see code/README.md)
├── Figures/  Tables/  Presentations/  Videos/  Structure_fitting/   (manuscript folders, empty)
```

### 3.2 Every move made

All moves used `mv -n` (never overwrite). **No file was deleted or modified.** File count 35 before and 35 after (2026-10-04, first pass).

| Date | Old path | New path |
|---|---|---|
| 10-04 | `Calculations/MD/` | `Calculations/legacy/` (folder renamed) |
| 10-04 | `Calculations/MD/Active_learning/` | `Calculations/legacy/opls-aa_classical/` — holds the **classical OPLS-AA** inputs, not active learning |
| 10-04 | `Calculations/MD/fixdeform/` | `Calculations/legacy/mace-off23-small_fixdeform/` |
| 10-04 | `Calculations/MD/fixdeform_uma/` | `Calculations/legacy/uma_fixdeform/` |
| 10-04 | `…/Active_learning/298k.cif` | `Data/crystal_structures/298k.cif` |
| 10-04 | `…/fixdeform/300_cell.xyz`, `coord_300cell.lmp`, `from_xyz_to_lmp_2.tcl` | `Calculations/00_reference_structures/298K_P-1_Z2/` |
| 10-04 (2nd pass) | empty `01_dft_benchmark/{A_molecular_conformers, B_torsion_scans, C_dimers_clusters, D_dipole_polarizability, E_periodic_crosscheck}` | renamed for the CP2K-only benchmark: `A_numerical_convergence`, `B_functional_dispersion`, `C_basis_pseudopotential`, `E_polarizability_dipole`, `D_torsions_isolated_molecule`; new `00_structure_prep`, `F_gamma_phonons_raman` |

**Path consequences** (only if a legacy run is re-run):
- `cif2lammps.py` defaults to `298k.cif` in the working directory → pass `../../../Data/crystal_structures/298k.cif`.
- The MACE/UMA LAMMPS inputs `read_data coord.lmp` → use `00_reference_structures/298K_P-1_Z2/coord_300cell.lmp`.
- `from_xyz_to_lmp_2.tcl` and `300_cell.xyz` were moved together, so it still works.

**DtBuBP files that live in the 9MA folder** (not moved — they are referenced by the 9MA log; see §8 Q8):
`9MA/cp2k/298k_major.xyz`, `npt_mace.inp`, `submit.sh`, `298k_mace_npt-1.cell`, `298k_mace_npt-1.ener`, `create_cp2k_model.py`, `build_cp2k_plumed.sh`, `cp2k_mace_leonardo_handoff.md`.

---

## 4. Existing work, audited

### 4.1 Classical OPLS-AA model — `Calculations/legacy/opls-aa_classical/`

| File | What it is |
|---|---|
| `cif2lammps.py` | Builds an OPLS-AA LAMMPS data file from the CIF: parses cell/symmetry, keeps the **major** tBu disorder component, expands P-1, finds bonds under PBC, assigns 7 OPLS types (CA_H, CA_CC, CA_CT, CQ, CT3, HA, HC), writes `system.data` and `molecule.xyz`. |
| `system.data` | 92 atoms (Z = 2, the **298 K cell**), 94 bonds, 168 angles, 236 dihedrals, 24 impropers; `real` units, `lj/cut/coul/long`. |
| `molecule.xyz` | One 46-atom molecule (= molecule 1 of the 298 K cell). |
| `validate_160.in` + `check_160.py` | 160 K NPT check against the 160 K cell (0.2 + 0.5 ns, 4×2×1, 12 Å cutoff). Pass criteria: V within 3–4 %, a/b/c within 2 %, angles within 1°, twist overlapping 32–42°. V2 (ring–ring twist) is the only unfitted parameter (`-var v2`). |
| `rotors.in` | Fixed-T runs for tBu hop kinetics → Arrhenius fit. |
| `ramp.in` | 500 ps at T0, then 80 → 300 K over 20 ns (11 K/ns), Langevin + `nph tri` (pdamp 2 ps). |
| `cycle.in` | 80 → 300 → 80 K heat/cool cycle on a 3×3×3 replicate. |

**No outputs** from these runs are in the folder. Inconsistencies in §7.

### 4.2 MACE-OFF23-small, LAMMPS fix-deform cooling run — `Calculations/legacy/mace-off23-small_fixdeform/`

**This trajectory seeds the DFT dataset.** Why fix-deform and not NPT: LAMMPS' `mliap unified` MACE interface does not pass the MACE stress to the barostat (established in the 9MA work, 9MA `LOG.md` 2026-10-04), so the volume was imposed instead.

| Setting | Value |
|---|---|
| Code | LAMMPS (4 Jul 2026 dev), KOKKOS, 1 GPU |
| Potential | `pair_style mliap unified MACE-OFF23_small.model-mliap_lammps.pt` |
| Cell | Z = 2, 92 atoms, experimental 298 K cell (V₀ = 860.01 Å³) |
| Units / timestep | `metal`, Δt = 0.5 fs |
| Equilibration | 100 ps NVT at 300 K (Nosé–Hoover, τ = 50 fs) |
| Ramp | 10 ns NVT 300 → 50 K; `fix deform` isotropic scale 0.97312 on lengths and tilts, V 860.01 → 792.51 Å³ (0.27 Å³/K × 250 K) |
| Output | positions every 100 steps (50 fs) → `all_1.lammpstrj` |
| Speed | 2.29 ns/day (92 atoms, 1 GPU) |

Contents: `all_1.lammpstrj` (750 MB, **197 389 frames**, steps 200 000 → 19 938 800, T_target 300 → **53.3 K**; positions only); `traj.xyz` (**stale**, 32 405 frames); `log.lammps` (only to step 3.44 M ≈ 259 K); `plot_tbu_rotations.py` → rotor/twist `.dat/.png/.pdf`; `volume_vs_temperature_ONCOOLING.png` (experimental V(T), §2.3).

**Findings:**

1. **Density far too high.** At the imposed experimental volumes:

   | Step | T_target (K) | ⟨T⟩ (K) | ⟨P⟩ (bar) |
   |---|---|---|---|
   | equil. (300 K, V_exp) | 300 | — | −7 054 |
   | 0.70 M | 293.8 | 297.3 | −6 618 |
   | 1.20 M | 287.5 | 290.0 | −6 789 |
   | 1.70 M | 281.3 | 284.3 | −6 865 |
   | 2.20 M | 275.0 | 277.5 | −6 860 |
   | 2.70 M | 268.8 | 271.5 | −6 875 |
   | 3.20 M | 262.5 | 265.7 | −7 215 |

   ≈ −0.7 GPa of tension at V_exp. Consistent with the CP2K-MACE NPT collapse to −11 % volume (§4.4.2).
2. **Rotors freeze progressively on cooling** (from `tbu_rotations.png`, approximate): free hopping 300 → ~150 K; tBu 4 (C78) settles ~145 K, tBu 3 (C50) ~135 K, tBu 2 (C21) ~110 K, tBu 1 (C9) ~100 K. Physical meaning to be judged with the DFT-quality model.
3. **Biphenyl twist** stays at ≈ ±40° (experiment ≈ 40°), with a few brief excursions through planarity at 250–300 K.
4. **Cell size.** MACE-OFF23-small: r_cut 4.5 Å, 2 layers → receptive field ≈ 9 Å; the cell's shortest periodic distances are ≲ 8.2 Å. Fine for DFT labelling (also periodic), not for production NPT (§5.4).
5. Isotropic NVT at imposed volumes samples neither cell-shape fluctuations nor the phase transitions → strained cells and the other polymorphs must be added (§5.2).

### 4.3 UMA (fairchem), fix-deform — `Calculations/legacy/uma_fixdeform/`

Same protocol split into `equil_uma.in` + `ramp_uma.in` (fairchem-LAMMPS allows one `run` per script), UMA `task_name=omc`. `all_uma.lammpstrj`: 9 235 frames, only ~0.46 ns (T ≈ 288 K). No log. Incomplete; kept for reference.

### 4.4 Related prior work: the 9MA project (`Box/Documents/Manuscripts/9MA/`)

Read on 2026-10-04 (`README.md`, `LOG.md`, `PROJECT_CONTEXT.md`, `data/README.md`, `cp2k/cp2k_mace_leonardo_handoff.md`, `cp2k/9MA_mace_cp2k_test/README.md`, `cp2k/npt_al/`, `code/` (topoal), `spectroscopy/`). The 9MA project (9-methylanthracene + dithiin topochemical Diels–Alder; MACE + Deep-TDA CV + OPES) built and debugged **exactly the toolchain this project needs**. What carries over:

#### 4.4.1 The CP2K-MACE engine on Leonardo (working)
- Custom **CP2K master (f89ef8372a) + MACE via libtorch, CUDA/A100**, Libxc 7.1.2 (with Fortran), tblite off, **PLUMED 2.10.0 with libtorch** linked in. Binary `$HOME/software/cp2k-mace/install/bin/cp2k.psmp`; env `$HOME/cp2k_torch_env` (torch 2.7.1+cu126); environment script `9MA/cp2k/9MA_mace_cp2k_test/env_cp2k_mace.sh`.
- Required fixes, all documented in `9MA/cp2k/cp2k_mace_leonardo_handoff.md`: binutils 2.41 linker on PATH; `LD_LIBRARY_PATH` must start with `cp2k-mace/install/lib64` (otherwise the old 2026.2 libcp2k is loaded); `EWALD_TYPE NONE` + `DO_ELECTROSTATICS F` for pure MACE; local patch `v_ptr(:, :)` in `src/manybody_e3nn.F` (virial rank).
- **Virial validated** (9MA test 02): CP2K analytical ⅓Tr(σ) vs −dE/dV = 0.717 vs 0.731 GPa (reactant), 0.648 vs 0.662 GPa (product).
- Model export: `create_cp2k_model.py my.model --dtype float64` → `my.model-cp2k.pth`. **CP2K truncates file names at ~80 characters** → link the model as `model.pth` in each run folder.
- Speed: ~0.016–0.02 s/step for 92 atoms on one A100.

#### 4.4.2 The DtBuBP CP2K-MACE NPT test (Will, in the 9MA folder)
- `9MA/cp2k/npt_mace.inp`: FIST + MACE-OFF23-small, **NPT_I, 298 K, 1 bar**, CSVR τ 100 fs, barostat τ 1000 fs, Δt 0.5 fs; start = `298k_major.xyz` (the 92-atom DtBuBP 298 K cell, major disorder components, **X-ray H positions**).
- Result (`298k_mace_npt-1.cell`, 19.35 ps): V 860 → ~742 Å³ within 0.5 ps, then oscillates; **mean V after 2 ps = 764.0 Å³ (−11.2 %)**. The run was not interpreted as physics. The 9MA tests later showed the engine is correct, so the collapse is MACE-OFF23's stress/E(V) for this crystal.

#### 4.4.3 `9MA/cp2k/npt_al/` — NPT active learning with DFT stresses (proven)
The template for our Stage 05 (and the labelling half of Stage 02):

| step | 9MA setting |
|---|---|
| MD | CP2K-MACE NPT_I, 300 K, 1 bar, **Δt 0.25 fs**, frame every 2.5 fs (positions, cell, stress, MACE forces/energy); runs of ≤ 5 ps; **volume watchdog** stops a run at 1.6 × V_mean (EXIT file), keeping the frames before the blow-up |
| select | ~200 frames/round spread **evenly over volume bins** (0.92–1.45 × V_mean, 12 bins), ≥ 25 fs apart within a run, MACE |F| < 25 eV/Å |
| label | CP2K `RUN_TYPE ENERGY_FORCE` + `STRESS_TENSOR ANALYTICAL` (**not REFTRAJ** — REFTRAJ wrote all-zero stress files), BLYP-D3, DZVP-MOLOPT-SR-GTH, GTH-BLYP, 600/60 Ry, NN10 smoothing, EPS_SCF 1e-8, EPS_DEFAULT 1e-16; 4 array jobs; 45 min per-frame timeout |
| collect | extxyz: `REF_energy` (eV), `REF_forces` (eV/Å), `REF_stress` (eV/Å³, **ASE sign, Voigt order**). Old data without stress → zero stress weight |
| train | `mace_run_train` from MACE-OFF23-small (`--foundation_model`), `--freeze=4`, `--loss=stress`, E/F/σ weights 10/10/1e4 (SWA 10/100/1e4), lr 1e-3, EMA 0.999, batch 16, 500 epochs, SWA from 400, float64, seed 1, `--E0s=average`; then export to CP2K |
| decide | stable volume in the second half of every unbiased run; stress RMSE < 0.15 GPa and force RMSE < 0.10 eV/Å **on fresh frames the model has not seen**; max 6 rounds |

Results on 9MA: round 0 (no stress training) all runs blew up (stress RMSE 0.86 GPa); **round 1 (154 stress frames) NPT stable** (stress RMSE 0.21 GPa, force RMSE 0.034 eV/Å); round 2 stable. The analytical DFT stress was validated against numerical (≤ 0.002 GPa) with the `smoketest`.

#### 4.4.4 Lessons from 9MA that apply here (traps)
1. **X-ray H positions** (riding H, C–H 0.94–1.01 Å) gave +14.7 GPa and 7.5 eV/Å forces on H at the start of every NPT run. → Relax H at fixed cell (DFT) before any benchmark or MD (`01_dft_benchmark/00_structure_prep`).
2. REFTRAJ: `EVAL` defaults to `NONE`, `LAST_SNAPSHOT` must be set, and **stress is not written** → use ENERGY_FORCE single points per frame.
3. Units: Hartree → eV × 27.211386; Hartree/Bohr → eV/Å × 51.422086; CP2K stress printed in bar (master) / GPa — convert and fix the sign convention once, in one place.
4. Every element needs a `&KIND`; never carry a cell over from a template (compare shortest lattice vectors).
5. One stuck SCF can stall a whole labelling job → per-frame timeouts, `MAX_SCF` limits, drop frames with atoms < 0.7 Å apart.
6. MLIP extrapolation fragments structures; the frames just before a blow-up are the most valuable labels.
7. Deleting per-epoch MACE checkpoints saves tens of GB (9MA: 22 GB).
8. A bash `cmd | grep -q` under `set -o pipefail` fails spuriously — capture output first.

#### 4.4.5 Spectroscopy experience (9MA `spectroscopy/`)
Will has previously computed crystal Γ-point frequencies and **Raman spectra with CRYSTAL** (PBE-D/def2-SVP, `INTRAMAN`/`RAMSPEC`), compared them with experimental Raman (`Red_RT_init_full.asc`) and THz-TDS (80 K and RT), and projected reaction displacements onto phonon modes (PDCA). The same comparison scripts (`FreqvsExp.py`, `FreqvsExp_THz.py`) are a model for Stage 01F/06 validation — now with CP2K.

#### 4.4.6 Code infrastructure
- `9MA/code/` = the `topoal` package (GitHub `wstoll-ur/Deep-LDA_Topochem`): CP2K/LAMMPS/MACE/PLUMED engine writers and parsers, Slurm profile for Leonardo (`profiles/leonardo.yaml`: accounts `IscrB_MET2SAF` (GPU) / `IscrB_MET2SAF_0` (CPU, serial), partitions `boost_usr_prod`, `dcgp_usr_prod`, `lrd_all_serial`), automatic self-logging, pytest with fake engines.
- `9MA/cp2k/npt_al/npt_al.py` (single-file driver, 930 lines, with tests).
- Our `code/` repository should **reuse** these (CP2K deck writers, extxyz conventions, Slurm chaining, logging) rather than start over (§6.4).

---

## 5. The plan, stage by stage

### 5.0 Stage 00 — Reference structures

- [x] 298 K, Z = 2 cell (92 atoms): `00_reference_structures/298K_P-1_Z2/`. (Same atoms as `9MA/cp2k/298k_major.xyz`.)
- [ ] 160 K, Z = 12 cell (552 atoms) from `160k.cif` — **waiting on file**.
- [ ] Low-T phase cell — **waiting on file**.
- [ ] Disorder: build both ordered variants of the 298 K structure (major/major and the alternatives) — they are rotor configurations the dynamics visits.
- [ ] **Relax H positions at the experimental cell (CP2K, heavy atoms fixed)** for every structure before use (lesson 4.4.4-1).

### 5.1 Stage 01 — Choosing the level of theory (CP2K only)

All calculations in CP2K Quickstep (GPW: Gaussian-type basis + plane-wave density, GTH pseudopotentials). The "pseudopotential" axis of the benchmark is real here: each functional has its matched GTH set (GTH-PBE, GTH-BLYP, GTH-PBE0/HF for hybrids), and C/H use q4/q1.

**Practical constraint:** the production labels need E, F, **analytical stress**, dipole and polarizability for several thousand atoms-worth of frames per round. Analytical stress in GPW is routine for GGA; for hybrids (via ADMM) stress and LINRES/POLAR support must be verified on our build before a hybrid is considered for production. Plan accordingly: a **GGA + D3(BJ)** production level, with hybrid spot checks.

#### 5.1.0 `00_structure_prep`
H-only relaxations of the 298 K (and later 160 K, low-T) cells at a provisional level (PBE-D3(BJ)/DZVP-MOLOPT-SR-GTH, 600 Ry). Record C–H lengths before/after.

#### 5.1.A `A_numerical_convergence`
- `CUTOFF` 400 → 1200 Ry, `REL_CUTOFF` 40 → 80 Ry: converge energy differences, forces **and stress** (stress converges more slowly than energy; target ≤ 0.02 GPa, forces ≤ 1 meV/Å, ΔE ≤ 0.1 meV/atom).
- `EPS_DEFAULT`, `EPS_SCF`, `NGRIDS`, XC grid smoothing (NN10 as in 9MA vs none).
- **Analytical vs numerical stress** at the experimental cell and at ±5 % strained cells (the 9MA `smoketest`), for every functional that reaches the shortlist.
- k-points: Γ-only is standard for the 92-atom cell (8.2 × 10.2 × 11.8 Å, an insulator); verify with a 2×2×2 k-mesh on one frame, or with a 2×1×1 supercell at Γ.

#### 5.1.B `B_functional_dispersion`
- Candidates: **PBE-D3(BJ)**, **BLYP-D3(BJ)** (continuity with 9MA, which used BLYP-D3 zero-damping), revPBE-D3(BJ), optB88-vdW or rVV10 (non-local), r²SCAN-D3(BJ)/rVV10 if the GTH set exists in our build; **PBE0-D3(BJ) with ADMM** as the hybrid reference (single points and one relaxation).
- D3 zero-damping vs BJ; three-body term (ATM) on/off; D4 if our CP2K build provides it.
- Tests:
  1. 0 K variable-cell relaxation (`CELL_OPT`, fixed symmetry not imposed) of each polymorph → V_0K vs experiment. Expected: V_0K a few % below the lowest-T experimental volume (thermal expansion + zero-point); report the ratio, not just the error.
  2. **Pressure at the experimental 298 K cell** (H relaxed): the DFT number MACE has to match.
  3. **Relative lattice energies of the three polymorphs** — the ordering must be compatible with the observed transitions.
  4. Lattice energy (crystal − isolated molecule in a large box) vs the experimental sublimation enthalpy, if available (to check).

#### 5.1.C `C_basis_pseudopotential`
- Basis: DZVP-MOLOPT-SR-GTH (9MA), DZVP-MOLOPT-GTH, TZV2P-MOLOPT-GTH, and the newer MOLOPT-UZH sets (DZVP/TZVP/TZV2P-MOLOPT-PBE-GTH-q4/q1, if in our data dir).
- Pseudopotential: GTH matched to the functional vs a "wrong" GTH (e.g. GTH-PBE with BLYP) to quantify the sensitivity.
- BSSE: counterpoise estimate on a dimer and on the lattice energy for DZVP vs TZV2P.
- Pick the smallest basis whose stress and forces stay within the A-tolerances of the TZV2P result.

#### 5.1.D `D_torsions_isolated_molecule`
- Isolated molecule in a large cubic box (≥ 25 Å, `POISSON_SOLVER MT` or `WAVELET`, `PERIODIC NONE`):
  - biphenyl twist scan 0–90° (relaxed, 10–15° steps, denser near 0° and 90°): minimum position and both barriers;
  - tBu rotation scan 0–120°: barrier.
- Compare with literature gas-phase values for biphenyl and *tert*-butylbenzene (to collect in §10), and across the B/C shortlist.
- In-crystal tBu barrier: constrained rotation of one rotor in the 298 K cell (relevant to the hopping rates in §4.2-2).

#### 5.1.E `E_polarizability_dipole`
- **Polarizability:** `&PROPERTIES &LINRES &POLAR` (DFPT; `PERIODIC_DIPOLE_OPERATOR T` = Berry phase for the crystal; `DO_RAMAN` computes the dipole–dipole polarizability [CP2K-POLAR]).
- **Dipole:** `&DFT &PRINT &MOMENTS PERIODIC T` (Berry phase; defined modulo a polarization quantum → jumps in MD that must be unwrapped [CP2K-MOMENTS]).
- Tests: α (isotropic and anisotropy) of the isolated molecule vs basis (MOLOPT sets have no diffuse functions — check how much α is underestimated against the largest basis available / literature value for biphenyl derivatives) and vs functional; α of the 298 K cell vs cutoff; whether LINRES/POLAR runs with ADMM hybrids in our build; cost per frame relative to ENERGY_FORCE.
- Decide whether the production labels include α for every frame or for a subset (α labels can be fewer than E/F/σ labels).

#### 5.1.F `F_gamma_phonons_raman`
- `VIBRATIONAL_ANALYSIS` with `INTENSITIES T` + `LINRES/POLAR` (Raman) and `MOMENTS` (IR) [CP2K-VIB] on the DFT-relaxed 298 K cell (and 160 K/low-T cells later): low-frequency modes and Raman activities.
- Compare with experimental low-frequency Raman (if available, §8 Q5), in the style of Will's CRYSTAL comparisons for 9MA (§4.4.5).
- This is the harmonic, 0 K reference that the MD Raman (Stage 06) must approach at low T.

**Decision rule (to fill in `results/`):** choose the cheapest functional/basis/GTH combination that (i) is converged per A, (ii) places the 0 K volumes and polymorph ordering sensibly (B), (iii) gets the torsion profiles right (D), and (iv) gives Γ-point low-frequency Raman closest to experiment (F). Record the full decision table here.

### 5.2 Stage 02 — Building the DFT dataset

> **Status 2026-10-04:** campaign r0 implemented (§9, "Campaign r0"): 2000 fix-deform + 200 strained frames selected; CP2K deck, smoke test, labelling and collection code ready. Pools 3–6 below come in later campaigns.

**Source pools:**
1. **Fix-deform MACE-OFF23 frames** (`legacy/mace-off23-small_fixdeform/all_1.lammpstrj`, 197 389 frames, 300 → 53 K): sub-sample a few hundred, stratified by temperature (10 K bins) and by diversity (farthest-point sampling on rotor/twist dihedrals and/or MACE descriptors).
2. **Strained cells** (isotropic ±1–6 % and anisotropic/shear strains of selected frames): teaches MACE the stress around and beyond the right density — the 9MA lesson that the expanded side must be labelled explicitly.
3. **The 160 K (Z = 12) and low-T phase cells** once the CIFs arrive, plus short MD of them.
4. **Both disorder variants** of the 298 K structure.
5. **The DtBuBP CP2K-MACE NPT trajectory** (`9MA/cp2k/298k_mace_npt-*`, collapsed cells at ~740–800 Å³) if the position file is on Leonardo — exactly the compressed configurations MACE-OFF goes to.
6. Isolated molecules (gas-phase reference; torsion scan geometries from 01D).

**Labels per configuration (CP2K, one `ENERGY_FORCE` job per frame):** `REF_energy` (eV), `REF_forces` (eV/Å), `REF_stress` (eV/Å³, ASE sign, Voigt), `REF_dipole` (e·Å, Berry phase, unwrapped consistently within a trajectory), `REF_polarizability` (Å³, 3×3), plus `config_type`, `source`, `source_frame`, `T_target`, `level_of_theory`.

**Size and cost:** ~500–1500 periodic configurations of 92 atoms (+ some of 552 atoms). Timing per frame to come from 01A/01E.

**Train/validation/test split:** by trajectory segment / temperature block, not random by frame (consecutive frames are correlated).

### 5.3 Stage 03 — Fine-tuning MACE

| Model | Targets | Starting point | Notes |
|---|---|---|---|
| **(a) Potential** | E, F, σ | MACE-OFF23-small (as 9MA; fast in CP2K) — compare MACE-OFF24-medium or a multihead foundation model if (a) underfits | Start from the 9MA `npt_al` train settings (§4.4.3): `--freeze=4`, `--loss=stress`, `--stress_weight=1e4`, SWA, float64. Alternative: multihead fine-tuning with replay [MACE-releases]. Export with `create_cp2k_model.py`. |
| **(b) Dielectric** | μ, α | `AtomicDielectricMACE` (`--loss="dipole_polar"`) **trained from scratch** on our labels (MACE-MDP cannot be fine-tuned in released MACE; see §9, 2026-10-04 correction) | Used only in post-processing (Stage 06), so it does not need a CP2K export. |

**Validation (`03_mace_finetune/validation/`):** RMSE/MAE of E (meV/atom), F (meV/Å), σ (GPa), μ, α on held-out data, by temperature; P at the DFT-relaxed experimental 298 K cell vs DFT; 0 K relaxed cells of the three polymorphs vs DFT; torsion profiles vs 01D; Γ-point phonons vs 01F; MACE virial vs −dE/dV (9MA test-02 style); speed in CP2K.

### 5.4 Stage 04 — NPT 78 → 300 K (CP2K-MACE)

- **Engine:** CP2K FIST + MACE (stress passes through; LAMMPS mliap does not).
- **Ensemble:** start with NPT_I (as validated on 9MA), then **NPT_F** (fully flexible triclinic cell) — needed for anisotropic thermal expansion and the phase transitions. CSVR thermostat; barostat τ ≈ 1 ps; Δt 0.5 fs (0.25 fs for the first AL rounds, as in 9MA).
- **Temperature ramp:** CP2K `&MD` has no built-in linear ramp of the thermostat target → run in segments (e.g. 5–10 K steps with restarts) or with a small driver; document the protocol chosen.
- **Cells:** a supercell whose perpendicular widths exceed ~2× the receptive field (e.g. 2×2×2 of the Z = 2 cell, 736 atoms), plus the Z = 12 cell (552 atoms) and the low-T cell. A transition between different Z only happens in a cell commensurate with both phases — choose that once `160k.cif` is in hand.
- **Protocol:** equilibrate at 78 K, heat to 300 K; cool back for hysteresis at ~155 K and ~210 K; fixed-T NPT points at the experimental temperatures of §2.3.
- **Comparison:** V(T) and cell parameters vs §2.3; twist and rotor statistics vs the crystallographic disorder.

### 5.5 Stage 05 — Active learning (adapted from 9MA `npt_al`)

Changes needed relative to `9MA/cp2k/npt_al` (to implement when we start coding):

| `npt_al` (9MA) | DtBuBP version |
|---|---|
| one fixed mean cell, `V_mean` from config | per-start cell; volume bands relative to the experimental V(T) at the run temperature |
| reactant/product starts at 300 K | starts from each polymorph (and disorder variant), at several temperatures 78–300 K |
| NPT_I only | NPT_I first rounds, NPT_F later |
| OPES on the Deep-TDA CV | no CV bias needed initially; optionally mild temperature/pressure excursions (e.g. 350 K, ±0.5 GPa) to sample beyond the target range |
| labels E, F, σ (BLYP-D3) | E, F, σ at the Stage-01 level, + μ, α (all frames or a subset) |
| `&KIND` H C N O S, ATOMS H C N O S | H C only |
| stop: V stable, σ RMSE < 0.15 GPa, F RMSE < 0.10 eV/Å | same, plus: V(T) within the experimental curve tolerance (to set), and a committee/force-deviation check on long runs |

Uncertainty: in addition to `npt_al`'s "fresh-frame error" test, a committee of 3–4 models (different seeds) gives on-the-fly force/stress spread in long production runs. Failure detectors: volume watchdog, C–H < 0.9 or > 1.3 Å, C–C < 1.2 or > 1.8 Å, temperature runaway. Each round = one §9 entry with frame counts and errors.

### 5.6 Stage 06 — Low-frequency Raman

- Production CP2K-MACE NPT (or NVT at the NPT-average cell) at the temperatures of interest, positions + cell saved every 2–5 fs.
- Evaluate α(t) with model (b) in Python (batched on GPU) over the saved frames; the dielectric model does not have to run inside CP2K.
- Spectrum = Fourier transform of the polarizability autocorrelation: isotropic (tr α) and anisotropic (traceless part); quantum correction factor; Bose–Einstein and frequency prefactors matching the experimental quantity.
- Sampling: ≤ 80 fs needed for 200 cm⁻¹ (Nyquist), so 2–5 fs is ample; ≈ 1 cm⁻¹ resolution needs ≥ 33 ps per window; average many windows and runs.
- Check at low T against the harmonic CP2K Raman (01F) and against experiment.

---

## 6. Computing environment and conventions

### 6.1 Leonardo (CINECA)
- **Accounts/partitions** (from 9MA): GPU `IscrB_MET2SAF` on `boost_usr_prod`; CPU `IscrB_MET2SAF_0` on `dcgp_usr_prod` (112 cores/node, used for CP2K DFT); serial `lrd_all_serial`; 24 h jobs.
- **DFT:** module `cp2k/2024.1--intel-oneapi-mpi--2021.10.0--intel--2021.10.0` (`profile/chem-phys`), `cp2k.popt`, `CP2K_DATA_DIR=$CP2K_HOME/share/data` — used for all 9MA DFT. Alternative: the custom master build (newer features; check D4, MOLOPT-UZH, LINRES with ADMM).
- **CP2K-MACE MD:** custom master build, `env_cp2k_mace.sh` (§4.4.1).
- **MACE training:** `~/mace_env` (9MA `npt_al` config) on one A100.

### 6.2 Software environment rule (Will's standing preference)
All work on Will's machines runs in a dedicated virtual environment named **`Claude`**; nothing is installed into or removed from his personal environments. On Leonardo: a separate venv `Claude` for analysis and dataset tooling; the existing `cp2k_torch_env` and `mace_env` are used as they are (not modified). The package list will be frozen to `code/environment.yml`.

### 6.3 Conventions
- **Units:** eV, Å, fs/ps, K, GPa (state bar when used); α in Å³ (1 a.u. = 0.148185 Å³); μ in e·Å (state if Debye).
- **Stress sign:** ASE convention (positive = tensile), Voigt order (xx, yy, zz, yz, xz, xy) — as in 9MA `npt_al`.
- **Calculation IDs:** `S<stage>-<sub>-<NNN>`, e.g. `S01A-cutoff-003`. Folder name = ID; each ID gets a §9 entry.
- **Every run folder** keeps input(s), job script, exact command, software version/module list, and `NOTES.md` (purpose, settings, result, decision fed).
- **Never on GitHub:** trajectories, DFT outputs, model checkpoints, CIFs (CSD licence). GitHub: code, configs, small tables, this log.
- Before deleting anything: a manifest in `logs/` (as in 9MA).

### 6.5 BlueHive (University of Rochester CIRC), from 2026-10-05
- **Transfer:** private GitHub repo `wstoll-ur/dtbubp` (BlueHive login needs Duo, so no direct copy). Work tree = the Box folder; Claude pushes from the Cowork VM with a write deploy key (git dir kept in the VM, re-cloned per session); BlueHive clone at `/scratch/wstoll/DtBuDp/DtBuDp` (= `/gpfs/fs2/scratch/...`) with its own write deploy key (`Host github-dtbubp`). `.gitignore` is a whitelist (code, configs, notes, `frames.extxyz`, `eos/V430`, `bluehive/`); reports come back through `bluehive/reports/`. On bluehive3: `module load git` first; repo set to `pull.rebase false`.
- **Compute = Vermont nodes bhx0123, bhx0124, bhx0125** (lab nodes outside Slurm, launched by ssh/nohup like Will's `vermont_crystal.sh`): 24 cores (2× Xeon E5-2650 v4, Broadwell, AVX2), 62 GB, RHEL 7.9, **no shared file system** with bluehive3 (only node-local `/home` = `/local_scratch/home`, 1.3 TB free), 10 GbE (`eno1`, 192.168.18.0/24), **no InfiniBand**; passwordless ssh node↔node works; nodes have internet. Missing on the nodes: bzip2, unzip, python3, git, apptainer.
- **CP2K = 2024.1 (same version as all Leonardo runs), compiled by `bluehive/build_cp2k.sh` on bhx0123** into `/home/wstoll/Claude/cp2k-2024.1` and copied to 0124/0125 at the same path; environment `source ~/Claude/cp2k-2024.1.env`. Toolchain: GCC 11.2.0 (`gcc/11.2.0/b2` module; no 12/13 on the nodes), OpenMPI 4.1.5, OpenBLAS 0.3.25 (TARGET HASWELL), ScaLAPACK 2.2.1, FFTW 3.3.10, libxc 6.2.2, libint 2.6.0 (lmax 5), libxsmm 1.17, CMake 3.28.1; all optional packages off (COSMA, ELPA, SIRIUS, HDF5, GSL, spglib, libvori, PLUMED, libgrpp, ...). `python3` for the arch files and fypp = a wrapper `~/Claude/bin/python3` around `/software/python3/3.7.1` (loading the module breaks the system python 2 that libint needs).
- **Not usable:** the nodes' `cp2k/2025.1` module (built for newer CPUs: SIGILL on Broadwell); the CP2K Apptainer container (no apptainer on the nodes; pull fails on GPFS); the Slurm `standard` partition (Will: not to be used).
- **Running jobs:** dtbubp writes the job scripts with `--dry-run`; `bluehive/vermont.sh run <job.sh> <node>[,node,...] [task]` copies the job folder to the node(s) (`/home/wstoll/dtbubp_runs/...`), rewrites paths, starts it with `setsid nohup`; `status`, `pull` (results back), `watch`, `kill`. Launchers in `[dft]`: `launcher = "mpirun"`, lanes pinned with `--cpu-set` (2 frames × 12 ranks per node for labels); multi-node runs use `[bench] hosts` (`mpirun --prefix ... --host bhx0123:24,... -x PATH -x LD_LIBRARY_PATH`, TCP on 192.168.18.0/24).

### 6.4 Future GitHub repository (`code/`)
Planned layout (created when coding starts):
```
code/
├── README.md
├── environment.yml
├── dtbubp/            (python package: io, structures, cp2k decks/parsers, selection, dielectric, raman)
├── scripts/           (thin CLIs)
├── workflows/         (Slurm templates for Leonardo)
└── tests/
```
Reuse from 9MA: `npt_al.py` (driver pattern, CP2K parsers, stress conversion, Slurm chaining), `topoal/engines/{cp2k,mace,hpc}.py`, `create_cp2k_model.py`. Legacy DtBuBP scripts (`cif2lammps.py`, `check_160.py`, `traj2xyz_ase.py`, `plot_tbu_rotations.py` ×2) are consolidated into the package. Option to discuss: a shared library between the 9MA and DtBuBP repositories (§8 Q9).

---

## 7. Issues found during the audit

| # | Where | Issue | Impact / action |
|---|---|---|---|
| 1 | `Trimorphism.pdf` | 0 bytes locally | Re-sync from Box. |
| 2 | `160k.cif`, low-T CIF | Referenced but not in the folder | Needed for Stages 00, 02, 04. Will is adding them. |
| 3 | `opls-aa_classical/cif2lammps.py`, `system.data` | Say "built from 160k.cif, Z = 12", but the input is `298k.cif` and `system.data` is the 92-atom Z = 2 cell | Labels wrong; content = 298 K cell. |
| 4 | `opls-aa_classical/validate_160.in` | Replicates the Z = 2 data 4×2×1 and compares with the Z = 12 cell; c ≈ 10.8 Å < the 24 Å its comment requires | The 160 K validation was not valid as set up. |
| 5 | `opls-aa_classical/rotors.in`, `ramp.in` | Comments ("1656 atoms, 72 tBu") assume a Z = 12 cell | Same root cause as #3. |
| 6 | `opls-aa_classical/cycle.in` | `include lmp.in` missing; comments mention 550 K | Out of date. |
| 7 | `mace-off23-small_fixdeform/traj.xyz` | Stale: 32 405 vs 197 389 frames | Reconvert from `all_1.lammpstrj`. |
| 8 | `mace-off23-small_fixdeform/log.lammps` | Covers only to step 3.44 M of 20.2 M | Most of the P history missing; harmless for DFT relabelling. |
| 9 | `traj2xyz_ase.py` | Docstring mentions velocities and N/O types | Cosmetic. |
| 10 | Fix-deform inputs | `NCELLS` comment says 3×3×3 | Cosmetic; NCELLS = 1 ran. |
| 11 | MACE-OFF23 | ⟨P⟩ ≈ −0.7 GPa at V_exp; CP2K NPT collapses −11 % | Main failure mode → stress labels (§5). |
| 12 | All fix-deform runs | 92-atom cell < 2× receptive field | Fine for labelling; supercell for production. |
| 13 | `298k_major.xyz`, `300_cell.xyz`, `coord_300cell.lmp` | X-ray (riding) H positions; every LAMMPS and CP2K run so far started from them | Large initial forces/stress (9MA saw +14.7 GPa). Relax H first (§5.0). |
| 14 | DtBuBP CP2K files in `9MA/cp2k/` | DtBuBP test lives in the 9MA project | Cross-referenced here; whether to copy them over is Q8. |

---

## 8. Open questions and decisions needed

| # | Question | Why it matters | Default if no answer |
|---|---|---|---|
| Q1 | ~~Production functional~~ **Answered 2026-10-04:** eventually the benchmark winner; **for now (campaign r0) the 9MA settings (BLYP-D3, DZVP-MOLOPT-SR-GTH, 600/60 Ry), benchmark skipped.** | | |
| Q2 | ~~CP2K build for DFT labels~~ **Answered 2026-10-04: the CP2K 2024.1 module** (as all 9MA DFT). | | |
| Q3 | Dipoles: needed for IR as well, or only Raman (α)? | Berry-phase dipoles need unwrapping | Label them anyway (cheap via MOMENTS); train only if IR is wanted |
| Q4 | α labels on every frame or a subset? | LINRES/POLAR cost per frame | Decide from the 01E timing |
| Q5 | Experimental **low-frequency Raman** (vs T) and the full VT-XRD table — where are they? | Validation for 01F/06 and §2.3 | Put them in `Data/` |
| Q6 | Experimental sublimation enthalpy of DtBuBP? | Lattice-energy check in 01B | Skip if not available |
| Q7 | Which transitions/temperatures does the manuscript report (heating vs cooling)? | Success criteria of Stage 04 | Read `Trimorphism.pdf` once synced |
| Q8 | Copy the DtBuBP CP2K test (`9MA/cp2k/298k_major.xyz`, `npt_mace.inp`, `.cell`, `.ener`, `submit.sh`) into `DtBuDp/Calculations/legacy/cp2k-mace_npt_test/`? | Keep each project self-contained | Copy (not move), leave the 9MA log references intact |
| Q9 | Share code between the 9MA (`topoal`, `npt_al`) and DtBuBP repositories, or copy what we need? | Maintenance | Copy now; refactor into a shared library later |

---

## 9. Changelog

### 2026-10-04 — Stage 0: orientation, audit, organization (Claude)
- Surveyed all 35 files in the project folder (§4.1–4.3).
- Identified the molecule (DtBuBP, C₂₀H₂₆, 46 atoms) and the known structures; read the experimental V(T) curve (steps near 150–160 K and 210–215 K).
- Measured from the 298 K structure (major component): biphenyl twist −39.6°; tBu methyl dihedrals in §2.1.
- Audited the MACE-OFF23-small fix-deform run: 197 389 frames, 300 → 53 K; ⟨P⟩ ≈ −0.66 to −0.72 GPa at V_exp; rotors freeze between ~145 and ~100 K; twist stable at ±40°.
- Audited the UMA run (incomplete) and the OPLS-AA inputs (no outputs; inconsistencies in §7).
- Reorganized the folder (§3.2); no files deleted or edited. Created stage folders `00`–`06` and `code/`.
- First version of the plan used Gaussian 16 for the benchmark (superseded below).

### 2026-10-04 — Switch to CP2K only; 9MA project read (Claude)
- **Decision (Will):** no Gaussian; all electronic-structure work in CP2K.
- Read the 9MA project (§4.4). Key connections:
  - Will's CP2K-MACE NPT test of DtBuBP (`9MA/cp2k/npt_mace.inp`) collapsed to 764 Å³ (−11.2 %), consistent with the −0.7 GPa found here.
  - The CP2K-MACE engine (custom master + MACE + PLUMED, virial patch validated) and the `npt_al` stress active-learning loop (NPT stable after one round on 9MA) are the templates for Stages 02–05.
  - X-ray H positions caused huge initial stresses on 9MA; the same applies to every DtBuBP run so far (§7 #13).
- Rewrote §1, §5 and §6 for CP2K: benchmark axes are numerical settings, functional + dispersion, basis + GTH pseudopotential, isolated-molecule torsions, LINRES/POLAR polarizability and Berry-phase dipoles, and Γ-point phonons + Raman.
- Renamed the empty `01_dft_benchmark` subfolders accordingly (§3.2).
- **Next:** Will adds `160k.cif`, the low-T CIF and a synced `Trimorphism.pdf`; answer Q1–Q9; then write the 01/00 H-relaxation and 01A convergence inputs (first coding step).

### 2026-10-04 — Campaign r0: label the fix-deform frames, then fine-tune (Claude)

**Decisions (Will):**
- Order of work changed: **first** label the existing fix-deform frames with E, F, stress, dipole and polarizability and fine-tune MACE-OFF; **then** NPT. The Stage-01 benchmark is skipped for now.
- Level of theory = the 9MA settings; DFT with the **CP2K 2024.1 module**. Eventually the benchmark winner replaces it (a later relabel).
- **2000** fix-deform frames. Code lives in `DtBuDp/code`.
- Claude added **200 strained cells** on top (10 % extra DFT; `[strain] n_frames = 0` removes them). Reason: the fix-deform volumes only span 793–860 Å³ (0.92–1.00 × V_exp), while MACE-OFF23 NPT goes to 764 Å³ and the 9MA NPT runs failed on the *expanded* side until expanded cells were labelled.

**Code written: `code/` = package `dtbubp`** (numpy only; tests with synthetic CP2K outputs: 7 pass, locally and in the `Claude` venv on the Mac).

| module | does |
|---|---|
| `io.py` | LAMMPS dump reader: sorts atoms by id every frame (the fix-deform dump has no `sort id`), skips unwanted frames fast, converts LAMMPS bounding boxes to cell vectors; extxyz reader/writer |
| `geometry.py` | molecules from the bond graph, **whole molecules** (needed for a consistent dipole), rigid-molecule and affine strains, Berry-dipole branch folding |
| `select.py` | frame selection (below) |
| `cp2k.py` | the label deck (from `templates/label.inp`) and parsers for energy, forces, stress, Berry dipole, LINRES polarizability, wall time; RUN_TYPE DEBUG polarizability table |
| `label.py` | smoke test, per-frame folders, packed Slurm array job, `collect` to extxyz in MACE units |
| `train.py`, `evaluate.py` | MACE training jobs (potential + dielectric), test-set evaluation, CP2K export |
| `cli.py` | `python -m dtbubp.cli --config config.toml select|smoketest|label|status|collect|train|evaluate|inputs` |

**The DFT label (one CP2K run per frame, `RUN_TYPE ENERGY_FORCE`):**
- 9MA settings: BLYP + D3 (zero damping), DZVP-MOLOPT-SR-GTH, GTH-BLYP-q4/q1, CUTOFF 600 / REL_CUTOFF 60 Ry, NGRIDS 5, NN10 XC smoothing, OT/DIIS + FULL_KINETIC, EPS_SCF 1e-8 (outer 10 × 50 inner), EPS_DEFAULT 1e-16, Γ point, `STRESS_TENSOR ANALYTICAL`.
- Added `&DFT &PRINT &MOMENTS PERIODIC T` (Berry-phase dipole) and `&PROPERTIES &LINRES` (FULL_ALL, EPS 1e-8, MAX_ITER 300) `&POLAR DO_RAMAN T, PERIODIC_DIPOLE_OPERATOR T` (DFPT polarizability).
- Verified in the CP2K 2024.1 source: LINRES runs post-SCF inside ENERGY_FORCE (`qs_energies_properties → linres_calculation_low`, `src/qs_energy_utils.F`), so one run per frame gives all five labels; output formats taken from `qs_linres_polar_utils.F` (`POLAR| Polarizability tensor [a.u.]`) and `qs_moments.F` (`Dipole moment [Debye]`, "defined modulo integer multiples of the cell matrix").
- **Packing:** one exclusive DCGP node (112 cores) runs 4 frames at once (28 MPI ranks each, `srun --exact`, 120 GB each); 30 array tasks; 120 min per-frame timeout; re-running `label` skips finished frames.

**Smoke test** (to run first on Leonardo; one node, three runs side by side on the frame nearest 300 K):
1. the production deck → all five labels present, wall time;
2. the same frame with `STRESS_TENSOR NUMERICAL` → analytical stress must agree within 0.05 GPa (9MA: 0.002 GPa);
3. `RUN_TYPE DEBUG` with `DEBUG_POLARIZABILITY T` and a zero-amplitude `&PERIODIC_EFIELD` → analytical vs finite-field (Berry phase, DE = 5e-4 a.u.) polarizability within 2 %. This also tests LINRES together with the NN10 smoothing; if it fails, `xc_smoothing = false`.

**Label conventions** (`parsed/*.xyz`, written by `collect`): `REF_energy` eV; `REF_forces` eV/Å; `REF_stress` eV/Å³, ASE sign (CP2K printed stress negated; same convention as 9MA npt_al, whose sign was validated against −dE/dV), Voigt order; `REF_dipole` e·Å, Berry phase folded to the branch nearest zero (valid because the cell is centrosymmetric and built from neutral non-polar molecules; frames whose folded dipole exceeds half a quantum get `config_dipole_weight = 0`); `REF_polarizability` **e·Å²/V** (units of MACE-MDP/SPICE-α; 1 e·Å²/V = 14.3996 Å³), 3×3 row-major, symmetrised; `polarizability_A3` the same in Å³; `pressure_GPa` the DFT static pressure; plus provenance (`frame_id`, `config_type`, `split`, `step`, `time_ns`, `T_target`, `volume`, `parent_id`).

**Frame selection — run on the Mac, 2026-10-04 16:11** (`Calculations/02_dft_dataset/frame_selection/`):
- Source: `legacy/mace-off23-small_fixdeform/all_1.lammpstrj` (ramp frames only, steps ≥ 200 000) with the rotor/twist angles from `tbu_rotations.dat` / `tbu_rotations_biphenyl.dat`.
- 25 temperature bins of 10 K (53–300 K), 81 frames per bin (54 in the partial 50–60 K bin). Per bin: half evenly spaced in time, half farthest-point sampling on (cos 3φ, sin 3φ) of the four tBu rotors and (cos φ, sin φ) of the two biphenyl twists; frames of one bin ≥ 250 fs apart. Seed 1.
- Result: **2000 fix-deform frames** (989 even, 1011 FPS), none broken or missing; T 53.3–299.9 K; V 793.4–860.0 Å³. FPS picked up the rare states (biphenyl twists up to ±130°, rotors in transit), see `selection.png`.
- **200 strained cells** (seed 2) from 200 frames spread over T: 150 rigid-molecule (linear isotropic strain −4 % … +5 % plus deviatoric σ 1 %, max 2.5 %) and 50 affine (|ε| ≤ 1.5 %); a strain bringing two molecules closer than 1.75 Å is halved. V 707.8–977.1 Å³ (0.82–1.14 × V_exp); closest intermolecular contact 1.76 Å (unstrained frames reach 1.74 Å).
- Molecules made whole in every frame (2 × 46 atoms, checked); split by 0.2-ns time blocks (block % 10 = 3 → test, = 8 → valid): **train 1772 / valid 224 / test 204**.

**Fine-tuning set-up** (`config.toml` `[train_potential]`, `[train_dielectric]`):
- Potential: `mace_run_train` from MACE-OFF23-small with the 9MA npt_al settings (`--freeze=4`, `--loss=stress`, E/F/σ weights 10/10/1e4, SWA from epoch 400 with 10/100/1e4, lr 1e-3, EMA 0.999, batch 16, 500 epochs, float64, `--E0s=average`), but with our time-block `--valid_file`/`--test_file` instead of a random 10 % split. Exported with `code/tools/create_cp2k_model.py` (Will's exporter, copied from `9MA/cp2k`).
- Dielectric: `AtomicDielectricMACE`, `--loss=dipole_polar`, dipole/polarizability weights 1/10, fine-tuned from **MACE-MDP** (`--finetune_dipoles_polarizabilities`; needs a recent mace-torch — checked in the MACE repository, Sept 2026) or from scratch (`mode = "scratch"`). Note: MACE's dipole includes a Σqᵢrᵢ term, which is why molecules are made whole; the polarizability (sum of atomic tensors) does not depend on positions this way.
- Both jobs evaluate themselves on the test set (energy/atom, forces, stress, pressure + sign check; dipole, α full/isotropic/anisotropic).

**Environment:** on the Mac everything ran in a new venv `~/Claude` inside the Cowork VM (numpy from the system, + tomli, pytest, ase); nothing else installed. On Leonardo the jobs use the existing `cp2k/2024.1` module, `~/mace_env` and `~/cp2k_torch_env` unchanged; `dtbubp` is used from `PYTHONPATH`, not installed.

**Caveats to watch:**
- `evaluate` imports `tomllib` (python ≥ 3.11): if `~/mace_env` is python 3.10, evaluation fails without stopping training (then run it with python 3.11).
- `srun --exact` packing and `--mem=120G` per frame are untested on DCGP (CHECK in the smoke test log).
- The frames carry MACE-OFF23 hydrogen positions (thermalised), not X-ray ones: C–H 1.04–1.15 Å in a 177 K frame.

**Next:** copy to Leonardo and run the smoke test (`Calculations/02_dft_dataset/RUN.md`).


### 2026-10-04 — Correction: the dielectric model is trained from scratch, not fine-tuned from MACE-MDP (Will, Claude)
- **Will:** MACE-MDP cannot be fine-tuned. **Checked and confirmed for released MACE:** in mace-torch ≤ v0.3.16 (latest release, 2026-05-10) `run_train` loads a foundation model only for energy models (MACE, ScaleShiftMACE, MACELES, PolarMACE); an `AtomicDielectricMACE` is always built new. The `--finetune_dipoles_polarizabilities` option I had relied on exists only on the MACE **main branch** (commit 0f3f828, 2026-07-01, not in any release) and is tested there only on a tiny self-trained model, not on the MACE-MDP checkpoint. My earlier plan used it without checking the release history — that was wrong.
- **Change:** `[train_dielectric] mode = "scratch"` (config template and `02_dft_dataset/config.toml`); `finetune_mdp` kept as an option but not recommended. From-scratch architecture (flags verified against v0.3.16): r_max 5.0 Å, 2 interactions, correlation 3, max_ell 3, hidden irreps 64x0e+64x1o+64x2e (l = 2 even features are needed for the polarizability), MLP irreps 16x0e+16x1o+16x2e (the non-linear dipole/polar readout gates the 1o and 2e channels), 8 radial / 5 cutoff basis; lr 1e-2 with plateau scheduler (patience 5), EMA 0.99, batch 8, 500 epochs, SWA from 400; dipole/polarizability weights 1/10; `--E0s=average` kept because run_train requires it.
- Labels keep MACE-MDP units (e·Å²/V) so that MACE-MDP can still be used as a zero-shot baseline in evaluation (not implemented).
- RUN.md: MACE-MDP download removed. Tests: 7 pass.
- Consequence to watch: ~1800 training frames of only 92 atoms (2 molecules) is a small set for a from-scratch tensor model; if the test error on the polarizability anisotropy is poor, add isolated-molecule and dimer polarizabilities (cheap) or more frames.


### 2026-10-04 — Smoke test on Leonardo: PASSED (Will ran it; Claude interpreted)
- Frame nearest 300 K (fix-deform, V ≈ 860 Å³), one DCGP node, three CP2K 2024.1 runs side by side (28 MPI ranks each).
- **All five labels produced** in one ENERGY_FORCE run: energy, forces, analytical stress, Berry dipole, LINRES polarizability. **Wall time 6.7 min** per frame on 28 ranks. The whole job took > 15 min because the numerical-stress run (extra SCF cycles at strained cells) and the DEBUG polarizability run (SCF cycles under finite fields) are much slower; production runs only the label deck.
- **Stress:** analytical vs numerical max |Δσ| = 0.001 GPa (pressure +1.1036 vs +1.1045 GPa) → `STRESS_TENSOR ANALYTICAL` stays.
- **Polarizability:** analytical (LINRES) vs finite field max error 0.32 % → LINRES works with the NN10 smoothing; `xc_smoothing = true` stays.
- **Values:** α diagonal 639.3 / 526.7 / 611.7 a.u. = 94.7 / 78.0 / 90.6 Å³ per cell → α_iso ≈ 87.8 Å³ per cell ≈ 44 Å³ per molecule (plausible: biphenyl ≈ 20 Å³ plus two *tert*-butyl groups). Folded Berry dipole (−0.012, −0.010, 0.135) e·Å, 0.03 of half a quantum → folding unambiguous.
- **Heads-up — DFT static pressure +1.10 GPa at the experimental-volume frame** (CP2K sign: the cell wants to expand), whereas MACE-OFF23 gives about −0.7 GPa there. One frame only; part of it is likely intramolecular (MACE-OFF/ωB97M bond lengths are shorter than BLYP's, which reads as compression to BLYP), plus thermal displacements (the kinetic pressure, ≈ +0.44 GPa for 92 atoms at 300 K, comes on top in MD). If `collect` confirms a large positive pressure over all fix-deform frames, BLYP-D3 will give an NPT cell larger than experiment — the opposite of 9MA, where DFT favoured smaller cells. This matters for the functional benchmark later, not for labelling now.
- **Settings changed for production** (`config.toml` and the template): per-task time limit 24 h → **6 h** (each of the 30 tasks needs ≈ 19 rounds × 4 frames × ~7 min ≈ 2.5 h; shorter jobs schedule faster), per-frame timeout 120 → **40 min**, smoke-test limit 2 h. Cost estimate: 2200 frames × 6.7 min / 4 per node ≈ **60 node-hours**.
- **Next:** `dtb label`.


### 2026-10-05 — DFT labelling finished; MACE training set up (Will, Claude)
- All 2200 frames labelled on Leonardo (30-task array, ~6.7 min per frame on 28 ranks). Node `lrdn4344` had a broken job-container tmpfs and killed tasks 3 and 20 at start-up; both were resubmitted alone with `--exclude=lrdn4344` and completed.
- Code fix: `dtbubp/project.py` falls back to a JSON copy of the config (`.config.json`, written on every TOML read) when python has no TOML parser.
- **Training environment = `~/cp2k_torch_env`** (Will, 2026-10-05): python 3.11.7, mace-torch **0.3.16**, which has `AtomicDielectricMACE` and `--polarizability_key`. `[env] mace` in `config.toml` and the template changed from `~/mace_env` to `cp2k_torch_env` (+ `module load python/3.11.7`). The same env also exports the potential to CP2K.
- Next: `dtb collect` → check `parsed/summary.txt`, `parsed/collect.json` → `dtb train both`.


### 2026-10-05 — Potential r0 trained; fixed-T NPT tests set up (Will ran training; Claude)
- **Potential r0** (`03_mace_finetune/models/potential_r0/dtbubp_pot_r0_stagetwo.model`, 500 epochs, ~2 h on one A100) — final errors (MACE error table, stage-two model):

  | set | E (meV/atom) | F (meV/Å) | rel. F | stress (meV/Å³) |
  |---|---|---|---|---|
  | train | 2.8 | 32.2 | 4.7 % | 0.4 (0.06 GPa) |
  | valid | 1.0 | 14.6 | 2.3 % | 0.3 (0.05 GPa) |
  | test, fix-deform | **6.0** | **63.8** | 9.1 % | 0.7 (**0.11 GPa**) |
  | test, strained affine | 1.0 | 17.3 | 2.2 % | 0.3 |
  | test, strained rigid | 2.5 | 17.2 | 2.5 % | 0.3 |

  Stress is within the 0.15 GPa target everywhere. **Open question:** the fix-deform test frames are 4× worse in forces and 6× in energy than validation, although both are held-out time blocks of the same ramp. Likely a few outlier frames (or a temperature region) in the test blocks — to check from `eval_potential.json/png` (per-frame errors vs T) before trusting the model at all temperatures.
- **Fixed-T NPT tests** (`code/dtbubp/npt.py`, commands `npt-setup` / `npt-analyze`; `Calculations/04_npt_ramp_78-300K/fixedT_r0/` with `config.toml` and `RUN.md`):
  - T = 298, 260, 220, 180, 120, 80 K; 2×2×2 supercell of the Z = 2 cell (736 atoms, 16 molecules); **NPT_F**, 1 bar, CSVR 100 fs, barostat 1 ps, Δt 0.5 fs, 100 ps per T (first 20 ps discarded); cell/energy/stress every 10 fs, positions every 200 fs, restart every 5 ps.
  - Start structure = the fix-deform frame nearest each T (thermalised H, volume near experiment), never the X-ray H positions.
  - Engine = Will's CP2K master + MACE build (`env` = 9MA `env_cp2k_mace.sh` without PLUMED); model linked as `model.pth` in each run folder (80-character file-name limit); volume watchdog 0.80–1.25 × V0.
  - Analysis: mean volume per molecule and unit-cell parameters vs the experimental V(T) (values read off the VT-XRD plot) and the 298 K CIF cell.
  - **Interpretation rule:** only 298 / 260 / 220 K are in the experimental room-T phase; 180 / 120 / 80 K test the metastable room-T structure (the real crystal is Z = 12 / low-T phase there).
- Tests: 8 pass (new: NPT setup + analysis on synthetic CP2K output).


### 2026-10-05 — XRD structures of all three phases found; NPT analysis compares V per molecule phase by phase (Claude)
- `DtBuDp/Structures/` (new; Box online-only, staged to read) holds the single-crystal structures (Olex2/NoSpherA2 refinements):

  | T (K) | phase | space group | Z | a, b, c (Å) | α, β, γ (°) | V_cell (Å³) | **V/molecule (Å³)** | file |
  |---|---|---|---|---|---|---|---|---|
  | 298 | I | P-1 | 2 | 8.2340, 10.1638, 11.8166 | 66.646, 83.975, 71.361 | 860.01 | **430.00** | 298k.cif |
  | 215 (heating) | I | P-1 | 2 | 8.2368, 10.0409, 11.7105 | 67.039, 83.756, 70.345 | 839.54 | **419.77** | 215_heating/5_215.cif (R1 0.042) |
  | 180 | II | P-1 | 12 | 8.1439, 23.1803, 26.5087 | 92.011, 91.075, 95.452 | 4977.33 | **414.78** | 180/1_180.cif (R1 0.031) |
  | 160 | II | P-1 | 12 | 8.1301, 23.1269, 26.4805 | 92.164, 91.157, 95.384 | 4952.06 | **412.67** | (160k.cif, not in folder) |
  | 150 | III | P-1 | **10** | 7.1465, 23.4247, 25.7589 | 108.558, 93.185, 90.609 | 4079.79 | **407.98** | 150/4_150.cif (R1 0.044) |

  The low-T phase (III) is **Z = 10** (P-1, a ≈ 7.15 Å — a different packing, not a superstructure of phase I). The VT cooling curve values (§2.3) are already per molecule and agree with these points (419.7 / 414.7 / 412.7 / 408.0). Phase ranges used in the analysis (approximate, cooling): III < 155 K ≤ II < 212 K ≤ I.
- `npt-analyze` now: (1) compares MD V/molecule (V_box / 16) with the experimental V_cell / Z of the phase present at each T and flags temperatures where experiment is in phase II/III while the MD cell stays phase I; (2) lists every XRD structure with its Z and V/molecule and the MD value at the same T; (3) compares cell parameters only for phase-I temperatures; (4) plot shades the phase ranges and shows the XRD points by Z.
- `npt-setup` now only submits temperatures that have not started (per-submission `runs_<stamp>.txt` / `npt_job_<stamp>.sh`), so 215 / 160 / 150 K can be added to `temperatures` to sit on the XRD points.
- **Folder changes noticed (made outside this log):** the top level now has `DFT/`, `DSC/`, `Raman/`, `Structures/`; `Data/` (with `crystal_structures/298k.cif`), `Figures/`, `Tables/`, `Presentations/`, `Videos/`, `Structure_fitting/` and `Trimorphism.pdf` are no longer there. `DFT/`, `DSC/`, `Raman/` appear empty from here (online-only Box folders may not list). `298k.cif` is not found anywhere in DtBuDp now — to restore or locate (the NPT config keeps its cell values).


### 2026-10-05 — First NPT result with potential r0: the cell collapses by ~12 % (Will ran; Claude)
- `npt-analyze` after ~20 ps per run (2×2×2, NPT_F, 1 bar):

  | T (K) | ⟨T⟩ | exp phase | V/mol MD | V/mol exp | err |
  |---|---|---|---|---|---|
  | 298 | 294.9 | I (Z=2) | 375.2 | 430.0 | −12.7 % |
  | 260 | 256.6 | I | 372.7 | 425.1 | −12.3 % |
  | 220 | 216.6 | I | 373.1 | 420.2 | −11.2 % |
  | 180 | 177.1 | II (Z=12) | 366.4 | 414.7 | −11.7 % |
  | 120 | 118.7 | III (Z=10) | 368.8 | 405.5 | −9.1 % |
  | 80 | 77.9 | III | 369.0 | 400.9 | −8.0 % |

  (Averages over only ~0.1–0.7 ps past the 20 ps discard; trend reliable, digits not.)
- **This is the same collapse as MACE-OFF23 in CP2K (−11 %, 382 Å³/molecule) and is inconsistent with the DFT labels:** the smoke-test frame (V = 430 Å³/molecule) has a DFT static pressure of **+1.10 GPa** (wants to expand), and r0 reproduces DFT stress on held-out frames to ≈0.1 GPa. An MD cell 12 % smaller than experiment would need DFT to be strongly negative there. So either (a) the potential has a spurious dense minimum outside the labelled volumes, or (b) CP2K-MACE does not run the model we think (wrong file/head) or mis-handles its virial.
- New diagnostic `npt-diagnose` (`code/dtbubp/diagnose.py`), Python/ASE only, no CP2K: heads in the .model; DFT vs MACE static pressure on the labelled frames; rigid-molecule E(V)/P(V) scan of the 298 K frame (model's static equilibrium volume); MACE pressure on the last saved frame of every NPT run. If Python-MACE says the MD cells are strongly compressed (large positive P) → engine/export problem; if it says they are at equilibrium → the potential is wrong there (fix with labels from the collapsed cells, i.e. active learning).
- Recommendation: run the diagnostic, then cancel the NPT jobs (they are not usable either way).


### 2026-10-05 — Diagnosis: CP2K-MACE runs r0 correctly; r0's own equilibrium is ~380 Å³/molecule (Claude)
- `readlink T298/model.pth` → `potential_r0/dtbubp_pot_r0_stagetwo.model-cp2k.pth`; export metadata: 2 types (H C), r_max 4.5, float64. Model has one head (`Default`). → the right model is running.
- `npt-diagnose` (GPU job 59355822):
  - **DFT vs MACE static pressure on labelled frames:** fix-deform (396.7–430.0 Å³/mol) DFT +1.896 / MACE +1.895 GPa, RMSE 0.065, corr 0.99; strained-rigid (359–488) +1.784 / +1.774, RMSE 0.028, corr 1.00. **The model reproduces DFT.**
  - **Rigid-molecule scan of the 298 K frame (MACE):** energy minimum at **380 Å³/molecule**; the virial pressure is +2.1 to +5.6 GPa at every volume and never crosses zero.
  - **Last NPT frames, MACE in Python:** static P between −0.05 and −0.74 GPa at 366–380 Å³/mol, i.e. ≈ −(kinetic pressure): **the MD cells sit at r0's equilibrium; CP2K and Python agree.**
- **Interpretation (and a correction of my earlier reading of the smoke test):** the large positive DFT pressures of the labelled frames are **intramolecular**: those frames carry MACE-OFF23 bond lengths, shorter than BLYP's, so BLYP sees every bond compressed. That virial says nothing about the intermolecular equilibrium, and it disappears in MD as soon as the bonds relax to BLYP lengths. The rigid-molecule energy minimum (380) — not the virial — reflects where the lattice wants to be, and it matches the NPT result. The "+1.1 GPa means the cell wants to expand" statement of the smoke-test entry was wrong.
- **Open question:** is ~380 Å³/molecule (−12 % vs experiment) the true BLYP-D3(zero)/DZVP-MOLOPT-SR answer (BSSE of the small basis + D3 overbinding; 9MA DFT also favoured cells a few % smaller than X-ray), or an error of r0 where it has few labels? No labelled frame has BLYP-relaxed molecules at small volume.
- **Decisive test written: `eos-scan` / `eos-dft` / `eos-analyze`** (`code/dtbubp/eos.py`): 8 volumes 350–470 Å³/molecule; at each, rigid scaling of the 298 K frame then fixed-cell relaxation with r0 (so molecules have BLYP-like geometry); DFT single points with the r0 label deck on exactly those structures (8 × ~7 min, one CPU node). Output: DFT vs MACE E(V) and P(V), static P = 0 volume, and the volume where static + kinetic pressure = 0 at 80 and 298 K.
  - If DFT also gives ~380 → the level of theory is the problem: the functional/basis benchmark (skipped so far) becomes necessary (TZV2P or counterpoise, D3(BJ)/D4, other functionals).
  - If DFT gives ~420–430 → r0 is wrong at those volumes: label these frames + NPT frames and retrain (active learning).
- Recommendation: cancel the NPT jobs (`scancel -n dtb_npt`).

### 2026-10-05 — EOS result: BLYP-D3/DZVP itself puts the crystal at ~377 Å³/molecule; r0 is faithful. The level of theory is the problem (Will ran; Claude)
- The first "eos-dft ended after a minute" was the GPU `eos-scan` job; the DFT job had not been submitted yet (no `eos_dft_job.sh` in `eos/`). Submitted `eos-dft` from `cp2k_torch_env`; all 8 single points ended normally.
- `eos-analyze` (E per molecule relative to the minimum, eV; P static, GPa):

| V/mol (Å³) | E_MACE | E_DFT | P_MACE | P_DFT |
|---|---|---|---|---|
| 350 | 0.1478 | 0.1620 | +1.976 | +1.931 |
| 365 | 0.0232 | 0.0399 | +0.730 | +0.695 |
| 380 | 0.0000 | 0.0204 | −0.167 | −0.201 |
| 395 | 0.0463 | 0.0000 | −0.752 | −0.776 |
| 410 | 0.1366 | 0.0957 | −1.130 | −1.149 |
| 430 | 0.2976 | 0.1953 | −1.398 | −1.388 |
| 450 | 0.4737 | 0.2721 | −1.439 | −1.401 |
| 470 | 0.6467 | 0.4611 | −1.348 | −1.306 |

  - Static P = 0: **MACE r0 377.2, DFT 376.6 Å³/molecule**. Static + classical kinetic P = 0: MACE 379.4 (80 K) / 388.2 (298 K); DFT 378.9 / 387.5. Experiment: 400.9 (80 K), 430.0 (298 K).
  - **Pressures: r0 reproduces DFT to ≤ 0.05 GPa over 350–470 Å³/mol.** The NPT collapse is what BLYP-D3(zero)/DZVP-MOLOPT-SR-GTH predicts (−10 % at 298 K, −5.5 % at 80 K static estimate), not a model failure. Active learning would not fix it.
  - **Caveat — DFT energies are not consistent with the DFT pressures at large V.** Integrating P_DFT from 380 to 470 gives ΔE ≈ +0.64 eV/molecule; the DFT energies give +0.44 (MACE: +0.64 from P, +0.65 from E — consistent). Point to point the DFT E(V) scatters by ~0.05 eV/molecule (~1 meV/atom); the DFT E minimum (~390 by a 3-point fit) therefore disagrees with its P = 0 (377). Likely causes: grid-point count changing discretely with the cell at fixed 600 Ry cutoff (energy steps that the analytical stress does not see), and/or residual DFT forces on the MACE-relaxed structures. The pressures are smooth and agree with r0, so the equilibrium volume is taken from P. Possibly related to the 6 meV/atom energy error on the fix-deform test set (to check).
- **Decision point:** the functional/basis benchmark that was skipped is now required. Suspects: BSSE of DZVP-MOLOPT-SR (overbinding → too dense) and D3(zero) overbinding of the many tert-butyl H···H contacts. The NPT thermal expansion is also too small (369 → 375 from 80 to 298 K vs exp 401 → 430); that is a separate issue (anharmonic tert-butyl / phase behaviour) to revisit once the static volume is right.

### 2026-10-05 — Level-of-theory benchmark r1: CP2K cell optimisation, 10 levels, overnight (Claude wrote; Will submits)
- **Why CELL_OPT and not single points:** the EOS structures carry BLYP-relaxed molecules; any other functional would see their bonds as strained and that intramolecular stress would dominate the pressure (same trap as the fix-deform labels). A full cell optimisation (atoms + all six cell parameters, P = 1 bar) relaxes both at each level. Figure of merit: static equilibrium volume per molecule. A good level sits a few % *below* the 80 K experimental value (400.9 Å³/mol), since zero-point and thermal expansion are missing.
- **Code:** `code/dtbubp/bench.py`, `code/templates/cellopt.inp`, `code/config_bench_template.toml` → `Calculations/01_dft_benchmark/config.toml`; CLI `bench-setup [--only a,b] [--dry-run]`, `bench-analyze`; test `tests/test_bench.py` (11/11 pass, `~/Claude` venv).
  - Start structure: `fixedT_r0/eos/V430` (298 K frame at the experimental 298 K volume, molecules relaxed with r0), the same for every level.
  - CP2K 2024.1 module, `RUN_TYPE CELL_OPT`, `DIRECT_CELL_OPT`, BFGS, `PRESSURE_TOLERANCE 100 bar`, `MAX_FORCE 4.5e-4 Ha/Bohr`, `MAX_ITER 300`, `KEEP_SYMMETRY F`; grid/SCF = label deck (600/60 Ry, NN10, EPS_SCF 1e-8). No LINRES (cheap).
  - One exclusive DCGP node (112 MPI ranks) per level, Slurm array `dtb_bench`, 24 h. Cell, trajectory and restart written every step; resubmitting `bench_job.sh` continues an unfinished level from `bench-1.restart` (EXT_RESTART + wavefunction restart).
- **Levels** (each differs from a neighbour by one thing):

| # | level | tests |
|---|---|---|
| 01 | BLYP-D3(zero) / DZVP-MOLOPT-SR | control = r0 label deck (EOS says ~377) |
| 02 | BLYP-D3(zero) / DZVP-MOLOPT | basis (diffuse) |
| 03 | BLYP-D3(zero) / TZV2P-MOLOPT | basis (BSSE) |
| 04 | BLYP-D3(BJ) / TZV2P | damping |
| 05 | BLYP-D3(BJ)+C9 / TZV2P | three-body ATM |
| 06 | PBE-D3(BJ) / TZV2P | functional |
| 07 | revPBE-D3(BJ) / TZV2P | functional |
| 08 | revPBE-D3(BJ)+C9 / TZV2P | three-body ATM |
| 09 | rVV10 / TZV2P | non-local dispersion (libxc; may not run in 2024.1) |
| 10 | PBE-D3(BJ) / DZVP-MOLOPT-SR | basis, PBE |

  - `c9` absent in 01–03 = keyword not written (identical to the labels); 04–10 set `CALCULATE_C9` explicitly.
- **Not yet decided / to check after the run:** whether the winner's cost with LINRES (TZV2P) is affordable for relabelling; the cell parameters (not only V); the DFT energy-vs-pressure inconsistency seen in the EOS (grid steps) — check that the winner's EOS is consistent before relabelling.
- **First submission (array job, 2026-10-05 04:49):** levels 04–08 and 10 aborted at input parsing: `found an unknown keyword CALCULATE_C9 in section PAIR_POTENTIAL` (CP2K 2024.1). The keyword is `CALCULATE_C9_TERM` (default F). Fixed in `bench.py`: c9 = true writes `CALCULATE_C9_TERM T`; c9 = false writes nothing (= default = label deck). Levels 01, 02, 03 and 09 (rVV10 runs in 2024.1) were unaffected and kept running; the six failed levels were resubmitted with `--only`.
- **Interim `bench-analyze` (2026-10-05 morning)** — static V/molecule (Å³), current value for running levels:

| level | status | steps | V/mol | vs 80 K (400.9) | s/step |
|---|---|---|---|---|---|
| 01 BLYP-D3 / DZVP-SR (control) | converged | 245 | 360.4 | −10.1 % | 16 |
| 02 BLYP-D3 / DZVP | running | 151 | 364.0 | −9.2 % | 182 |
| 03 BLYP-D3 / TZV2P | running | 146 | 362.9 | −9.5 % | 191 |
| 04 BLYP-D3(BJ) / TZV2P | running | 147 | 361.7 | −9.8 % | 188 |
| 05 BLYP-D3(BJ)+C9 / TZV2P | running | 146 | 365.5 | −8.8 % | 186 |
| 06 PBE-D3(BJ) / TZV2P | running | 152 | 376.8 | −6.0 % | 182 |
| 07 revPBE-D3(BJ) / TZV2P | running | 142 | 367.2 | −8.4 % | 193 |
| 08 revPBE-D3(BJ)+C9 / TZV2P | running | 143 | 372.1 | −7.2 % | 192 |
| 09 rVV10 / TZV2P | running | 139 | 366.4 | −8.6 % | 199 |
| 10 PBE-D3(BJ) / DZVP-SR | converged | 259 | 370.1 | −7.7 % | 16 |

  - The "FAILED / unknown keyword CALCULATE_C9" shown by `bench-analyze` for 04–08 and 10 was stale: CP2K appends to an existing `output.out`, and the parser looked at the whole file. Fixed: status and errors come from the text after the last `PROGRAM STARTED AT`.
  - Readings so far: basis effect small (BLYP: DZVP-SR → TZV2P +0.7 %; PBE: +1.8 %); D3 zero → BJ ≈ 0; three-body C9 +1.0 % (BLYP), +1.3 % (revPBE); functional matters most: PBE > revPBE ≈ rVV10 > BLYP. **No level is within a few % of experiment;** best PBE-D3(BJ)/TZV2P at −6 % vs the 80 K volume.
  - **Control 01 converged at 360.4, not the EOS's 377:** the EOS scaled the cell rigidly (fixed shape); the full cell optimisation also changes the shape a lot — a 8.23 → 6.80 Å, α 66.6 → 76.4°, γ 71.4 → 78.6° — for every level. The short a ≈ 6.8–7.0 Å resembles the phase III (Z = 10) short axis (7.15 Å at 150 K), i.e. at 0 K the Z = 2 cell relaxes toward a denser, phase-III-like packing. A static Z = 2 optimisation is therefore compared with a phase III experimental volume (80 K is phase III) — roughly the right comparison, but not like for like.
  - Added levels 11 PBE-D3(BJ)+C9 / TZV2P and 12 PBE-D3(BJ)+C9 / DZVP-SR (combining the two expansions seen so far).
  - Cost: TZV2P ≈ 12× DZVP-SR per optimisation step (190 vs 16 s on 112 cores), relevant for relabelling with LINRES.
- **Stopped early to save CPU budget (Will low on IscrB_MET2SAF_0 hours, 2026-10-05).** Plan: cancel every TZV2P / rVV10 level (volumes had plateaued: < 0.3 Å³ change over the last steps, |P| < 0.05 GPa ≈ 0.5 % in V) and level 11; keep only 12 (PBE-D3(BJ)+C9 / DZVP-SR, ~16 s/step, ~1 node-hour). Restart files are kept, so any level can be resumed with `bench-setup --only`. **Provisional choice: PBE-D3(BJ)** (best static volume at both basis sets: 376.6 TZV2P, 370.1 DZVP-SR); C9 to be decided from level 12.
- **Level 06 (PBE-D3(BJ)/TZV2P) at cancellation, step 154:** P = −191 bar (tolerance 100), max gradient 1.6e-3 Ha/Bohr (limit 4.5e-4), energy change −5.8e-5 Ha/step (1.6 meV per cell), last step not downhill: a flat, soft landscape (tBu rotations), not a volume problem. With a molecular-crystal bulk modulus of ~8 GPa the residual −0.019 GPa is ≈ −0.2 % in V: converged volume ≈ 375.7 ± 0.5 Å³/molecule (−6.3 % vs 80 K). Ranking unchanged.


### 2026-10-05 — Move of the DFT work to BlueHive; r1 relabel at PBE-D3(BJ)/TZV2P set up (Will decided; Claude)
- **Why:** Leonardo CPU (DCGP) hours are no longer usable; Leonardo GPUs still are (MACE training/MD stay there). Will has 3 free BlueHive nodes.
- **Final state of the Leonardo benchmark** (`bench-analyze` on the copied `cellopt/` files; levels 02–09 were stopped at ~150 steps when the CPU hours ended, so their volumes are last values, not converged; 11 and 12 never ran):

| level | status | steps | V/mol | vs 80 K |
|---|---|---|---|---|
| 01 BLYP-D3 / DZVP-SR | converged | 245 | 360.4 | −10.1 % |
| 02 BLYP-D3 / DZVP | stopped | 154 | 363.9 | −9.2 % |
| 03 BLYP-D3 / TZV2P | stopped | 149 | 362.7 | −9.5 % |
| 04 BLYP-D3(BJ) / TZV2P | stopped | 150 | 361.6 | −9.8 % |
| 05 BLYP-D3(BJ)+C9 / TZV2P | stopped | 149 | 365.3 | −8.9 % |
| **06 PBE-D3(BJ) / TZV2P** | stopped (P −191 bar) | 155 | **376.4** | **−6.1 %** |
| 07 revPBE-D3(BJ) / TZV2P | stopped | 145 | 366.9 | −8.5 % |
| 08 revPBE-D3(BJ)+C9 / TZV2P | stopped | 145 | 372.0 | −7.2 % |
| 09 rVV10 / TZV2P | stopped | 142 | 366.1 | −8.7 % |
| 10 PBE-D3(BJ) / DZVP-SR | converged | 259 | 370.1 | −7.7 % |
| 11 PBE-D3(BJ)+C9 / TZV2P | never ran | | | |
| 12 PBE-D3(BJ)+C9 / DZVP-SR | never ran | | | |

- **Decisions (Will):** (1) run bench **level 12** (PBE-D3(BJ)+C9 / DZVP-MOLOPT-SR) on one BlueHive node; (2) **relabel the same 2200 r0 frames** (2000 fix-deform + 200 strained, `02_dft_dataset/frame_selection/frames.extxyz`) at **level 06, PBE-D3(BJ) / TZV2P-MOLOPT-GTH / GTH-PBE**, with the full label deck (E, F, analytical stress, Berry dipole, LINRES α), on the other nodes. Campaign folder `Calculations/02_dft_dataset_r1_pbe-d3bj_tzv2p/` (config only; it reuses r0's frame selection).
- **Transfer route (Will: BlueHive login needs Duo):** private GitHub repo `wstoll-ur/dtbubp`. Work tree = this Box folder; Claude pushes from the Cowork VM with a write deploy key; BlueHive clones/pulls with its own write deploy key and pushes small reports back (`bluehive/push_reports.sh` → `bluehive/reports/`). `.gitignore` is a whitelist: code, configs, notes, `frames.extxyz` (12 MB; private repo — an exception to the "no trajectories on GitHub" rule of §6.3, for transfer), `eos/V430` (bench start structure, copied from Leonardo). Not in git: legacy/, label/, smoketest outputs, parsed/, models, CIFs, `cellopt/`, and files the jobs rewrite (LOG.md, state.json, job scripts).
- **Code changes (`dtbubp`):**
  - The label deck's level of theory is now set in `[dft]` (`functional`, `dispersion`, `c9`, `basis`; defaults = r0, so the r0 deck is unchanged). The XC/vdW/basis helpers moved from `bench.py` to `cp2k.py` and are shared by the label and CELL_OPT decks.
  - Cluster-specific launch: `[dft] cp2k_exe` (default `cp2k.popt`) and `mpi_flags` (default `--mpi=pmi2 --cpu-bind=cores`, the Leonardo value); `cp2k.launch()` builds the srun line for label, smoke test and bench.
  - `[slurm.*]` account and qos are optional.
  - Config reading without a TOML parser: each config gets a committed `.<stem>.json` copy with the TOML's sha256; used only if it matches.
  - Tests: 14 pass (new `tests/test_levels.py`: r0 defaults, PBE-D3(BJ)/TZV2P deck, launcher, optional account).
- **BlueHive-specific settings are placeholders marked CHECK** (CP2K module, partition, cores/memory per node, MPI flags, time limit) until `bluehive/probe.sh` reports what BlueHive has. `Calculations/01_dft_benchmark/config_bluehive.toml` = the bench config with `dft_config` → the r1 config (same grid/SCF), ranks and time limit for BlueHive.
- **Cost warning:** TZV2P was ~12× DZVP-SR per CELL_OPT step. r0 labels took 6.7 min per frame on 28 Leonardo ranks, so r1 may take ~1–1.5 h per frame (incl. LINRES) → on the order of 2000+ node-hours for 2200 frames, i.e. weeks on 2 nodes. The r1 smoke test (timing, analytical vs numerical stress, LINRES vs finite field at TZV2P) runs first; then decide whether to label all 2200 frames, a subset first, or drop α from most frames.


### 2026-10-05 (afternoon) — CP2K 2024.1 compiled on the BlueHive Vermont nodes; multi-node set-up (Will ran; Claude)
- **Where the DFT runs:** Will's three free nodes are the lab's Vermont nodes bhx0123–0125 (outside Slurm; §6.5). `probe_vermont.sh`: 24 Broadwell cores, 62 GB, RHEL 7.9, no shared file system with bluehive3, no apptainer. Will decided **not** to use the Slurm `standard` partition and to **compile CP2K**.
- **Dead ends (recorded so they are not retried):** (1) CP2K 2024.1 Apptainer container — pull fails on GPFS (`unpriv.lsetxattr: invalid argument`) and the nodes have no apptainer; (2) the nodes' `cp2k/2025.1` module — SIGILL at start-up (compiled for a newer CPU); (3) conda-forge CP2K and the Slurm test scripts were written (`build_conda_cp2k.sh`, `test_slurm_cp2k.sh`) but not used.
- **Build (`bluehive/build_cp2k.sh fetch|start|log|test|deploy`)**, finished 14:03 on bhx0123 (toolchain + CP2K `make` ≈ 1 h after the fixes); attempts and fixes in order: no `bzip2` on the nodes → source repacked as .tar.gz on bluehive3; the node `module` function returns non-zero on a harmless `unalias sudo` → no `set -e`; loading `python3/3.7.1` broke python 2 → libint's Fortran interface (`libint_f.mod`) not generated → python3 only via a wrapper, incomplete libint removed and rebuilt; the MPI toolchain also builds COSMA/ELPA/HDF5/... (HDF5 needs lbzip2, ELPA python3) → all optional packages switched off; arch-file generation needs python3 → wrapper. Result: `cp2k.psmp` (CP2K 2024.1, OpenMPI 4.1.5, `-D__LIBXSMM -D__parallel -D__MPI_F08 -D__FFTW3 -D__LIBINT -D__LIBXC -D__SCALAPACK`), deployed to bhx0124/0125 (copy via bluehive3 scratch, without build/obj/lib).
- **Tests so far:** CP2K 2024.1 starts on all three nodes, finds its data directory, runs 4 ranks pinned side by side, 1 rank serial, and 12 ranks over 3 nodes. **No calculation has finished yet:** every methane test aborted with *SCF run NOT converged* — a defect of the test input (default OT without outer SCF, 50 steps), not of the build (energy had reached −8.0776 Ha). Fixed input (OT FULL_ALL/DIIS + OUTER_SCF, as the production decks) committed, not yet rerun. Earlier test reports were empty because `mpirun` read the rest of a heredoc sent over ssh → test scripts now run as files with stdin closed.
- **Multi-node (Will: "string all three nodes together" for the level-12 cell optimisation):** `multinode.sh check` — node↔node ssh OK, 3-node OpenMPI start OK (after forwarding `PATH`/`LD_LIBRARY_PATH` with `-x`, remote ranks have no login environment), network 10 GbE, no InfiniBand. The 12-rank methane run over 3 nodes took 1 min 40 s wall with sys > user time — a sign that TCP communication may dominate. **Decision pending on `multinode.sh timing`:** one real level-12 step (92 atoms, PBE-D3(BJ)+C9/DZVP-MOLOPT-SR, energy + forces + analytical stress) on 24 ranks/1 node vs 72 ranks/3 nodes; the cell optimisation goes to whichever is faster (Leonardo: 16 s/step on 112 cores).
- **Code (`dtbubp`, 17 tests pass):** `[dft] launcher = "mpirun"` (local, `--cpu-set` lanes) and multi-node (`hosts`, `mpi_prefix`, `mpi_extra`, `-x` forwarding); `bench` passes `[bench] hosts`; `vermont.sh run` accepts a node list (folder copied to every node, started on the first). r1 config: 2 frames × 12 ranks per node, `source $HOME/Claude/cp2k-2024.1.env`; bench config `config_bluehive.toml`: 72 ranks on bhx0123–0125.
- **Timing result (Will, `multinode.sh timing`, level-12 single point, first SCF iterations, identical energies on both runs → the build is numerically consistent across nodes):** 24 ranks / 1 node ≈ 10–16 s per SCF iteration (mean ≈ 13 s); 72 ranks / 3 nodes ≈ 13–20 s (mean ≈ 17 s). **3 nodes over 10 GbE are slower → one node per calculation** (`config_bluehive.toml`: ranks 24, `hosts` commented out). Cost implication: ~13 s per SCF iteration at DZVP-SR on 24 Broadwell cores is ~10× slower than Leonardo (whole CELL_OPT step 16 s on 112 cores); TZV2P labels (~12× DZVP-SR per step, plus LINRES) will cost hours per frame per node — the r1 smoke test must give the number before committing 2200 frames. Possible speed-up: rebuild CP2K's own code with `-march=haswell` (the toolchain arch file only set `-mtune=haswell`; the libraries are AVX2-optimised already).
- **Correction — bhx0123 is shared (found when level 12 ran there at ~20 s per SCF iteration):** load ≈ 31 with only our 24 ranks visible (`top`: 33 tasks; other users' processes are hidden from us), 15.8 GB swap in use, 2 users logged in; it already showed load ≈ 6.6 with nothing of ours at 11:31. **The timing comparison above was therefore confounded:** the "one node" run (13 s) and the 3-node run both included the busy bhx0123. Level 12 moved to **bhx0124: ≈ 5 s per SCF iteration** (iterations 2–5: 4.5–5.5 s; same energies) — ~2.5× faster than the "one node" timing, ~4× faster than on busy bhx0123. Multi-node over 0124+0125 was not retested (10 GbE; one clean node per calculation is the plan). **bhx0123 is to be treated as part-time until the other user is identified (ask the lab: Vermont CRYSTAL jobs launch the same way).** Revised cost: ~5 s/SCF iteration at DZVP-SR on one clean node (Leonardo ≈ 1 s on 112 cores).
- **Next:** bench level 12 on bhx0124 → r1 smoke test on bhx0125 (PBE-D3(BJ)/TZV2P: timing per frame, analytical vs numerical stress, LINRES vs finite field) → r1 labelling of the 2200 frames on the free nodes.

---

### 2026-10-05 — Level 12 warm start from level 06 prepared (Codex; Will to run on BlueHive)
- Will requested restarting the running Vermont level-12 optimisation from level 06's latest saved geometry to reduce optimisation time.
- Added `bluehive/start_level12_from06.sh`: self-contained cell vectors and 92 Cartesian atomic positions extracted from the local `06_pbe_d3bj_tzv2p/bench-1.restart` (saved step 154, not fully converged; 376.3772 Å³/molecule).
- Retains level 12 PBE-D3(BJ)+C9 / DZVP-MOLOPT-SR-GTH settings and a fresh ATOMIC SCF guess. Creates a separate timestamped `01_dft_benchmark/from06_*/` directory to prevent the existing level-12 restart from overriding the seed. Uses 24 MPI ranks on bhx0124, following the latest cluster update.
- Verified preparation in an isolated temporary repository, atom count, target electronic settings, absence of inherited restart files, and shell syntax. Will stops the existing registered run and launches the replacement after pulling GitHub; Codex has not changed the remote running job.
- GitHub workflow (Will's preference): Codex maintains a separate checkout outside Box, copies only intended changes into it, and commits/pushes scripts so BlueHive can pull. Check remote changes before each push; do not overwrite newer Claude or BlueHive updates.

## 10. References

- [CP2K-POLAR] CP2K manual, FORCE_EVAL/PROPERTIES/LINRES/POLAR (`DO_RAMAN`, `PERIODIC_DIPOLE_OPERATOR` = Berry phase) — https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/PROPERTIES/LINRES/POLAR.html
- [CP2K-MOMENTS] CP2K manual, FORCE_EVAL/DFT/PRINT/MOMENTS (`PERIODIC T` = Berry phase; jumps during MD) — https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/DFT/PRINT/MOMENTS.html
- [CP2K-VIB] CP2K manual, VIBRATIONAL_ANALYSIS (`INTENSITIES`: IR/Raman with MOMENTS and LINRES/POLAR) — https://manual.cp2k.org/cp2k-2024_1-branch/CP2K_INPUT/VIBRATIONAL_ANALYSIS.html
- [CP2K-2020] Kühne et al., "CP2K: An electronic structure and molecular dynamics software package — Quickstep", J. Chem. Phys. 152, 194103 (2020) — https://pubs.aip.org/aip/jcp/article/152/19/194103/199081
- [MACE-releases] ACEsuit/mace releases (AtomicDielectricMACE, `dipole_polar` loss, multihead fine-tuning with replay, LoRA, exports) — https://github.com/ACEsuit/mace/releases
- [MACE-MDP] "MACE-MDP: A General Dipole and Polarizability Model for Organic Molecules and Materials", ChemRxiv — https://chemrxiv.org/doi/10.26434/chemrxiv.15000716 ; SPICE-alpha: https://zenodo.org/records/19205036
- [MACE-POLAR-1] https://arxiv.org/abs/2602.19411
- [Acene-PRB] "Investigating anharmonicities in polarization-orientation Raman spectra of acene crystals with machine learning", Phys. Rev. B — https://journals.aps.org/prb/abstract/10.1103/t4s2-45t8 (FHI-aims PBE+TS polarizabilities, SA-GPR α model, MACE dynamics)
- [Raman-ML-review] "Machine learning accelerates Raman computations from molecular dynamics for materials science", J. Chem. Phys. 163, 120901 — https://pubs.aip.org/aip/jcp/article/163/12/120901/3365046
- [Pol-models] "Polarizability models for simulations of finite temperature Raman spectra from machine learning molecular dynamics" — https://www.researchgate.net/publication/379800917
- MACE-OFF23: Kovács et al., "MACE-OFF: Short-Range Transferable Machine Learning Force Fields for Organic Molecules" — full citation to add.
- 9MA project internal docs: `9MA/LOG.md`, `9MA/PROJECT_CONTEXT.md`, `9MA/cp2k/cp2k_mace_leonardo_handoff.md`, `9MA/cp2k/9MA_mace_cp2k_test/README.md`, `9MA/cp2k/npt_al/README.md`.
