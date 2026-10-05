"""End-to-end test with synthetic inputs (no CP2K, no Slurm, no MACE).

  fake fix-deform dump (the real 92-atom 298 K cell, thermal noise, wrapped, atom order shuffled,
  a repeated step as `dump_modify append` leaves) + fake rotor .dat files
  -> select (frames, strained copies, splits, whole molecules)
  -> label setup (dry run: inputs + array script pass `bash -n`)
  -> fake CP2K 2024.1 outputs (energy, forces, stress, Berry dipole, POLAR) -> collect
  -> labels in MACE units, splits, stress sign, dipole folding, polarizability units.
Run:  python -m pytest -q tests
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dtbubp import cp2k, geometry as geo, units
from dtbubp.io import cell_from_bounds, iter_lammps_dump, read_extxyz
from dtbubp.project import Campaign

HERE = Path(__file__).resolve().parent
CODE = HERE.parent


def read_lmp(path):
    t = path.read_text().splitlines()
    lo = {}
    for ln in t:
        p = ln.split()
        if ln.endswith("xlo xhi"):
            lo["x"] = (float(p[0]), float(p[1]))
        if ln.endswith("ylo yhi"):
            lo["y"] = (float(p[0]), float(p[1]))
        if ln.endswith("zlo zhi"):
            lo["z"] = (float(p[0]), float(p[1]))
        if ln.endswith("xy xz yz"):
            xy, xz, yz = map(float, p[:3])
    cell = np.array([[lo["x"][1] - lo["x"][0], 0, 0], [xy, lo["y"][1] - lo["y"][0], 0], [xz, yz, lo["z"][1] - lo["z"][0]]])
    i = t.index(" Atoms # atomic")
    rows = [ln.split() for ln in t[i + 2:] if ln.strip()]
    types = np.array([int(r[1]) for r in rows])
    pos = np.array([[float(x) for x in r[2:5]] for r in rows])
    origin = np.array([lo["x"][0], lo["y"][0], lo["z"][0]])
    return cell, types, pos - origin


def write_dump(path, frames, cell0):
    rng = np.random.default_rng(3)
    with open(path, "w") as f:
        for step, cell, types, pos in frames:
            xy, xz, yz = cell[1, 0], cell[2, 0], cell[2, 1]
            xlo, ylo, zlo = 0.0, 0.0, 0.0
            xhi, yhi, zhi = cell[0, 0], cell[1, 1], cell[2, 2]
            xlb, xhb = xlo + min(0, xy, xz, xy + xz), xhi + max(0, xy, xz, xy + xz)
            ylb, yhb = ylo + min(0, yz), yhi + max(0, yz)
            fr = pos @ np.linalg.inv(cell)
            pos = (fr - np.floor(fr)) @ cell                     # wrapped, molecules broken
            order = rng.permutation(len(pos))
            f.write(f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(pos)}\n")
            f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
            f.write(f"{xlb} {xhb} {xy}\n{ylb} {yhb} {xz}\n{zlo} {zhi} {yz}\nITEM: ATOMS id type x y z\n")
            for i in order:
                f.write(f"{i + 1} {types[i]} {pos[i, 0]:.8g} {pos[i, 1]:.8g} {pos[i, 2]:.8g}\n")


def fake_cp2k_output(path, nat, rng, dip_shift_cells=(0, 0, 0), cell=None, stress_gpa=None, pol_au=None):
    F = rng.normal(0, 0.01, (nat, 3))
    S = stress_gpa if stress_gpa is not None else rng.normal(0, 0.3, (3, 3))
    S = 0.5 * (S + S.T)
    mu_true = rng.normal(0, 0.05, 3)                                # e*A, small
    mu = mu_true + np.array(dip_shift_cells) @ cell                 # Berry branch shifted by lattice vectors
    mu_D = mu / units.DEBYE_EA
    P = pol_au if pol_au is not None else np.diag([480.0, 470.0, 455.0]) + rng.normal(0, 3, (3, 3))
    lines = [" DBCSR| ...", " ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:          -301.123456789012",
             "", " ATOMIC FORCES in [a.u.]", "", " # Atom   Kind   Element          X              Y              Z"]
    for i in range(nat):
        lines.append(f"  {i + 1:5d}  {1 if i % 2 else 2:5d}      C   {F[i, 0]:14.8f} {F[i, 1]:14.8f} {F[i, 2]:14.8f}")
    lines += [" SUM OF ATOMIC FORCES           0.0 0.0 0.0     0.0", "",
              " STRESS| Analytical stress tensor [GPa]",
              " STRESS|                        x                   y                   z"]
    for lab, row in zip("xyz", S):
        lines.append(f" STRESS|      {lab}   {row[0]:19.12f} {row[1]:19.12f} {row[2]:19.12f}")
    lines += ["  Reference Point [Bohr]           0.00000000      0.00000000      0.00000000",
              "  Dipole vectors are based on the periodic (Berry phase) operator.",
              "  They are defined modulo integer multiples of the cell matrix [Debye].",
              "  Dipole moment [Debye]",
              f"    X={mu_D[0]:15.7E} Y={mu_D[1]:15.7E} Z={mu_D[2]:15.7E}     Total=   {np.linalg.norm(mu_D):13.7f}",
              "", " POLAR| Polarizability tensor [a.u.]",
              f" POLAR| xx,yy,zz           {P[0, 0]:18.12f} {P[1, 1]:18.12f} {P[2, 2]:18.12f}",
              f" POLAR| xy,xz,yz           {P[0, 1]:18.12f} {P[0, 2]:18.12f} {P[1, 2]:18.12f}",
              f" POLAR| yx,zx,zy           {P[1, 0]:18.12f} {P[2, 0]:18.12f} {P[2, 1]:18.12f}",
              "  *** SCF run converged in    14 steps ***",
              " CP2K                                 1  1.0    0.120    0.135 1234.567 1234.789",
              "", "  **** **** ******  **  PROGRAM ENDED AT                 2026-10-05 10:00:00.000"]
    path.write_text("\n".join(lines) + "\n")
    return F, S, mu_true, P


@pytest.fixture()
def camp(tmp_path):
    cell0, types, pos0 = read_lmp(HERE / "data" / "cell298.lmp")
    leg = tmp_path / "Calculations" / "legacy" / "fd"
    leg.mkdir(parents=True)
    rng = np.random.default_rng(0)
    nfr, steps = 400, []
    frames = []
    for k in range(nfr):
        step = 200000 + 100 * k
        s = (1 - 0.03 * k / nfr)
        cell = cell0 * s
        pos = (pos0 @ np.linalg.inv(cell0)) @ cell + rng.normal(0, 0.02, pos0.shape)
        frames.append((step, cell, types, pos))
        steps.append(step)
    frames.insert(200, frames[199])                      # repeated step (dump append)
    write_dump(leg / "all_1.lammpstrj", frames, cell0)
    t_ns = (np.array(steps) - 200000) * 0.5e-6
    T = 300 - 250 * t_ns / 10.0 * 50                     # compress the ramp so the fake covers many bins
    ang = rng.uniform(-180, 180, (nfr, 4))
    bip = rng.normal(40, 8, (nfr, 2))
    np.savetxt(leg / "tbu.dat", np.column_stack([steps, t_ns, T, ang]), header="step time_ns T_target_K a b c d")
    np.savetxt(leg / "bip.dat", np.column_stack([steps, t_ns, T, bip]), header="step time_ns T_target_K b1 b2")
    d = tmp_path / "Calculations" / "02_dft_dataset"
    d.mkdir()
    cfg = (CODE / "config_template.toml").read_text()
    cfg = cfg.replace("../legacy/mace-off23-small_fixdeform/all_1.lammpstrj", "../legacy/fd/all_1.lammpstrj")
    cfg = cfg.replace("../legacy/mace-off23-small_fixdeform/tbu_rotations.dat", "../legacy/fd/tbu.dat")
    cfg = cfg.replace("../legacy/mace-off23-small_fixdeform/tbu_rotations_biphenyl.dat", "../legacy/fd/bip.dat")
    cfg = cfg.replace("n_frames      = 2000", "n_frames      = 60").replace("n_frames      = 200", "n_frames      = 12")
    cfg = cfg.replace("block_ns     = 0.2", "block_ns     = 0.002")
    cfg = cfg.replace("chunks            = 30", "chunks            = 3")
    (d / "config.toml").write_text(cfg)
    return Campaign(d / "config.toml"), cell0


def test_units():
    assert abs(units.HA_BOHR_EV_A - 51.42206748) < 1e-6
    assert abs(units.AU_POL_A3 - 0.148184711) < 1e-8
    assert abs(1 / units.A3_TO_EA2V - 14.3996454) < 1e-6
    assert abs(1 / units.DEBYE_EA - 4.80320471) < 1e-7


def test_cell_from_bounds_roundtrip():
    cell = np.array([[8.234, 0, 0], [3.248395, 9.63072, 0], [1.240299, 4.525164, 10.845118]])
    xy, xz, yz = cell[1, 0], cell[2, 0], cell[2, 1]
    lines = [f"{0 + min(0, xy, xz, xy + xz)} {8.234 + max(0, xy, xz, xy + xz)} {xy}",
             f"{0 + min(0, yz)} {9.63072 + max(0, yz)} {xz}", f"0 10.845118 {yz}"]
    assert np.allclose(cell_from_bounds(lines, True), cell)


def test_fold_dipole():
    cell = np.array([[8.234, 0, 0], [3.248395, 9.63072, 0], [1.240299, 4.525164, 10.845118]])
    mu = np.array([0.03, -0.02, 0.01])
    for n in ([1, 0, 0], [0, -1, 2], [3, 1, -1]):
        f, r = geo.fold_dipole(mu + np.array(n) @ cell, cell)
        assert np.allclose(f, mu, atol=1e-10) and r < 0.05


def test_inputs(camp):
    c, cell0 = camp
    for kind in ("label", "stress_numerical", "debug_polar"):
        inp = cp2k.make_input(c.cfg["dft"], cell0, kind)
        assert "{{" not in inp and "&KIND C" in inp and "&KIND H" in inp and "&KIND N" not in inp
        if kind == "label":
            assert "RUN_TYPE ENERGY_FORCE" in inp and "STRESS_TENSOR ANALYTICAL" in inp
            assert "&POLAR" in inp and "PERIODIC_DIPOLE_OPERATOR  T" in inp and "PERIODIC   T" in inp
            assert "NN10_SMOOTH" in inp and "CUTOFF      600" in inp and "DFTD3" in inp
        if kind == "stress_numerical":
            assert "STRESS_TENSOR NUMERICAL" in inp and "&POLAR" not in inp
        if kind == "debug_polar":
            assert "RUN_TYPE DEBUG" in inp and "DEBUG_POLARIZABILITY  T" in inp and "&PERIODIC_EFIELD" in inp
        lines = [ln for ln in inp.splitlines() if ln.strip()[:2] in ("A ", "B ", "C ")]
        assert len(lines) == 3


def test_full_pipeline(camp, tmp_path):
    c, cell0 = camp
    from dtbubp import label, select
    summ = select.run(c)
    assert summ["fixdeform_frames"] == 60 and summ["strained_frames"] == 12
    assert set(summ["split_counts"]) == {"train", "valid", "test"} and summ["split_counts"]["test"] > 0
    frames = list(read_extxyz(c.p("selection_dir") / "frames.extxyz"))
    mols = json.loads((c.p("selection_dir") / "molecules.json").read_text())
    assert sorted(len(m) for m in mols) == [46, 46]
    for fr in frames:
        # whole molecules: every bond short WITHOUT the minimum image
        for m in mols:
            x = fr["pos"][m]
            d = np.linalg.norm(x[:, None] - x[None], axis=-1)
            assert (np.sort(d, axis=1)[:, 1] < 1.7).all()
    st = [f for f in frames if f["info"]["config_type"].startswith("strained")]
    assert any(f["info"]["config_type"] == "strained_affine" for f in st)
    assert all(f["info"]["min_contact_A"] >= 1.75 for f in st)
    # labelling (dry run)
    label.setup(c, dry=True)
    lab = c.root / "label"
    assert len([p for p in lab.iterdir() if p.is_dir()]) == len(frames)
    r = subprocess.run(["bash", "-n", str(lab / "label_job.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "--array=0-2" in (lab / "label_job.sh").read_text()
    chunks = sum(len((lab / f"chunk_{k}.txt").read_text().split()) for k in range(3))
    assert chunks == len(frames)
    # fake CP2K outputs; one unfinished, one SCF failure
    rng = np.random.default_rng(5)
    truth = {}
    dirs = sorted(p for p in lab.iterdir() if p.is_dir())
    for k, fd in enumerate(dirs):
        meta = json.loads((fd / "meta.json").read_text())
        cell = np.array(meta["cell"])
        truth[fd.name] = fake_cp2k_output(fd / "output.out", 92, rng, dip_shift_cells=(k % 3 - 1, 1, 0), cell=cell)
    (dirs[0] / "output.out").write_text("partial\n")
    t = (dirs[1] / "output.out").read_text()
    (dirs[1] / "output.out").write_text(t.replace("PROGRAM ENDED", "SCF run NOT converged\n PROGRAM ENDED"))
    st = label.collect(c)
    assert st["kept"] == len(frames) - 2 and st["not_finished"] == 1 and st["scf_failed"] == 1
    out = list(read_extxyz(c.p("parsed_dir") / "labels.xyz"))
    for fr in out:
        F, S, mu, P = truth[fr["info"]["frame_id"]]
        assert np.allclose(fr["arrays"]["REF_forces"], F * units.HA_BOHR_EV_A, atol=1e-6)
        assert abs(fr["info"]["REF_energy"] - (-301.123456789012 * units.HA_EV)) < 1e-6
        # ASE sign: CP2K +p (wants to expand) -> negative ASE stress
        Sv = fr["info"]["REF_stress"]
        assert np.allclose(Sv[:3], -np.diag(S) / units.EV_A3_GPA, atol=1e-9)
        assert abs(fr["info"]["pressure_GPa"] - np.trace(S) / 3) < 1e-6
        assert np.allclose(fr["info"]["REF_dipole"], mu, atol=1e-6)
        Ps = 0.5 * (P + P.T)
        assert np.allclose(np.asarray(fr["info"]["REF_polarizability"]).reshape(3, 3), Ps * units.AU_POL_EA2V, rtol=1e-8)
        assert np.allclose(np.asarray(fr["info"]["polarizability_A3"]).reshape(3, 3), Ps * units.AU_POL_A3, rtol=1e-8)
    n_split = sum(len(list(read_extxyz(c.p("parsed_dir") / f"{s}.xyz"))) for s in ("train", "valid", "test"))
    assert n_split == len(out)
    # ASE reads our extxyz and finds what MACE needs
    ase_io = pytest.importorskip("ase.io")
    a = ase_io.read(c.p("parsed_dir") / "train.xyz", index=0)
    assert a.arrays["REF_forces"].shape == (92, 3) and len(a.info["REF_stress"]) == 6
    assert np.asarray(a.info["REF_polarizability"]).size == 9 and len(a.info["REF_dipole"]) == 3


def test_smoketest_and_train_scripts(camp):
    c, cell0 = camp
    from dtbubp import label, select, train
    select.run(c)
    label.smoketest(c, dry=True)
    sm = c.root / "smoketest"
    assert subprocess.run(["bash", "-n", str(sm / "smoke_job.sh")]).returncode == 0
    rng = np.random.default_rng(1)
    fake_cp2k_output(sm / "label" / "output.out", 92, rng, cell=cell0, stress_gpa=np.eye(3) * 0.4)
    fake_cp2k_output(sm / "stress_numerical" / "output.out", 92, rng, cell=cell0, stress_gpa=np.eye(3) * 0.41)
    dbg = ["DEBUG|========================= POLARIZABILITY ================================",
           "DEBUG| Coordinates     P(numerical)    P(analytical)    Difference    Error [%]"]
    A = np.diag([480.0, 470, 455])
    for i, a in enumerate("xyz"):
        for j, b in enumerate("xyz"):
            dbg.append(f"          {a}{b}      {A[i, j] * 1.004:16.8f} {A[i, j]:16.8f}  {0.0:12.3g}  {0.4:9.3f}")
    (sm / "debug_polar" / "output.out").write_text("\n".join(dbg) + "\n")
    assert label.smoketest_check(c, sm)
    for what in ("potential", "dielectric"):
        getattr(train, what)(c, dry=True)
    for d in (c.p("potential_dir"), c.p("dielectric_dir")):
        s = (d / "train_job.sh").read_text()
        assert subprocess.run(["bash", "-n", str(d / "train_job.sh")]).returncode == 0
        assert "--valid_file=" in s and "--test_file=" in s and "valid_fraction" not in s
    assert "--stress_key=REF_stress" in (c.p("potential_dir") / "train_job.sh").read_text()
    s = (c.p("dielectric_dir") / "train_job.sh").read_text()
    assert "--finetune_dipoles_polarizabilities" not in s and "--polarizability_key=REF_polarizability" in s
    assert "--model=AtomicDielectricMACE" in s and "64x2e" in s and "16x2e" in s


def test_dump_reader_sorts_and_dedups(camp):
    c, cell0 = camp
    steps = [fr["step"] for fr in iter_lammps_dump(c.path(c.cfg["select"]["dump"]))]
    assert len(steps) == 401 and len(set(steps)) == 400
    fr = next(iter_lammps_dump(c.path(c.cfg["select"]["dump"])))
    assert (np.diff(fr["ids"]) > 0).all()
