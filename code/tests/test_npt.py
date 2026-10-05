"""NPT test set-up and analysis with fake frames, a fake model file and a fake CP2K cell/energy output."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from dtbubp import geometry as geo, npt
from dtbubp.io import read_xyz, write_extxyz_frame
from dtbubp.project import Campaign

from test_pipeline import read_lmp, HERE, CODE


def make_campaign(tmp_path):
    cell0, types, pos0 = read_lmp(HERE / "data" / "cell298.lmp")
    syms = ["C" if t == 1 else "H" for t in types]
    mols = geo.molecules(syms, pos0, cell0)
    pos0 = geo.make_whole(syms, pos0, cell0, mols)
    root = tmp_path / "Calculations"
    fs = root / "02_dft_dataset" / "frame_selection"
    fs.mkdir(parents=True)
    with open(fs / "frames.extxyz", "w") as f:
        for k, T in enumerate([300.0, 262.0, 181.0, 79.0]):
            write_extxyz_frame(f, syms, pos0, cell0 * (1 - 0.0001 * (300 - T)), None,
                               {"frame_id": f"f{k:04d}", "config_type": "fixdeform", "T_target": T})
    m = root / "03_mace_finetune" / "models" / "potential_r0"
    m.mkdir(parents=True)
    (m / "dtbubp_pot_r0_stagetwo.model-cp2k.pth").write_text("fake")
    d = root / "04_npt_ramp_78-300K" / "fixedT_r0"
    d.mkdir(parents=True)
    (d / "config.toml").write_text((CODE / "config_npt_template.toml").read_text())
    return Campaign(d / "config.toml"), cell0


def test_npt_setup_and_analyze(tmp_path):
    c, cell0 = make_campaign(tmp_path)
    sc = npt.setup(c, dry=True)
    assert subprocess.run(["bash", "-n", str(sc)]).returncode == 0
    s = sc.read_text()
    assert "--array=0-5" in s and "--gres=gpu:1" in s and "EXIT" in s and "runs_" in s and "plumed-2.10.0" in s
    runs = sorted(c.root.glob("runs_*.txt"))[-1].read_text().split()
    assert runs == ["T298", "T260", "T220", "T180", "T120", "T080"]
    rd = c.root / "T260"
    inp = (rd / "md.inp").read_text()
    assert "ENSEMBLE NPT_F" in inp and "TEMPERATURE 260" in inp and "POT_FILE_NAME model.pth" in inp
    assert "STEPS 200000" in inp and "{{" not in inp and "ATOMS H C" in inp
    assert (rd / "model.pth").is_symlink()
    meta = json.loads((rd / "meta.json").read_text())
    assert meta["n_atoms"] == 736 and meta["n_molecules"] == 16 and meta["source_frame"] == "f0001"
    assert abs(meta["V_exp_per_molecule"] - 425.1) < 0.01
    _, syms, pos = next(read_xyz(rd / "start.xyz"))
    cellS = np.array([[float(x) for x in ln.split()[1:4]] for ln in inp.splitlines() if ln.strip()[:2] in ("A ", "B ", "C ")])
    assert abs(geo.cell_volume(cellS) - 8 * geo.cell_volume(cell0 * (1 - 0.0001 * 38))) < 1e-6
    # supercell: 16 whole molecules, nothing overlapping
    mols = geo.molecules(syms, pos, cellS)
    assert sorted(len(m) for m in mols) == [46] * 16
    assert geo.min_intermolecular_distance(pos, cellS, mols) > 1.5
    # fake CP2K output: 60 ps of cell rows, volume settles to 1.01 x V0
    t = np.arange(0, 60001, 10.0)
    V0 = meta["V0"]
    V = V0 * (1 + 0.01 * (1 - np.exp(-t / 3000)))
    s3 = (V / V0) ** (1 / 3)
    with open(rd / "md-1.cell", "w") as f:
        f.write("#   Step   Time [fs]  Ax Ay Az Bx By Bz Cx Cy Cz Volume\n")
        for k, (tt, sc_, vv) in enumerate(zip(t, s3, V)):
            row = (cellS * sc_).ravel()
            f.write(f"{k * 20} {tt:.3f} " + " ".join(f"{x:.10f}" for x in row) + f" {vv:.10f}\n")
    with open(rd / "md-1.ener", "w") as f:
        f.write("# Step Nr. Time[fs] Kin.[a.u.] Temp[K] Pot.[a.u.] Cons Qty[a.u.] UsedTime[s]\n")
        for k, tt in enumerate(t):
            f.write(f"{k * 20} {tt:.3f} 1.0 {260 + np.sin(k):.3f} -1.0 -1.0 0.1\n")
    # a second setup submits nothing new for started temperatures, but does for an added one
    (c.root / "T298" / "md-1.cell").write_text("# started\n")
    cfg = (c.root / "config.toml").read_text().replace("temperatures         = [298,", "temperatures         = [215, 298,")
    (c.root / "config.toml").write_text(cfg)
    c2 = Campaign(c.root / "config.toml")
    sc2 = npt.setup(c2, dry=True)
    lists = sorted(c.root.glob("runs_*.txt"))
    assert sc2.name.startswith("npt_job_") and sc2 != sc
    assert len(lists) == 2
    new = lists[-1].read_text().split()
    assert "T215" in new and "T298" not in new and "T260" not in new
    (c.root / "T298" / "md-1.cell").unlink()
    out = npt.analyze(c)
    r = out["T260"]
    assert abs(r["V_per_molecule"] - 1.01 * V0 / 16) < 0.05 * V0 / 16 * 0.1
    assert abs(r["a"] - np.linalg.norm(cellS[0]) / 2 * 1.01 ** (1 / 3)) < 0.01
    assert abs(r["T_mean"] - 260) < 1
    txt = (c.root / "npt_summary.txt").read_text()
    assert "Z = 12" in txt or "Z=12" in txt
    assert "Structures/150/4_150.cif" in txt and "407.98" in txt and "414.78" in txt
    assert out["T260"]["exp_Z"] == 2
    assert npt.phase_of(c, 180)["Z"] == 12 and npt.phase_of(c, 120)["Z"] == 10 and npt.phase_of(c, 298)["Z"] == 2


def test_diagnose_with_fake_calculator(tmp_path, monkeypatch):
    """diagnose.run end to end with an ASE Lennard-Jones calculator standing in for MACE."""
    from ase.calculators.lj import LennardJones
    from dtbubp import diagnose
    c, cell0 = make_campaign(tmp_path)
    npt.setup(c, dry=True)
    # labels.xyz with a DFT pressure, as written by collect
    fr = next(iter(__import__("dtbubp.io", fromlist=["read_extxyz"]).read_extxyz(
        c.root / "../../02_dft_dataset/frame_selection/frames.extxyz")))
    lab = c.root / "../../02_dft_dataset/parsed"
    lab.mkdir(parents=True)
    with open(lab / "labels.xyz", "w") as f:
        for k in range(6):
            write_extxyz_frame(f, fr["symbols"], fr["pos"], fr["cell"] * (1 + 0.01 * k), None,
                               {"config_type": "fixdeform", "pressure_GPa": 0.1 * k, "T_target": 300.0})
    # one NPT run with a position file and a cell file
    rd = c.root / "T298"
    _, syms, pos = next(read_xyz(rd / "start.xyz"))
    inp = (rd / "md.inp").read_text()
    cellS = np.array([[float(x) for x in ln.split()[1:4]] for ln in inp.splitlines() if ln.strip()[:2] in ("A ", "B ", "C ")])
    with open(rd / "md-pos-1.xyz", "w") as f:
        for i in (0, 400):
            f.write(f"{len(syms)}\n i = {i}, time = {i * 0.5:.3f}, E = -1.0\n" +
                    "".join(f"{s} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n" for s, x in zip(syms, pos)))
    with open(rd / "md-1.cell", "w") as f:
        f.write("# Step Time Ax Ay Az Bx By Bz Cx Cy Cz Volume\n")
        for i in (0, 400):
            f.write(f"{i} {i * 0.5} " + " ".join(f"{x:.8f}" for x in cellS.ravel()) + f" {geo.cell_volume(cellS):.6f}\n")
    monkeypatch.setattr(diagnose, "_calc", lambda c_, d: (LennardJones(sigma=1.0, epsilon=0.001, rc=3.0),
                                                           {"model": "fake", "heads": ["Default"], "atomic_numbers": [1, 6], "r_max": 4.5}))
    rep = diagnose.run(c, stride=1)
    assert len(rep["labels"]) == 6 and len(rep["ev_scan"]) == 14 and "T298" in rep["npt_last"]
    assert (c.root / "diagnose.txt").exists()


def test_eos_with_fake_calculator(tmp_path, monkeypatch):
    from ase.calculators.lj import LennardJones
    from dtbubp import diagnose, eos
    c, cell0 = make_campaign(tmp_path)
    (c.root / "../../02_dft_dataset/config.toml").write_text((CODE / "config_template.toml").read_text())
    monkeypatch.setattr(diagnose, "_calc", lambda c_, d: (LennardJones(sigma=1.0, epsilon=0.0005, rc=2.5),
                                                           {"model": "fake", "heads": ["Default"], "atomic_numbers": [1, 6], "r_max": 4.5}))
    cfg = (c.root / "config.toml").read_text().replace("max_steps      = 500", "max_steps      = 3")
    (c.root / "config.toml").write_text(cfg)
    c = Campaign(c.root / "config.toml")
    eos.scan(c, device="cpu")
    d = c.root / "eos"
    names = (d / "frames.txt").read_text().split()
    assert names == ["V350", "V365", "V380", "V395", "V410", "V430", "V450", "V470"]
    inp = (d / "V380" / "input.inp").read_text()
    assert "STRESS_TENSOR ANALYTICAL" in inp and "&POLAR" in inp
    meta = json.loads((d / "V380" / "meta.json").read_text())
    assert abs(geo.cell_volume(np.array(meta["cell"])) / 2 - 380) < 1e-6
    eos.dft(c, dry=True)
    assert subprocess.run(["bash", "-n", str(d / "eos_dft_job.sh")]).returncode == 0
    from test_pipeline import fake_cp2k_output
    rng = np.random.default_rng(0)
    for k, n in enumerate(names):
        v = float(n[1:])
        P = 0.02 * (400 - v)                                  # DFT crosses zero at 400 A^3/molecule
        fake_cp2k_output(d / n / "output.out", 92, rng, cell=np.array(json.loads((d / n / "meta.json").read_text())["cell"]),
                         stress_gpa=np.eye(3) * P)
    out = eos.analyze(c)
    assert abs(out["DFT"]["V_static_P0"] - 400) < 1e-6
    assert out["DFT"]["V_P_static_eq_minus_Pkin_298K"] > 400
