"""Equation of state of the room-T (Z = 2) crystal: MACE r0 vs DFT on the SAME relaxed structures.

Question it answers: is the dense NPT cell (~375 A^3/molecule, exp 430) what BLYP-D3/DZVP really predicts,
or a defect of the MACE potential? The fix-deform labels cannot tell: their molecules carry MACE-OFF23
bond lengths, which BLYP sees as compressed, so their DFT pressure (~+1.9 GPa) is mostly intramolecular.

  eos-scan     (GPU python job, minutes) start from the fix-deform frame nearest 298 K; for each target
               volume per molecule: scale the cell with rigid molecules, relax the atoms at fixed cell with
               MACE r0 (BFGS), record MACE energy and static pressure, write the structure + CP2K label deck.
  eos-dft      (one CPU node) DFT single points (the r0 label settings) on those structures.
  eos-analyze  DFT vs MACE: E(V), P(V), static equilibrium volume (P = 0) and the volume where the static
               pressure balances the kinetic pressure at 298 K (what NPT would give, roughly).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import cp2k, geometry as geo, units
from .io import read_extxyz, read_xyz, write_xyz
from .project import Campaign
from .slurm import submit, write_script


def _dir(c: Campaign) -> Path:
    d = c.p("run_dir") / "eos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dft_campaign(c: Campaign) -> Campaign:
    return Campaign(c.p("dft_config"))


def scan(c: Campaign, device: str = "cuda"):
    from ase import Atoms
    from ase.optimize import BFGS
    from .diagnose import _calc
    e = c.cfg["eos"]
    calc, info = _calc(c, device)
    frames = [f for f in read_extxyz(c.p("frames")) if f["info"]["config_type"] == "fixdeform"]
    fr = min(frames, key=lambda f: abs(f["info"]["T_target"] - 298))
    syms = fr["symbols"]
    mols = geo.molecules(syms, fr["pos"], fr["cell"])
    nmol = len(mols)
    d = _dir(c)
    dft = _dft_campaign(c).cfg["dft"]

    def relax(pos, cell):
        a = Atoms(symbols=syms, positions=pos, cell=cell, pbc=True)
        a.calc = calc
        BFGS(a, logfile=None).run(fmax=e["fmax"], steps=e["max_steps"])
        return a

    a0 = relax(fr["pos"], fr["cell"])                     # MACE-r0 (BLYP-like) intramolecular geometry
    pos0 = geo.make_whole(syms, a0.get_positions(), fr["cell"], mols)
    V0 = geo.cell_volume(fr["cell"]) / nmol
    rows = []
    for v in e["v_per_molecule"]:
        s = (v / V0) ** (1 / 3)
        pos, cell = geo.strain_rigid_molecules(pos0, fr["cell"], mols, syms, (s - 1) * np.eye(3))
        a = relax(pos, cell)
        E = a.get_potential_energy() / nmol
        S = a.get_stress(voigt=False)
        P = float(-np.trace(S) / 3 * units.EV_A3_GPA)
        name = f"V{int(round(v)):03d}"
        rd = d / name
        rd.mkdir(exist_ok=True)
        pw = geo.make_whole(syms, a.get_positions(), cell, mols)
        write_xyz(rd / "coord.xyz", syms, pw, f"eos {name}: rigid scale of {fr['info']['frame_id']} + MACE r0 fixed-cell relax")
        (rd / "input.inp").write_text(cp2k.make_input(dft, cell, "label"))
        (rd / "meta.json").write_text(json.dumps({"V_per_molecule": v, "n_molecules": nmol, "cell": cell.tolist(),
                                                  "E_mace_eV_per_molecule": E, "P_mace_GPa": P,
                                                  "fmax_reached": float(np.abs(a.get_forces()).max())}, indent=1))
        rows.append((v, E, P))
        print(f"{name}: V/mol {v:6.1f}  E_MACE/mol {E:.4f} eV  P_MACE {P:+.3f} GPa")
    (d / "frames.txt").write_text("\n".join(f"V{int(round(r[0])):03d}" for r in rows) + "\n")
    c.log("**EOS scan (MACE r0)**: relaxed rigid-molecule scan written", dir=d, model=info["model"],
          rows=[(round(v, 1), round(p, 3)) for v, _, p in rows])


def dft(c: Campaign, dry: bool = False):
    from .label import _pool_body
    dc = _dft_campaign(c)
    d = _dir(c)
    if not (d / "frames.txt").exists():
        raise SystemExit("run eos-scan first")
    sc = write_script(dc.cfg["slurm"]["cpu"], d / "eos_dft_job.sh", "dtb_eosdft", _pool_body(dc, d, "frames.txt"),
                      "04:00:00", d)
    jid = submit(sc, dry=dry)
    c.log("**EOS DFT submitted**", job=jid, dir=d)


def _cross(v, p, target):
    """Volume where p(v) = target (linear interpolation on the first crossing), or None."""
    for k in range(len(v) - 1):
        if (p[k] - target) * (p[k + 1] - target) <= 0 and p[k] != p[k + 1]:
            return float(v[k] + (target - p[k]) * (v[k + 1] - v[k]) / (p[k + 1] - p[k]))
    return None


def analyze(c: Campaign) -> dict:
    d = _dir(c)
    rows = []
    for name in (d / "frames.txt").read_text().split():
        rd = d / name
        m = json.loads((rd / "meta.json").read_text())
        r = cp2k.parse_output(rd / "output.out")
        pd = ed = None
        if r["ended"] and r["stress"] is not None and r["energy"] is not None:
            S = cp2k.cp2k_stress_to_ase(r["stress"], r["stress_unit"])
            pd = float(-np.trace(S) / 3 * units.EV_A3_GPA)
            ed = r["energy"] * units.HA_EV / m["n_molecules"]
        rows.append((m["V_per_molecule"], m["E_mace_eV_per_molecule"], m["P_mace_GPa"], ed, pd))
    R = np.array([[x if x is not None else np.nan for x in r] for r in rows], float)
    nat_per_mol = 46
    out = {}
    lines = [f"{'V/mol':>7} {'E_MACE':>9} {'E_DFT':>9} {'P_MACE':>8} {'P_DFT':>8}   (E per molecule, eV, relative to the minimum; P static, GPa)"]
    em, ed = R[:, 1] - np.nanmin(R[:, 1]), R[:, 3] - np.nanmin(R[:, 3])
    for k, r in enumerate(R):
        lines.append(f"{r[0]:7.1f} {em[k]:9.4f} {ed[k]:9.4f} {r[2]:+8.3f} {r[4]:+8.3f}")
    for lab, col in (("MACE r0", 2), ("DFT", 4)):
        ok = ~np.isnan(R[:, col])
        if ok.sum() < 2:
            continue
        v, p = R[ok, 0], R[ok, col]
        res = {"V_static_P0": _cross(v, p, 0.0)}
        for T in (80, 298):
            pk = nat_per_mol * units.KB_EV * T / v * units.EV_A3_GPA       # classical kinetic pressure, all atoms
            res[f"V_P_static_eq_minus_Pkin_{T}K"] = _cross(v, p + pk, 0.0)
        out[lab] = res
        lines.append(f"{lab}: static P = 0 at V/mol = {res['V_static_P0']};  static P + kinetic P = 0 at "
                     f"80 K: {res['V_P_static_eq_minus_Pkin_80K']}, 298 K: {res['V_P_static_eq_minus_Pkin_298K']}  (exp 298 K: 430.0)")
    txt = "\n".join(lines)
    (d / "eos_summary.txt").write_text(txt + "\n")
    (d / "eos_summary.json").write_text(json.dumps({"rows": R.tolist(), "fits": out}, indent=1))
    print(txt)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        ax[0].plot(R[:, 0], em, "C3o-", label="MACE r0")
        ax[0].plot(R[:, 0], ed, "ks--", label="DFT (BLYP-D3/DZVP)")
        ax[0].set(xlabel="V per molecule (A^3)", ylabel="E per molecule (eV, rel.)")
        ax[1].plot(R[:, 0], R[:, 2], "C3o-", label="MACE r0")
        ax[1].plot(R[:, 0], R[:, 4], "ks--", label="DFT")
        ax[1].axhline(0, color="k", lw=0.5)
        ax[1].axvline(430, color="C0", ls=":", label="exp 298 K")
        ax[1].set(xlabel="V per molecule (A^3)", ylabel="static P (GPa)")
        for a in ax:
            a.legend(fontsize=8)
        fig.savefig(d / "eos_summary.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)
    c.log("**EOS analysed**", summary=d / "eos_summary.txt")
    return out
