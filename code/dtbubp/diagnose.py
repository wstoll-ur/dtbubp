"""Why does the NPT cell go where it goes? Static checks of the MACE potential in Python (ASE), no CP2K.

  1. which head / elements the .model has (multi-head models: CP2K exports the last head);
  2. DFT vs MACE static pressure on the labelled frames (by config_type, vs volume);
  3. rigid-molecule E(V) / P(V) scan of a 298 K frame with MACE -> the static equilibrium volume of the model,
     next to the DFT pressures of the labelled frames at the same volumes;
  4. MACE pressure (Python) on the last saved frame of each NPT run -> if Python MACE says the MD cell is far
     from equilibrium while CP2K kept it there, the CP2K engine/export is the problem; if Python MACE agrees,
     the potential itself has the dense minimum.
Run inside cp2k_torch_env (ase + mace-torch), from the NPT folder:  dtbn npt-diagnose
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import geometry as geo, units
from .cp2k import from_voigt
from .io import read_extxyz, read_table, read_xyz
from .project import Campaign


def _calc(c: Campaign, device: str):
    import torch
    from mace.calculators import MACECalculator
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    mp = c.cfg["paths"].get("model_mace")
    model = c.path(mp) if mp else Path(str(c.p("model_cp2k")).replace("-cp2k.pth", ""))
    m = torch.load(str(model), map_location="cpu", weights_only=False)
    info = {"model": str(model), "heads": list(getattr(m, "heads", [None])),
            "atomic_numbers": [int(z) for z in m.atomic_numbers], "r_max": float(m.r_max)}
    return MACECalculator(model_paths=str(model), device=device, default_dtype="float64"), info


def _atoms(syms, pos, cell):
    from ase import Atoms
    return Atoms(symbols=syms, positions=pos, cell=cell, pbc=True)


def _p_gpa(stress_voigt_or_33) -> float:
    S = np.asarray(stress_voigt_or_33, float)
    S = from_voigt(S) if S.size != 9 else S.reshape(3, 3)
    return float(-np.trace(S) / 3 * units.EV_A3_GPA)


def run(c: Campaign, device: str = "cpu", stride: int = 4) -> dict:
    base = c.p("run_dir")
    calc, info = _calc(c, device)
    rep = {"model": info}
    lines = [f"model: {info['model']}", f"  heads: {info['heads']}   (CP2K export uses the LAST head unless --head is given)",
             f"  elements Z: {info['atomic_numbers']}   r_max {info['r_max']} A", ""]

    # ---- 2. labelled frames: DFT vs MACE static pressure
    lab = c.path(c.cfg["paths"].get("labels", "../../02_dft_dataset/parsed/labels.xyz"))
    rows = []
    for k, fr in enumerate(read_extxyz(lab)):
        if k % stride:
            continue
        a = _atoms(fr["symbols"], fr["pos"], fr["cell"])
        a.calc = calc
        pm = _p_gpa(a.get_stress(voigt=True))
        rows.append((fr["info"].get("config_type", "?"), geo.cell_volume(fr["cell"]) / (len(a) / 46),
                     float(fr["info"]["pressure_GPa"]), pm, float(fr["info"].get("T_target", np.nan))))
    rep["labels"] = rows
    lines.append(f"DFT vs MACE static pressure on labelled frames (every {stride}th):")
    lines.append(f"  {'config_type':<16} {'n':>4} {'V/mol':>13} {'P_DFT GPa':>10} {'P_MACE GPa':>11} {'RMSE':>6} {'corr':>5}")
    for ct in sorted({r[0] for r in rows}):
        R = np.array([r[1:4] for r in rows if r[0] == ct])
        corr = np.corrcoef(R[:, 1], R[:, 2])[0, 1] if len(R) > 2 else np.nan
        lines.append(f"  {ct:<16} {len(R):4d} {R[:, 0].min():6.1f}-{R[:, 0].max():6.1f} {R[:, 1].mean():+10.3f} "
                     f"{R[:, 2].mean():+11.3f} {np.sqrt(((R[:, 1] - R[:, 2]) ** 2).mean()):6.3f} {corr:5.2f}")
    lines.append("  (CP2K sign: positive = the cell wants to expand. Static: no kinetic term, which adds ~+0.4 GPa at 300 K.)")
    lines.append("")

    # ---- 3. rigid-molecule E(V) scan of the 298 K frame
    frames = [f for f in read_extxyz(c.p("frames")) if f["info"]["config_type"] == "fixdeform"]
    fr = min(frames, key=lambda f: abs(f["info"]["T_target"] - 298))
    mols = geo.molecules(fr["symbols"], fr["pos"], fr["cell"])
    scan = []
    for s in np.linspace(0.92, 1.05, 14):
        eps = (s - 1) * np.eye(3)
        pos, cell = geo.strain_rigid_molecules(fr["pos"], fr["cell"], mols, fr["symbols"], eps)
        a = _atoms(fr["symbols"], pos, cell)
        a.calc = calc
        scan.append((geo.cell_volume(cell) / 2, a.get_potential_energy() / 2, _p_gpa(a.get_stress(voigt=True))))
    scan = np.array(scan)
    rep["ev_scan"] = scan.tolist()
    i0 = int(np.argmin(scan[:, 1]))
    sign = np.sign(scan[:, 2])
    cross = [k for k in range(len(scan) - 1) if sign[k] != sign[k + 1]]
    v_p0 = float(np.interp(0, scan[::-1, 2], scan[::-1, 0])) if cross else None
    lines.append(f"Rigid-molecule scan of {fr['info']['frame_id']} (T_ramp {fr['info']['T_target']:.0f} K), MACE, per molecule:")
    lines.append(f"  {'V/mol':>7} {'E/mol (eV)':>12} {'P (GPa)':>8}")
    for v, e, p in scan:
        lines.append(f"  {v:7.1f} {e - scan[i0, 1]:12.4f} {p:+8.3f}")
    lines.append(f"  -> MACE static minimum near V/mol = {scan[i0, 0]:.1f}; P = 0 at {v_p0 if v_p0 is None else round(v_p0, 1)} A^3"
                 f"   (exp 298 K: 430.0; static DFT should sit a few % below the 298 K value)")
    lines.append("")

    # ---- 4. last frame of every NPT run, MACE in Python
    lines.append("Last saved NPT frame of each run, MACE static pressure in Python (should be about -kinetic, i.e. ~ -0.3..-0.4 GPa,")
    lines.append("if CP2K and Python agree and the run sits at equilibrium):")
    for rd in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("T")):
        pf, cf = rd / "md-pos-1.xyz", rd / "md-1.cell"
        if not (pf.exists() and cf.exists()):
            continue
        last = None
        for comment, syms, xyz in read_xyz(pf):
            last = (comment, syms, xyz)
        if last is None:
            continue
        m = re.search(r"i\s*=\s*(\d+)", last[0])
        C = read_table(cf)
        row = C[np.argmin(np.abs(C[:, 0] - int(m.group(1))))] if m else C[-1]
        cell = row[2:11].reshape(3, 3)
        a = _atoms(last[1], last[2], cell)
        a.calc = calc
        pm = _p_gpa(a.get_stress(voigt=True))
        nmol = len(a) // 46
        rep.setdefault("npt_last", {})[rd.name] = {"V_per_molecule": row[-1] / nmol, "P_MACE_python_GPa": pm}
        lines.append(f"  {rd.name}: step {int(row[0])}, V/mol {row[-1] / nmol:6.1f}, MACE(python) static P {pm:+.3f} GPa")
    txt = "\n".join(lines)
    (base / "diagnose.txt").write_text(txt + "\n")
    (base / "diagnose.json").write_text(json.dumps(rep, indent=1, default=float))
    print(txt)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for ct in sorted({r[0] for r in rows}):
            R = np.array([r[1:4] for r in rows if r[0] == ct])
            ax[0].scatter(R[:, 1], R[:, 2], s=6, label=ct)
        lo, hi = ax[0].get_xlim()
        ax[0].plot([lo, hi], [lo, hi], "k-", lw=0.6)
        ax[0].set(xlabel="DFT static P (GPa)", ylabel="MACE static P (GPa)", title="labelled frames")
        ax[0].legend(fontsize=7)
        R = np.array([r[1:4] for r in rows])
        ax[1].scatter(R[:, 0], R[:, 1], s=5, color="0.6", label="DFT, labelled frames")
        ax[1].plot(scan[:, 0], scan[:, 2], "C3o-", label="MACE, rigid scan of 298 K frame")
        ax[1].axhline(0, color="k", lw=0.5)
        ax[1].axvline(430.0, color="C0", ls=":", label="exp 298 K")
        ax[1].set(xlabel="V per molecule (A^3)", ylabel="static P (GPa)")
        ax[1].legend(fontsize=7)
        fig.savefig(base / "diagnose.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)
    return rep
