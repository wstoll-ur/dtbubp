"""Model vs DFT on the held-out test set (and valid set). Needs ase + mace-torch (Leonardo mace_env).

potential:  energy/atom (after removing the mean offset), forces, stress, pressure; by split and by
            temperature block; parity plots; and the sign check (MACE and DFT pressures must correlate).
dielectric: dipole and polarizability (full tensor, isotropic part, anisotropy) RMSE; parity plots.
Results: <model dir>/eval_<split>.json and eval_<kind>.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import units
from .cp2k import from_voigt
from .project import Campaign


def _load(path: Path):
    from ase.io import read
    return read(path, index=":")


def potential(c: Campaign, model: Path, device: str = "cuda") -> dict:
    import torch
    from mace.calculators import MACECalculator
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    calc = MACECalculator(model_paths=str(model), device=device, default_dtype="float64")
    res = {}
    plots = {}
    for split in ("test", "valid"):
        p = c.p("parsed_dir") / f"{split}.xyz"
        if not p.exists():
            continue
        frames = _load(p)
        eD, eM, dF, dS, pD, pM, T = [], [], [], [], [], [], []
        for a in frames:
            Ed, Fd, Sd = float(a.info["REF_energy"]), a.arrays["REF_forces"], from_voigt(a.info["REF_stress"])
            a.calc = calc
            Em, Fm, Sm = a.get_potential_energy(), a.get_forces(), a.get_stress(voigt=False)
            eD.append(Ed / len(a)); eM.append(Em / len(a))
            dF.append((Fm - Fd).ravel()); dS.append((Sm - Sd).ravel())
            pD.append(-np.trace(Sd) / 3 * units.EV_A3_GPA); pM.append(-np.trace(Sm) / 3 * units.EV_A3_GPA)
            T.append(a.info.get("T_target", np.nan))
        de = np.array(eM) - np.array(eD)
        dF, dS, pD, pM = np.concatenate(dF), np.concatenate(dS), np.array(pD), np.array(pM)
        corr = float(np.corrcoef(pD, pM)[0, 1]) if len(pD) > 2 else float("nan")
        r = {"frames": len(frames),
             "energy_rmse_meV_atom": float(np.sqrt((de ** 2).mean()) * 1000),
             "energy_rmse_meV_atom_after_offset": float((de - de.mean()).std() * 1000),
             "force_rmse_meV_A": float(np.sqrt((dF ** 2).mean()) * 1000),
             "stress_rmse_GPa": float(np.sqrt((dS ** 2).mean()) * units.EV_A3_GPA),
             "pressure_rmse_GPa": float(np.sqrt(((pM - pD) ** 2).mean())),
             "pressure_mean_DFT_GPa": float(pD.mean()), "pressure_mean_MACE_GPa": float(pM.mean()),
             "pressure_correlation": corr}
        if corr < 0:
            r["SIGN_ERROR"] = "MACE and DFT pressures anti-correlated: check the stress sign convention"
        res[split] = r
        plots[split] = (np.array(eD), np.array(eM), pD, pM, np.array(T))
    out = Path(model).parent
    (out / "eval_potential.json").write_text(json.dumps(res, indent=1))
    c.log("**evaluate potential**", model=model, **{k: json.dumps(v) for k, v in res.items()})
    _plot_pot(out / "eval_potential.png", plots)
    return res


def _plot_pot(path, plots):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for split, (eD, eM, pD, pM, T) in plots.items():
        ax[0].scatter(eD, eM - (eM - eD).mean(), s=4, label=split)
        ax[1].scatter(pD, pM, s=4, label=split)
    for a, lab in zip(ax, ("energy per atom (eV, offset removed)", "static pressure (GPa)")):
        lo, hi = a.get_xlim()
        a.plot([lo, hi], [lo, hi], "k-", lw=0.6)
        a.set(xlabel=f"DFT {lab}", ylabel=f"MACE {lab}")
        a.legend()
    fig.savefig(path, dpi=150)


def dielectric(c: Campaign, model: Path, device: str = "cuda") -> dict:
    import torch
    from mace.calculators import MACECalculator
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    calc = MACECalculator(model_paths=str(model), device=device, default_dtype="float64",
                          model_type="DipolePolarizabilityMACE")
    res, plots = {}, {}
    for split in ("test", "valid"):
        p = c.p("parsed_dir") / f"{split}.xyz"
        if not p.exists():
            continue
        frames = [a for a in _load(p) if "REF_polarizability" in a.info]
        mD, mM, aD, aM = [], [], [], []
        for a in frames:
            a.calc = calc
            calc.calculate(a)
            mM.append(np.asarray(calc.results["dipole"]).ravel())
            aM.append(np.asarray(calc.results["polarizability"]).reshape(3, 3))
            mD.append(np.asarray(a.info.get("REF_dipole", [np.nan] * 3)))
            aD.append(np.asarray(a.info["REF_polarizability"]).reshape(3, 3))
        mD, mM, aD, aM = map(np.array, (mD, mM, aD, aM))
        isoD, isoM = np.trace(aD, axis1=1, axis2=2) / 3, np.trace(aM, axis1=1, axis2=2) / 3
        anD = aD - isoD[:, None, None] * np.eye(3)
        anM = aM - isoM[:, None, None] * np.eye(3)
        ok = ~np.isnan(mD).any(1)
        r = {"frames": len(frames),
             "dipole_rmse_eA": float(np.sqrt(((mM[ok] - mD[ok]) ** 2).mean())) if ok.any() else None,
             "polar_rmse_eA2V": float(np.sqrt(((aM - aD) ** 2).mean())),
             "polar_rmse_A3": float(np.sqrt(((aM - aD) ** 2).mean()) / units.A3_TO_EA2V),
             "polar_iso_rmse_A3": float(np.sqrt(((isoM - isoD) ** 2).mean()) / units.A3_TO_EA2V),
             "polar_aniso_rmse_A3": float(np.sqrt(((anM - anD) ** 2).mean()) / units.A3_TO_EA2V),
             "polar_aniso_std_DFT_A3": float(anD.std() / units.A3_TO_EA2V)}
        res[split] = r
        plots[split] = (aD, aM)
    out = Path(model).parent
    (out / "eval_dielectric.json").write_text(json.dumps(res, indent=1))
    c.log("**evaluate dielectric**", model=model, **{k: json.dumps(v) for k, v in res.items()})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for split, (aD, aM) in plots.items():
            iD = np.trace(aD, axis1=1, axis2=2) / 3 / units.A3_TO_EA2V
            iM = np.trace(aM, axis1=1, axis2=2) / 3 / units.A3_TO_EA2V
            ax[0].scatter(iD, iM, s=4, label=split)
            off = [(0, 1), (0, 2), (1, 2)]
            ax[1].scatter(np.concatenate([aD[:, i, j] for i, j in off]) / units.A3_TO_EA2V,
                          np.concatenate([aM[:, i, j] for i, j in off]) / units.A3_TO_EA2V, s=4, label=split)
        for a, lab in zip(ax, ("alpha_iso (A^3)", "alpha off-diagonal (A^3)")):
            lo, hi = a.get_xlim()
            a.plot([lo, hi], [lo, hi], "k-", lw=0.6)
            a.set(xlabel=f"DFT {lab}", ylabel=f"MACE {lab}")
            a.legend()
        fig.savefig(out / "eval_dielectric.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)
    return res
