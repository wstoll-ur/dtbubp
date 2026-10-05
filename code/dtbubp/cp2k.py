"""CP2K input decks (from templates/label.inp; level of theory from [dft] functional/dispersion/c9/basis) and output parsers (CP2K 2024.1 and master layouts).

What a label run produces, and where it is parsed from:

| quantity | CP2K section | output lines parsed |
|---|---|---|
| energy | (always) | `ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:` (last one) |
| forces | FORCE_EVAL/PRINT/FORCES | `ATOMIC FORCES in [a.u.]` table (2024.1) or `FORCES|` lines (master) |
| stress | STRESS_TENSOR ANALYTICAL + PRINT/STRESS_TENSOR | `STRESS| Analytical stress tensor [GPa]` (or `[bar]`) |
| dipole | DFT/PRINT/MOMENTS PERIODIC T (Berry phase) | `Dipole moment [Debye]` then `X= .. Y= .. Z= ..` |
| polarizability | PROPERTIES/LINRES/POLAR (DO_RAMAN T, Berry-phase dipole operator) | `POLAR| Polarizability tensor [a.u.]` + 3 rows |

LINRES runs post-SCF inside ENERGY_FORCE (qs_energies_properties -> linres_calculation_low in
CP2K 2024.1, src/qs_energy_utils.F), so one run per frame gives all five.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import units

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"

_NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"


def _f(x: str) -> float:
    return float(x.replace("D", "E").replace("d", "e"))


def fill(template: str, **kw) -> str:
    out = template
    for k, v in kw.items():
        out = out.replace("{{" + k + "}}", str(v))
    left = re.findall(r"\{\{\w+\}\}", out)
    if left:
        raise ValueError(f"unfilled template keys: {left}")
    return out


def cell_block(cell: np.ndarray) -> str:
    return "\n".join(f"      {n} {v[0]:16.10f} {v[1]:16.10f} {v[2]:16.10f}" for n, v in zip("ABC", np.asarray(cell)))


# ============================================================================ level of theory
# Shared by the label deck (label.inp) and the benchmark (cellopt.inp). A "level" is a dict with
# functional (BLYP | PBE | revPBE | rVV10), dispersion (D3 | D3BJ | none), c9 (bool), basis (MOLOPT name).
# Defaults = the r0 label level (BLYP-D3(zero) / DZVP-MOLOPT-SR-GTH), so old configs give the old deck.
LEVEL_DEFAULTS = {"functional": "BLYP", "dispersion": "D3", "c9": False, "basis": "DZVP-MOLOPT-SR-GTH"}
PSEUDO = {"BLYP": "BLYP", "PBE": "PBE", "revPBE": "PBE", "rVV10": "PBE"}


def level_of(d: dict) -> dict:
    return {k: d.get(k, v) for k, v in LEVEL_DEFAULTS.items()}


def level_name(lv: dict) -> str:
    lv = level_of(lv)
    disp = {"D3": "-D3(zero)", "D3BJ": "-D3(BJ)", "none": ""}[lv["dispersion"]]
    return f"{lv['functional']}{disp}{'+C9' if lv['c9'] else ''} / {lv['basis']} / GTH-{PSEUDO[lv['functional']]}"


def xc_functional(f: str) -> str:
    if f in ("BLYP", "PBE"):
        return f"      &XC_FUNCTIONAL {f}\n      &END XC_FUNCTIONAL"
    if f == "revPBE":
        return ("      &XC_FUNCTIONAL\n        &PBE\n          PARAMETRIZATION REVPBE\n        &END PBE\n"
                "      &END XC_FUNCTIONAL")
    if f == "rVV10":                         # rPW86 exchange + PBE correlation (libxc) + rVV10 non-local kernel
        return ("      &XC_FUNCTIONAL\n        &GGA_X_RPW86\n        &END GGA_X_RPW86\n"
                "        &GGA_C_PBE\n        &END GGA_C_PBE\n      &END XC_FUNCTIONAL")
    raise ValueError(f"unknown functional {f}")


def vdw(lv: dict) -> str:
    lv = level_of(lv)
    f, disp = lv["functional"], lv["dispersion"]
    if f == "rVV10":
        return ("      &VDW_POTENTIAL\n        POTENTIAL_TYPE NON_LOCAL\n        &NON_LOCAL\n          TYPE RVV10\n"
                "          PARAMETERS 6.3 0.0093\n          KERNEL_FILE_NAME rVV10_kernel_table.dat\n"
                "        &END NON_LOCAL\n      &END VDW_POTENTIAL")
    if disp == "none":
        return ""
    typ = {"D3": "DFTD3", "D3BJ": "DFTD3(BJ)"}[disp]
    # CP2K 2024.1 keyword is CALCULATE_C9_TERM (default F, so c9 = false/absent = the r0 label deck);
    # "CALCULATE_C9" is rejected as an unknown keyword (bench r1, 2026-10-05).
    c9_line = "\n          CALCULATE_C9_TERM     T" if lv["c9"] else ""
    return ("      &VDW_POTENTIAL\n        DISPERSION_FUNCTIONAL PAIR_POTENTIAL\n        &PAIR_POTENTIAL\n"
            f"          TYPE                  {typ}\n          PARAMETER_FILE_NAME   dftd3.dat\n"
            f"          REFERENCE_FUNCTIONAL  {f}{c9_line}\n        &END PAIR_POTENTIAL\n      &END VDW_POTENTIAL")


def xc_grid(dft: dict) -> str:
    return ("      &XC_GRID\n        XC_DERIV       NN10_SMOOTH\n        XC_SMOOTH_RHO  NN10\n      &END XC_GRID"
            if dft.get("xc_smoothing", True) else "")


def make_input(dft: dict, cell: np.ndarray, kind: str = "label") -> str:
    """kind = label (E, F, analytical stress, dipole, polarizability)
            | stress_numerical (E, F, numerical stress only; smoke test)
            | debug_polar (RUN_TYPE DEBUG: analytical vs finite-field polarizability; smoke test)
    """
    lv = level_of(dft)
    moments = ("      &MOMENTS\n        PERIODIC   T\n        REFERENCE  ZERO\n        MAX_MOMENT 1\n      &END MOMENTS")
    lr = dft["linres"]
    properties = (
        "  &PROPERTIES\n    &LINRES\n"
        f"      PRECONDITIONER  {lr['preconditioner']}\n"
        f"      EPS             {lr['eps']}\n"
        f"      MAX_ITER        {lr['max_iter']}\n"
        "      &POLAR\n        DO_RAMAN                  T\n        PERIODIC_DIPOLE_OPERATOR  T\n"
        "      &END POLAR\n    &END LINRES\n  &END PROPERTIES")
    debug, efield, run_type, stress = "", "", "ENERGY_FORCE", "ANALYTICAL"
    if kind == "stress_numerical":
        moments, properties, stress = "", "", "NUMERICAL"
    elif kind == "debug_polar":
        run_type = "DEBUG"
        debug = ("&DEBUG\n  DEBUG_FORCES          F\n  DEBUG_STRESS_TENSOR   F\n  DEBUG_DIPOLE          F\n"
                 f"  DEBUG_POLARIZABILITY  T\n  DE                    {dft.get('debug_field_au', 0.0005)}\n"
                 "  STOP_ON_MISMATCH      F\n&END DEBUG")
        efield = ("    &PERIODIC_EFIELD\n      INTENSITY     0.0\n      POLARISATION  0.0 0.0 1.0\n"
                  "    &END PERIODIC_EFIELD")
    elif kind != "label":
        raise ValueError(kind)
    tmpl = (TEMPLATES / "label.inp").read_text()
    return fill(tmpl, RUN_TYPE=run_type, DEBUG=debug, STRESS=stress, CUTOFF=dft["cutoff"],
                REL_CUTOFF=dft["rel_cutoff"], EPS_DEFAULT=dft["eps_default"], EFIELD=efield,
                MAX_SCF=dft["max_scf"], EPS_SCF=dft["eps_scf"], XC_GRID=xc_grid(dft), MOMENTS=moments,
                PROPERTIES=properties, CELL=cell_block(cell), LEVEL=level_name(lv),
                XC_FUNCTIONAL=xc_functional(lv["functional"]), VDW=vdw(lv), BASIS=lv["basis"],
                PSEUDO=PSEUDO[lv["functional"]])


# ============================================================================ output parsing
def parse_output(path: Path) -> dict:
    """Everything a label run printed. Missing quantities are None.

    energy Ha; forces Ha/Bohr (N,3); stress 3x3 in `stress_unit`, CP2K sign (positive = the cell
    wants to expand); dipole_debye (3,) Berry phase (defined modulo e*R); polar_au 3x3 (Bohr^3).
    """
    t = Path(path).read_text(errors="replace") if Path(path).exists() else ""
    lines = t.splitlines()
    res = {"ended": "PROGRAM ENDED" in t, "scf_failed": "SCF run NOT converged" in t,
           "linres_failed": bool(re.search(r"LINRES.*not converged|Linear response NOT converged", t, re.I)),
           "energy": None, "forces": None, "stress": None, "stress_unit": None, "stress_kind": None,
           "dipole_debye": None, "polar_au": None, "walltime_s": None, "n_scf_steps": None}

    m = re.findall(r"ENERGY\| Total FORCE_EVAL \( \w+ \) energy \[(?:a\.u\.|hartree)\]:?\s+(" + _NUM + ")", t)
    if m:
        res["energy"] = _f(m[-1])

    # forces: 2024.1 "ATOMIC FORCES in [a.u.]" table (# Atom Kind Element X Y Z), or master "FORCES|"
    blocks, cur = [], None
    for ln in lines:
        if "ATOMIC FORCES in" in ln:
            cur = []
            blocks.append(cur)
            continue
        if cur is not None:
            if "SUM OF ATOMIC FORCES" in ln:
                cur = None
                continue
            tok = ln.split()
            if len(tok) == 6 and tok[0].isdigit():
                cur.append([_f(x) for x in tok[3:6]])
    if blocks and blocks[-1]:
        res["forces"] = np.array(blocks[-1])
    else:
        rows = [ln.split()[1:] for ln in lines if ln.strip().startswith("FORCES|")]
        rows = [r for r in rows if len(r) >= 5 and r[0].isdigit()]
        if rows:
            n = max(int(r[0]) for r in rows)
            res["forces"] = np.array([[_f(x) for x in r[2:5]] for r in rows[-n:]])

    # stress: last "STRESS| Analytical|Numerical stress tensor [unit]" block, else old "STRESS TENSOR [GPa]"
    for i, ln in enumerate(lines):
        mm = re.search(r"STRESS\|\s+(Analytical|Numerical) stress tensor \[(\w+)\]", ln)
        if mm:
            rows = []
            for ln2 in lines[i + 1:i + 6]:
                tok = ln2.split()
                if len(tok) == 5 and tok[0] == "STRESS|" and tok[1] in ("x", "y", "z"):
                    rows.append([_f(x) for x in tok[2:5]])
            if len(rows) == 3:
                res["stress"], res["stress_unit"], res["stress_kind"] = np.array(rows), mm.group(2), mm.group(1).upper()
    if res["stress"] is None:
        for i, ln in enumerate(lines):
            mm = re.search(r"STRESS TENSOR \[(\w+)\]", ln)
            if mm:
                rows = []
                for ln2 in lines[i + 1:i + 7]:
                    tok = ln2.split()
                    if len(tok) == 4 and tok[0] in ("X", "Y", "Z"):
                        rows.append([_f(x) for x in tok[1:4]])
                if len(rows) == 3:
                    res["stress"], res["stress_unit"] = np.array(rows), mm.group(1)

    # dipole (qs_moments.F): "Dipole moment [Debye]" then "    X=  -0.12E+00 Y= ... Z= ...     Total= ..."
    for i, ln in enumerate(lines):
        if "Dipole moment [Debye]" in ln and i + 1 < len(lines):
            mm = re.search(r"X=\s*(" + _NUM + r")\s+Y=\s*(" + _NUM + r")\s+Z=\s*(" + _NUM + ")", lines[i + 1])
            if mm:
                res["dipole_debye"] = np.array([_f(mm.group(k)) for k in (1, 2, 3)])
    res["dipole_berry"] = "periodic (Berry phase) operator" in t

    # polarizability (qs_linres_polar_utils.F, polar_print): a.u. block, rows xx,yy,zz / xy,xz,yz / yx,zx,zy
    for i, ln in enumerate(lines):
        if re.search(r"POLAR\|\s+Polarizability tensor \[a\.u\.\]", ln):
            vals = {}
            for ln2 in lines[i + 1:i + 5]:
                mm = re.match(r"\s*POLAR\|\s+(xx,yy,zz|xy,xz,yz|yx,zx,zy)\s+(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + ")", ln2)
                if mm:
                    vals[mm.group(1)] = [_f(mm.group(k)) for k in (2, 3, 4)]
            if len(vals) == 3:
                P = np.empty((3, 3))
                P[0, 0], P[1, 1], P[2, 2] = vals["xx,yy,zz"]
                P[0, 1], P[0, 2], P[1, 2] = vals["xy,xz,yz"]
                P[1, 0], P[2, 0], P[2, 1] = vals["yx,zx,zy"]
                res["polar_au"] = P

    mm = re.findall(r"^\s*CP2K\s+1\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s*$", t, re.M)
    if mm:
        res["walltime_s"] = float(mm[-1])
    steps = re.findall(r"\*\*\* SCF run converged in\s+(\d+) steps", t)
    if steps:
        res["n_scf_steps"] = int(steps[-1])
    return res


def parse_debug_polar(path: Path) -> np.ndarray | None:
    """RUN_TYPE DEBUG polarizability table -> rows (numerical, analytical) for xx..zz, or None."""
    t = Path(path).read_text(errors="replace") if Path(path).exists() else ""
    i = t.find("POLARIZABILITY ======")
    if i < 0:
        return None
    rows = []
    for ln in t[i:].splitlines()[2:12]:
        mm = re.match(r"\s*([xyz]{2})\s+(" + _NUM + r")\s+(" + _NUM + ")", ln)
        if mm:
            rows.append([_f(mm.group(2)), _f(mm.group(3))])
    return np.array(rows) if len(rows) == 9 else None


# ============================================================================ conversions
def to_voigt(S) -> np.ndarray:
    """3x3 -> (xx, yy, zz, yz, xz, xy): ASE/MACE order."""
    S = np.asarray(S, float)
    return np.array([S[0, 0], S[1, 1], S[2, 2], 0.5 * (S[1, 2] + S[2, 1]), 0.5 * (S[0, 2] + S[2, 0]),
                     0.5 * (S[0, 1] + S[1, 0])])


def from_voigt(v) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    if v.size == 9:
        return v.reshape(3, 3)
    xx, yy, zz, yz, xz, xy = v
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def cp2k_stress_to_ase(stress, unit) -> np.ndarray:
    """CP2K prints the pressure-sign stress (positive = the cell wants to expand; validated on 9MA:
    +0.717 GPa vs -dE/dV = +0.731 GPa). ASE/MACE: stress = (1/V) dE/d(strain), the opposite sign, eV/A^3."""
    return -np.asarray(stress, float) * units.STRESS_TO_EV_A3[unit]


# ============================================================================ launching
def launch(dft: dict, ranks, extra: str = "") -> str:
    """MPI launch line for one CP2K run. Cluster-specific bits come from [dft]:
    cp2k_exe (default cp2k.popt) and mpi_flags (default "--mpi=pmi2 --cpu-bind=cores", as on Leonardo)."""
    exe = dft.get("cp2k_exe", "cp2k.popt")
    flags = dft.get("mpi_flags", "--mpi=pmi2 --cpu-bind=cores")
    return " ".join(x for x in ("srun", extra, f"--ntasks={ranks} --cpus-per-task=1", flags, exe) if x)
