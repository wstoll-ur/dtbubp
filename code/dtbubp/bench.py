"""Level-of-theory benchmark: full CP2K cell optimisation of the Z = 2 crystal with several functionals,
dispersion corrections and basis sets; the static equilibrium volume per molecule is compared with experiment.

Why a cell optimisation and not single points on the EOS structures: those structures carry BLYP-relaxed
molecules. Another functional would see their bonds as strained, and that intramolecular stress would swamp
the intermolecular pressure (the same trap as the fix-deform labels). CELL_OPT relaxes molecules and cell
together at each level.

  bench-setup [--only a,b] [--dry-run]   one directory per level + one array job (one DCGP node per level)
  bench-analyze                          table: status, steps, V/molecule (current or final), vs experiment,
                                         cell parameters, seconds per optimisation step

Start structure: eos/V430 of fixedT_r0 (the 298 K frame rigid-scaled to the experimental 298 K volume,
molecules relaxed with r0 = BLYP-like). Every level starts from the same atoms and cell.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import cp2k, geometry as geo
from .io import read_xyz, write_xyz
from .project import Campaign
from .slurm import submit, write_script

_NUM = cp2k._NUM

PSEUDO = cp2k.PSEUDO
_xc_functional = cp2k.xc_functional      # moved to cp2k.py (shared with the label deck)
_vdw = cp2k.vdw


def make_input(dft: dict, b: dict, lv: dict, cell) -> str:
    xc_grid = cp2k.xc_grid(dft)
    tmpl = (cp2k.TEMPLATES / "cellopt.inp").read_text()
    return cp2k.fill(tmpl, NAME=lv["name"], DESCRIPTION=lv.get("description", ""), MAX_ITER=b["max_iter"],
                     P_TOL_BAR=b["pressure_tol_bar"], MAX_FORCE=b["max_force_au"], CUTOFF=dft["cutoff"],
                     REL_CUTOFF=dft["rel_cutoff"], EPS_DEFAULT=dft["eps_default"], MAX_SCF=dft["max_scf"],
                     EPS_SCF=dft["eps_scf"], XC_FUNCTIONAL=_xc_functional(lv["functional"]), XC_GRID=xc_grid,
                     VDW=_vdw(lv), CELL=cp2k.cell_block(cell), BASIS=lv["basis"], PSEUDO=PSEUDO[lv["functional"]])


def _levels(c: Campaign, only=None) -> list[dict]:
    lv = [x for x in c.cfg["bench"]["levels"] if x.get("run", True)]
    if only:
        want = set(only)
        lv = [x for x in c.cfg["bench"]["levels"] if x["name"] in want]
        missing = want - {x["name"] for x in lv}
        if missing:
            raise SystemExit(f"unknown levels: {sorted(missing)}")
    return lv


def setup(c: Campaign, dry: bool = False, only=None) -> Path:
    dc = Campaign(c.p("dft_config"))
    dft, b = dc.cfg["dft"], c.cfg["bench"]
    start = c.p("start_dir")
    _, syms, pos = next(read_xyz(start / "coord.xyz"))
    meta = json.loads((start / "meta.json").read_text())
    cell = np.array(meta["cell"])
    nmol = int(meta["n_molecules"])
    root = c.p("run_dir")
    root.mkdir(parents=True, exist_ok=True)
    names = []
    for lv in _levels(c, only):
        d = root / lv["name"]
        if (d / "output.out").exists() and not dry:
            print(f"{lv['name']}: already has output.out -> resubmitted as a continuation (restart file) if unfinished")
        d.mkdir(exist_ok=True)
        write_xyz(d / "coord.xyz", syms, pos, f"bench start: {start.name} ({meta['V_per_molecule']} A^3/molecule)")
        (d / "input.inp").write_text(make_input(dft, b, lv, cell))
        (d / "level.json").write_text(json.dumps({**lv, "n_molecules": nmol, "start": str(start),
                                                  "V0_per_molecule": geo.cell_volume(cell) / nmol}, indent=1))
        names.append(lv["name"])
    (root / "levels.txt").write_text("\n".join(names) + "\n")
    ranks = int(b.get("ranks", dft["cores_per_node"]))
    if b.get("hosts"):                       # multi-node run (BlueHive: the three Vermont nodes together)
        dft = dict(dft, hosts=b["hosts"], mpi_prefix=b.get("mpi_prefix", ""), mpi_extra=b.get("mpi_extra", ""))
    body = dc.cfg["env"]["dft"].strip() + f"""
export OMP_NUM_THREADS=1
cd {root}
L=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" levels.txt)
cd $L || exit 1
echo "level $L on $(hostname) at $(date)"
if grep -q "PROGRAM ENDED" output.out 2>/dev/null; then echo "$L already finished"; exit 0; fi
INP=input.inp
if [ -s bench-1.restart ]; then                  # continuation of a run killed at the time limit
  [ -f output.out ] && mv output.out output_part$(date +%s).out
  cp input.inp restart.inp
  sed -i 's/SCF_GUESS  ATOMIC/SCF_GUESS  RESTART/' restart.inp
  printf '&EXT_RESTART\\n  RESTART_FILE_NAME bench-1.restart\\n&END EXT_RESTART\\n' >> restart.inp
  INP=restart.inp
  echo "continuing from bench-1.restart"
fi
{cp2k.launch(dft, ranks, cpus=f"0-{ranks - 1}")} -i $INP -o output.out > srun.log 2>&1
echo "exit $? at $(date)"
grep -E "OPTIMIZATION COMPLETED|MAXIMUM NUMBER OF OPTIMIZATION STEPS|ABORT" output.out | head -3
"""
    sc = write_script(dc.cfg["slurm"]["cpu"], root / "bench_job.sh", "dtb_bench", body, b["time_limit"], root,
                      array=f"0-{len(names) - 1}")
    jid = submit(sc, dry=dry)
    c.log("**level-of-theory benchmark submitted** (CELL_OPT, one node per level)", job=jid, levels=names, dir=root)
    return sc


# ============================================================================ analysis
def _read_cells(path: Path) -> np.ndarray | None:
    """CP2K .cell file rows: step, time, Ax..Cz, volume -> (n, 11) without the time column issue."""
    if not path.exists():
        return None
    rows = []
    for ln in path.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        tok = ln.split()
        if len(tok) >= 12:
            try:
                rows.append([float(x) for x in tok[:12]])
            except ValueError:
                pass
    return np.array(rows) if rows else None


def parse_run(d: Path) -> dict:
    outs = sorted(d.glob("output_part*.out")) + ([d / "output.out"] if (d / "output.out").exists() else [])
    t = "\n".join(o.read_text(errors="replace") for o in outs)
    # CP2K appends to an existing output.out: status and errors come from the LAST run only
    # (bench r1: the C9 keyword aborts stayed at the top of the files of the resubmitted levels).
    last = t.split("PROGRAM STARTED AT")[-1] if outs else ""
    r = {"started": bool(outs), "ended": "PROGRAM ENDED" in last,
         "converged": bool(re.search(r"OPTIMIZATION COMPLETED", last)),
         "max_iter": "MAXIMUM NUMBER OF OPTIMIZATION STEPS" in last,
         "aborted": "ABORT" in last, "error": None}
    if r["aborted"]:
        m = re.search(r"\[ABORT\]\s*\*\s*\n(.*?)\n", last) or re.search(r"ABORT(.*)\n(.*)\n", last)
        if m:
            r["error"] = " ".join(" ".join(m.groups()).replace("*", " ").replace("\\___/", " ").split())[:160]
    en = re.findall(r"(?:Total Energy\s*=|OPT\|\s+Total energy \[hartree\])\s+(" + _NUM + ")", t, re.I)
    pr = re.findall(r"(?:Internal Pressure \[bar\]\s*=|OPT\|\s+Internal pressure \[bar\])\s+(" + _NUM + ")", t, re.I)
    tm = re.findall(r"(?:Used time\s*=|OPT\|\s+Used time \[s\])\s+(" + _NUM + ")", t, re.I)
    r["n_steps"] = len(en)
    r["energy_ha"] = cp2k._f(en[-1]) if en else None
    r["pressure_bar"] = cp2k._f(pr[-1]) if pr else None
    r["s_per_step"] = float(np.median([cp2k._f(x) for x in tm])) if tm else None
    cells = _read_cells(d / "bench-1.cell")
    if cells is not None:
        cell = cells[-1, 2:11].reshape(3, 3)
        r["cell"] = cell
        r["V_traj"] = cells[:, 11]
    else:
        vols = re.findall(r"CELL\| Volume \[angstrom\^3\]:?\s+(" + _NUM + ")", t)
        r["cell"], r["V_traj"] = None, np.array([cp2k._f(v) for v in vols]) if vols else None
    return r


def analyze(c: Campaign) -> dict:
    root = c.p("run_dir")
    e = c.cfg["experiment"]
    v80, v298 = e["V_per_molecule_80K"], e["V_per_molecule_298K"]
    # every level folder, not levels.txt (that only lists the last submission, e.g. a --only resubmission)
    names = sorted(p.parent.name for p in root.glob("*/level.json"))
    out, lines = {}, []
    hdr = (f"{'level':<26} {'status':<11} {'steps':>5} {'V/mol':>7} {'vs80K':>7} {'vs298K':>7} {'P(bar)':>8} "
           f"{'s/step':>7}   a      b      c      alpha  beta   gamma")
    lines += [f"Static (0 K, no zero-point) equilibrium volume per molecule. Experiment: {v80} (80 K), {v298} (298 K).",
              "A good level sits a few % BELOW the 80 K value (zero-point + thermal expansion are missing).", "", hdr]
    for n in names:
        d = root / n
        lv = json.loads((d / "level.json").read_text())
        nmol = lv["n_molecules"]
        r = parse_run(d)
        if r["converged"]:
            st = "converged"
        elif r["max_iter"]:
            st = "max_iter"
        elif r["aborted"]:
            st = "FAILED"
        elif r["ended"]:
            st = "ended?"
        elif r["started"]:
            st = "running"
        else:
            st = "queued"
        V = r["V_traj"][-1] / nmol if r["V_traj"] is not None and len(r["V_traj"]) else None
        lp = geo.lattice_parameters(r["cell"]) if r.get("cell") is not None else None
        row = {"status": st, "V_per_molecule": V, "steps": r["n_steps"], "pressure_bar": r["pressure_bar"],
               "s_per_step": r["s_per_step"], "lattice": lp, "error": r["error"],
               "V_traj_per_molecule": (r["V_traj"] / nmol).tolist() if r["V_traj"] is not None else None}
        out[n] = row
        fv = (lambda x, f: f.format(x) if x is not None else "-")
        lines.append(f"{n:<26} {st:<11} {r['n_steps']:>5} {fv(V, '{:7.1f}'):>7} "
                     f"{fv(None if V is None else 100 * (V / v80 - 1), '{:+6.1f}%'):>7} "
                     f"{fv(None if V is None else 100 * (V / v298 - 1), '{:+6.1f}%'):>7} "
                     f"{fv(r['pressure_bar'], '{:8.0f}'):>8} {fv(r['s_per_step'], '{:7.0f}'):>7}   "
                     + ("  ".join(f"{x:5.2f}" if k < 3 else f"{x:5.1f}" for k, x in enumerate(lp)) if lp else ""))
        if r["error"]:
            lines.append(f"{'':<26}   error: {r['error']}")
    exp_cell = e.get("cell_298")
    if exp_cell:
        lines.append(f"{'XRD 298 K (Z = 2)':<26} {'':<11} {'':>5} {v298:7.1f} {'':>7} {'':>7} {'':>8} {'':>7}   "
                     + "  ".join(f"{x:5.2f}" if k < 3 else f"{x:5.1f}" for k, x in enumerate(exp_cell)))
    lines.append("Cell parameters are in the start cell's setting (from the fix-deform supercell), which may not be the "
                 "CIF's reduced setting; compare volumes first.")
    txt = "\n".join(lines)
    (root / "bench_summary.txt").write_text(txt + "\n")
    (root / "bench_summary.json").write_text(json.dumps(out, indent=1, default=lambda x: list(x)))
    print(txt)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for k, (n, row) in enumerate(out.items()):
            if row["V_traj_per_molecule"]:
                ax.plot(row["V_traj_per_molecule"], label=n, color=f"C{k % 10}", ls="-" if k < 10 else "--")
        ax.axhline(v80, color="k", ls=":", lw=1)
        ax.axhline(v298, color="k", ls="--", lw=1)
        ax.text(0, v80, " exp 80 K", va="bottom", fontsize=8)
        ax.text(0, v298, " exp 298 K", va="bottom", fontsize=8)
        ax.set(xlabel="CELL_OPT step", ylabel="V per molecule (A^3)")
        ax.legend(fontsize=7, ncol=2)
        fig.savefig(root / "bench_summary.png", dpi=150)
    except Exception as ex:
        print("plot skipped:", ex)
    return out
