"""Fixed-temperature NPT tests with CP2K-MACE, compared with the experimental cell.

For every temperature in [npt] temperatures:
  * start = the fix-deform frame whose ramp temperature is closest (MACE-OFF23 thermalised H positions,
    cell already close to the experimental volume at that T; NOT the X-ray H positions, see 9MA lesson);
  * supercell [npt] supercell (default 2x2x2 = 736 atoms, 16 molecules) built from whole molecules;
  * CP2K FIST + MACE (same input structure as Will's working 9MA/cp2k/npt_mace.inp), NPT_F or NPT_I,
    CSVR thermostat, 1 bar; cell, energy and stress every `print_every` steps, positions every `traj_every`;
  * a watchdog stops a run whose volume leaves [v_min, v_max] x V0 (CP2K EXIT file), so a blow-up is caught.
One Slurm array task per temperature (1 GPU each). `npt-analyze` averages the second part of every run
and compares volume per molecule and cell parameters with experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import geometry as geo
from .cp2k import cell_block, fill
from .io import read_extxyz, read_table, write_xyz
from .project import Campaign
from .slurm import submit, write_script

MD_TEMPLATE = """! DtBuBP CP2K-MACE NPT test, T = {{TEMP}} K. Written by dtbubp.npt; do not edit by hand.
&GLOBAL
  PROJECT md
  RUN_TYPE MD
  PRINT_LEVEL LOW
  SEED {{SEED}}
&END GLOBAL
&FORCE_EVAL
  METHOD FIST
  STRESS_TENSOR ANALYTICAL
  &MM
    &POISSON
      &EWALD
        EWALD_TYPE NONE
      &END EWALD
    &END POISSON
    &FORCEFIELD
      DO_ELECTROSTATICS F
      &NONBONDED
        &MACE
          ATOMS H C
          POT_FILE_NAME model.pth
        &END MACE
      &END NONBONDED
    &END FORCEFIELD
  &END MM
  &SUBSYS
    &CELL
{{CELL}}
      PERIODIC XYZ
    &END CELL
    &TOPOLOGY
      COORD_FILE_NAME start.xyz
      COORD_FILE_FORMAT XYZ
      USE_ELEMENT_AS_KIND T
    &END TOPOLOGY
    &KIND C
      ELEMENT C
    &END KIND
    &KIND H
      ELEMENT H
    &END KIND
  &END SUBSYS
&END FORCE_EVAL
&MOTION
  &MD
    ENSEMBLE {{ENSEMBLE}}
    STEPS {{STEPS}}
    TIMESTEP {{TIMESTEP}}
    TEMPERATURE {{TEMP}}
    &THERMOSTAT
      TYPE CSVR
      REGION GLOBAL
      &CSVR
        TIMECON {{TTIMECON}}
      &END CSVR
    &END THERMOSTAT
    &BAROSTAT
      PRESSURE {{PRESSURE}}
      TIMECON {{BTIMECON}}
    &END BAROSTAT
    &PRINT
      &ENERGY
        &EACH
          MD {{EVERY}}
        &END EACH
      &END ENERGY
    &END PRINT
  &END MD
  &PRINT
    &TRAJECTORY
      FORMAT XYZ
      &EACH
        MD {{TRAJ_EVERY}}
      &END EACH
    &END TRAJECTORY
    &CELL
      &EACH
        MD {{EVERY}}
      &END EACH
    &END CELL
    &STRESS
      &EACH
        MD {{EVERY}}
      &END EACH
    &END STRESS
    &VELOCITIES OFF
    &END VELOCITIES
    &RESTART
      BACKUP_COPIES 0
      &EACH
        MD {{RESTART_EVERY}}
      &END EACH
    &END RESTART
    &RESTART_HISTORY OFF
    &END RESTART_HISTORY
  &END PRINT
&END MOTION
"""


def supercell(syms, pos, cell, reps) -> tuple[list[str], np.ndarray, np.ndarray]:
    na, nb, nc = reps
    S, P = [], []
    for i in range(na):
        for j in range(nb):
            for k in range(nc):
                shift = i * cell[0] + j * cell[1] + k * cell[2]
                S += list(syms)
                P.append(pos + shift)
    return S, np.vstack(P), cell * np.array(reps)[:, None]


def exp_v_per_molecule(c: Campaign, T: float) -> float:
    e = c.cfg["experiment"]
    return float(np.interp(T, e["T_K"], e["V_per_molecule_A3"]))


def phase_of(c: Campaign, T: float) -> dict:
    """Experimental phase at temperature T from [experiment.phases] (T ranges approximate; cooling data)."""
    for ph in c.cfg["experiment"].get("phases", []):
        if ph.get("T_min", -1e9) <= T < ph.get("T_max", 1e9):
            return {"name": ph["name"], "Z": ph["Z"], "space_group": ph.get("space_group", "")}
    return {"name": "?", "Z": None, "space_group": ""}


def xrd_points(c: Campaign) -> list[dict]:
    """Single-crystal structures: V per molecule = V_cell / Z of THAT structure (Z changes between phases)."""
    out = []
    for x in c.cfg["experiment"].get("xrd", []):
        d = dict(x)
        d["V_per_molecule"] = x["V_cell"] / x["Z"]
        out.append(d)
    return sorted(out, key=lambda d: d["T"])


def setup(c: Campaign, dry: bool = False, force: bool = False) -> Path:
    """Write and submit the runs. Temperatures whose folder already has output (md-1.cell) are skipped
    unless force=True, so temperatures can be added to the config while earlier runs are going. Every
    submission gets its own list and job script (runs_<stamp>.txt, npt_job_<stamp>.sh): a pending array task
    of an earlier submission never reads a list that changed under it."""
    import datetime as dt
    n = c.cfg["npt"]
    frames = [f for f in read_extxyz(c.p("frames")) if f["info"]["config_type"] == "fixdeform"]
    model = c.p("model_cp2k")
    if not model.exists() and not dry:
        raise SystemExit(f"CP2K model not found: {model} (did the training job's export step run?)")
    base = c.p("run_dir")
    base.mkdir(parents=True, exist_ok=True)
    runs, skipped = [], []
    for k, T in enumerate(n["temperatures"]):
        fr = min(frames, key=lambda f: abs(f["info"]["T_target"] - T))
        syms, pos, cell = supercell(fr["symbols"], fr["pos"], fr["cell"], n["supercell"])
        rd = base / f"T{int(round(T)):03d}"
        if (rd / "md-1.cell").exists() and not force:
            skipped.append(rd.name)
            continue
        rd.mkdir(exist_ok=True)
        write_xyz(rd / "start.xyz", syms, pos, f"from {fr['info']['frame_id']} (T_ramp {fr['info']['T_target']:.1f} K), "
                  f"supercell {n['supercell']}")
        link = rd / "model.pth"                     # CP2K truncates long file names (~80 chars): 9MA lesson
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(model)
        steps = int(round(n["time_ps"] * 1000 / n["timestep_fs"]))
        (rd / "md.inp").write_text(fill(MD_TEMPLATE, TEMP=T, SEED=1000 + k, CELL=cell_block(cell),
                                        ENSEMBLE=n["ensemble"], STEPS=steps, TIMESTEP=n["timestep_fs"],
                                        TTIMECON=n["thermostat_timecon_fs"], PRESSURE=n["pressure_bar"],
                                        BTIMECON=n["barostat_timecon_fs"], EVERY=n["print_every"],
                                        TRAJ_EVERY=n["traj_every"], RESTART_EVERY=n["restart_every"]))
        V0 = geo.cell_volume(cell)
        meta = {"T": T, "source_frame": fr["info"]["frame_id"], "T_ramp_source": fr["info"]["T_target"],
                "n_atoms": len(syms), "n_molecules": len(syms) // 46, "V0": V0, "steps": steps,
                "V_exp_per_molecule": exp_v_per_molecule(c, T), "exp_phase": phase_of(c, T)}
        (rd / "meta.json").write_text(json.dumps(meta, indent=1))
        runs.append(rd.name)
    if not runs:
        print(f"nothing to submit: every temperature has already started ({', '.join(skipped)}); use --force to redo")
        return base
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    k = 1
    while (base / f"runs_{stamp}.txt").exists():           # two submissions within one second
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{k}"
        k += 1
    listf = base / f"runs_{stamp}.txt"
    listf.write_text("\n".join(runs) + "\n")
    lo, hi = n["v_watchdog"]
    body = c.cfg["env"]["cp2k_mace"].strip() + f"""
export CP2K=${{CP2K:-$HOME/software/cp2k-mace/install/bin/cp2k.psmp}}
ldd $CP2K | grep libcp2k                     # must be cp2k-mace/install/lib64 (handoff, section 10)
MISSING=$(ldd $CP2K $(dirname $CP2K)/../lib64/libcp2k.so* 2>/dev/null | grep "not found" | sort -u)
if [ -n "$MISSING" ]; then echo "CP2K cannot start, missing libraries:"; echo "$MISSING"; exit 2; fi
RUN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {listf})
cd {base}/$RUN
if grep -q "PROGRAM ENDED" md.out 2>/dev/null; then echo "$RUN already finished"; exit 0; fi
if [ -f md-1.restart ]; then
  # continue a run killed at the time limit: CP2K reads positions, velocities, cell and thermostat state
  python - <<'EOF'
s = open('md.inp').read()
if '&EXT_RESTART' not in s:
    s += "&EXT_RESTART\\n  RESTART_FILE_NAME md-1.restart\\n&END EXT_RESTART\\n"
    open('md.inp', 'w').write(s)
EOF
fi
V0=$(python -c "import json;print(json.load(open('meta.json'))['V0'])")
rm -f EXIT
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=cores $CP2K -i md.inp -o md.out &
PID=$!
while kill -0 $PID 2>/dev/null; do
  sleep 30
  v=$(tail -n 1 md-1.cell 2>/dev/null | awk '{{print $NF}}')
  if [[ $v =~ ^[0-9]+\\.?[0-9]*$ ]] && awk -v v="$v" -v a="$V0" 'BEGIN{{exit !(v > {hi}*a || v < {lo}*a)}}'; then
    echo "$RUN: volume $v A^3 left [{lo}, {hi}] x V0 = $V0: stopping"; echo "$v" > STOPPED_AT_VOLUME; touch EXIT
    for i in $(seq 1 12); do kill -0 $PID 2>/dev/null || break; sleep 10; done
    kill $PID 2>/dev/null; break
  fi
done
wait $PID; echo "$RUN: CP2K exit code $?"
"""
    sc = write_script(c.cfg["slurm"]["gpu"], base / f"npt_job_{stamp}.sh", "dtb_npt", body, n["time_limit"], base,
                      array=f"0-{len(runs) - 1}")
    jid = submit(sc, dry=dry)
    c.log(f"**NPT tests submitted**: {len(runs)} temperatures, {n['ensemble']}, supercell {n['supercell']}, "
          f"{n['time_ps']} ps each", runs=runs, skipped_already_started=skipped, script=sc, model=model, job=jid)
    c.update_state("npt_setup", {"runs": runs, "job": jid})
    return sc


def _cell_params(row9) -> tuple:
    return geo.lattice_parameters(np.asarray(row9, float).reshape(3, 3))


def analyze(c: Campaign) -> dict:
    """Mean volume per molecule (MD: V / 16 molecules of the Z = 2 supercell) vs experiment, where the experimental
    volume per molecule is V_cell / Z of the phase present at that temperature (Z = 2, 12 or 10)."""
    n = c.cfg["npt"]
    base = c.p("run_dir")
    reps = np.array(n["supercell"])
    xrd = xrd_points(c)
    out, rows = {}, []
    for rd in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("T")):
        meta = json.loads((rd / "meta.json").read_text())
        cf = rd / "md-1.cell"
        if not cf.exists():
            continue
        C = read_table(cf)                                   # step time Ax Ay Az Bx By Bz Cx Cy Cz V
        t_ps = C[:, 1] / 1000
        keep = t_ps >= n["discard_ps"]
        T = meta["T"]
        ph = phase_of(c, T)
        x = min(xrd, key=lambda d: abs(d["T"] - T)) if xrd else None
        r = {"T": T, "ps_done": float(t_ps[-1]), "stopped": (rd / "STOPPED_AT_VOLUME").exists(),
             "V_exp_per_molecule": exp_v_per_molecule(c, T), "exp_phase": ph["name"], "exp_Z": ph["Z"],
             "xrd_same_T": x if (x and abs(x["T"] - T) <= 3) else None}
        if keep.sum() > 10:
            Vm = C[keep, -1] / meta["n_molecules"]
            lp = np.array([_cell_params(row) for row in C[keep, 2:11]])
            lp[:, :3] /= reps                                # back to the Z = 2 unit cell
            r.update({"V_per_molecule": float(Vm.mean()), "V_per_molecule_std": float(Vm.std()),
                      "V_err_pct": float((Vm.mean() / r["V_exp_per_molecule"] - 1) * 100),
                      "a": float(lp[:, 0].mean()), "b": float(lp[:, 1].mean()), "c": float(lp[:, 2].mean()),
                      "alpha": float(lp[:, 3].mean()), "beta": float(lp[:, 4].mean()), "gamma": float(lp[:, 5].mean())})
        ef = rd / "md-1.ener"
        if ef.exists():
            E = read_table(ef)
            k2 = E[:, 1] / 1000 >= n["discard_ps"]
            if k2.sum():
                r["T_mean"] = float(E[k2, 3].mean())
        out[rd.name] = r
        rows.append(r)
    rows.sort(key=lambda r: -r["T"])
    (base / "npt_summary.json").write_text(json.dumps({"runs": out, "xrd": xrd}, indent=1, default=str))

    L = ["MD: room-T cell (phase I, Z = 2), 2x2x2 supercell = 16 molecules -> V/molecule = V_box / 16.",
         "Experiment: V/molecule = V_cell / Z of the phase present at that T (phase I Z = 2, II Z = 12, III Z = 10).",
         "", f"{'T/K':>5} {'ps':>6} {'<T>':>6}  {'exp phase':<12} {'V/mol MD':>9} {'V/mol exp':>9} {'MD-exp':>7} {'err %':>7}"]
    for r in rows:
        tag = f"{r['exp_phase']} (Z={r['exp_Z']})"
        if "V_per_molecule" in r:
            flag = "  STOPPED (volume watchdog)" if r["stopped"] else ""
            if r["exp_Z"] not in (2, None):
                flag += "  <- exp is a different phase; MD stays in phase I"
            L.append(f"{r['T']:5.0f} {r['ps_done']:6.1f} {r.get('T_mean', float('nan')):6.1f}  {tag:<12} "
                     f"{r['V_per_molecule']:9.2f} {r['V_exp_per_molecule']:9.2f} "
                     f"{r['V_per_molecule'] - r['V_exp_per_molecule']:+7.2f} {r['V_err_pct']:+7.2f}{flag}")
        else:
            L.append(f"{r['T']:5.0f} {r['ps_done']:6.1f} {'':6}  {tag:<12}   (less than {n['discard_ps']:.0f} ps: no average yet)")

    L += ["", "Single-crystal XRD structures (V/molecule = V_cell / Z):",
          f"{'T/K':>5} {'phase':<6} {'Z':>3} {'V_cell':>9} {'V/mol':>8}   {'MD V/mol':>9} {'MD-XRD':>7}  source"]
    for x in xrd:
        md = next((r for r in rows if abs(r["T"] - x["T"]) <= 3 and "V_per_molecule" in r), None)
        mdtxt = f"{md['V_per_molecule']:9.2f} {md['V_per_molecule'] - x['V_per_molecule']:+7.2f}" if md else f"{'(no run)':>9} {'':>7}"
        L.append(f"{x['T']:5.0f} {x['phase']:<6} {x['Z']:3d} {x['V_cell']:9.2f} {x['V_per_molecule']:8.2f}   {mdtxt}  {x['source']}")

    L += ["", "Cell parameters (Z = 2 cell). Compared only where the experiment is also phase I:",
          f"{'':>14} {'a':>7} {'b':>7} {'c':>7} {'alpha':>7} {'beta':>7} {'gamma':>7}"]
    for r in rows:
        if "a" not in r:
            continue
        L.append(f"MD   {r['T']:5.0f} K  " + " ".join(f"{r[k]:7.3f}" if k in 'abc' else f"{r[k]:7.2f}"
                                                for k in ("a", "b", "c", "alpha", "beta", "gamma")))
        x = r["xrd_same_T"]
        if x and x["Z"] == 2:
            L.append(f"XRD  {x['T']:5.0f} K  " + " ".join(f"{v:7.3f}" if i < 3 else f"{v:7.2f}" for i, v in enumerate(x["cell"])))
    txt = "\n".join(L)
    (base / "npt_summary.txt").write_text(txt + "\n")
    print(txt)
    _plot(base, rows, c.cfg["experiment"], xrd, c)
    c.log("**NPT tests analysed**", summary=base / "npt_summary.txt")
    return out


def _plot(base, rows, e, xrd, c):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    shades = {2: "#ffffff", 12: "#eef3fb", 10: "#fbefe9"}
    for ph in e.get("phases", []):
        lo, hi = max(ph.get("T_min", 0), 0), min(ph.get("T_max", 400), 400)
        ax[0].axvspan(lo, hi, color=shades.get(ph["Z"], "0.95"), zorder=0)
        ax[0].text((max(lo, 60) + min(hi, 310)) / 2, 0.97, f"{ph['name']}  Z={ph['Z']}", transform=ax[0].get_xaxis_transform(),
                   ha="center", va="top", fontsize=8, color="0.35")
    ax[0].plot(e["T_K"], e["V_per_molecule_A3"], "-", color="0.4", lw=1, label="experiment, VT cooling curve")
    if xrd:
        mk = {2: "s", 12: "D", 10: "^"}
        for Z in sorted({x["Z"] for x in xrd}):
            pts = [x for x in xrd if x["Z"] == Z]
            ax[0].plot([x["T"] for x in pts], [x["V_per_molecule"] for x in pts], mk.get(Z, "o"), color="k", ms=7,
                       mfc="white", label=f"SC-XRD, Z = {Z}")
    d = [r for r in rows if "V_per_molecule" in r]
    if d:
        ax[0].errorbar([r["T"] for r in d], [r["V_per_molecule"] for r in d], yerr=[r["V_per_molecule_std"] for r in d],
                       fmt="o", color="C3", label="CP2K-MACE NPT r0 (phase I cell)")
    ax[0].set(xlabel="T (K)", ylabel="V per molecule (A^3)", xlim=(60, 310))
    ax[0].legend(fontsize=8, loc="lower right")
    n = c.cfg["npt"]
    for rd in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("T")):
        cf = rd / "md-1.cell"
        if cf.exists():
            C = read_table(cf)
            meta = json.loads((rd / "meta.json").read_text())
            ax[1].plot(C[:, 1] / 1000, C[:, -1] / meta["n_molecules"], lw=0.7, label=rd.name)
    ax[1].axvline(n["discard_ps"], color="0.5", ls=":")
    ax[1].set(xlabel="time (ps)", ylabel="V per molecule (A^3)", title="volume traces")
    ax[1].legend(fontsize=7, ncol=2)
    fig.savefig(base / "npt_summary.png", dpi=150)
