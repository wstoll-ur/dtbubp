"""DFT labelling with CP2K: smoke test, per-frame inputs, packed Slurm array jobs, collection.

Layout (inside the campaign folder):
  label/fNNNN/   coord.xyz, input.inp, meta.json  ->  output.out
  label/chunk_K.txt, label/label_job.sh
  smoketest/{label,stress_numerical,debug_polar}/
  parsed/labels.xyz (+ train.xyz, valid.xyz, test.xyz), parsed/collect.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import cp2k, geometry as geo, units
from .io import read_extxyz, read_xyz, write_extxyz_frame, write_xyz
from .project import Campaign
from .slurm import submit, write_script


def _frames(c: Campaign) -> list[dict]:
    p = c.p("selection_dir") / "frames.extxyz"
    if not p.exists():
        raise SystemExit(f"{p} missing: run `dtb select` first (locally, where the fix-deform dump is)")
    return list(read_extxyz(p))


def _env(c: Campaign) -> str:
    return c.cfg["env"]["dft"].strip()


def _pool_body(c: Campaign, workdir: Path, list_file: str) -> str:
    """Run the frames listed in `list_file` K at a time on one node (srun --exact steps)."""
    d = c.cfg["dft"]
    k = int(d["frames_per_node"])
    ranks = int(d["cores_per_node"]) // k
    mem = int(d["mem_per_node_gb"]) // k
    tmo = int(d["frame_timeout_min"])
    return _env(c) + f"""
export OMP_NUM_THREADS=1
cd {workdir}
run_frame() {{
  local f=$1
  grep -q "PROGRAM ENDED" $f/output.out 2>/dev/null && return 0
  ( cd $f && timeout {tmo}m {cp2k.launch(d, ranks, f"--exact --nodes=1 --mem={mem}G")} \\
        -i input.inp -o output.out > srun.log 2>&1 \\
    || echo "$f: CP2K failed or timed out (see $f/output.out, $f/srun.log)" )
}}
for f in $(cat {list_file}); do
  while [ $(jobs -rp | wc -l) -ge {k} ]; do sleep 10; done
  run_frame $f &
  sleep 2
done
wait
echo "all frames of {list_file} attempted"
"""


# ============================================================================ smoke test
def smoketest(c: Campaign, check: bool = False, dry: bool = False):
    """One frame (the fix-deform frame closest to 300 K), three CP2K runs on one node:
      label            the production deck: must yield E, F, stress, dipole, polarizability (+ timing)
      stress_numerical same frame, numerical stress: must match the analytical one (< 0.05 GPa)
      debug_polar      RUN_TYPE DEBUG: analytical vs finite-field polarizability (< 2 %)
    """
    d = c.root / "smoketest"
    kinds = ("label", "stress_numerical", "debug_polar")
    if check:
        return smoketest_check(c, d)
    fr = min((f for f in _frames(c) if f["info"]["config_type"] == "fixdeform"),
             key=lambda f: abs(f["info"]["T_target"] - 300.0))
    for k in kinds:
        kd = d / k
        kd.mkdir(parents=True, exist_ok=True)
        write_xyz(kd / "coord.xyz", fr["symbols"], fr["pos"], f"smoketest frame {fr['info']['frame_id']}")
        (kd / "input.inp").write_text(cp2k.make_input(c.cfg["dft"], fr["cell"], k))
    (d / "frames.txt").write_text("\n".join(kinds) + "\n")
    sc = write_script(c.cfg["slurm"]["cpu"], d / "smoke_job.sh", "dtb_smoke", _pool_body(c, d, "frames.txt"),
                      c.cfg["dft"]["smoke_time_limit"], d)
    jid = submit(sc, dry=dry)
    c.log("**smoke test submitted**: label / numerical stress / debug polarizability on one frame",
          frame=fr["info"]["frame_id"], T_target=round(fr["info"]["T_target"], 1), job=jid, dir=d)


def smoketest_check(c: Campaign, d: Path) -> bool:
    ok = True
    rep = {}
    lab = cp2k.parse_output(d / "label" / "output.out")
    rep["label_ended"] = lab["ended"]
    for q in ("energy", "forces", "stress", "dipole_debye", "polar_au"):
        have = lab[q] is not None
        rep[f"has_{q}"] = have
        ok &= have
    if lab["walltime_s"]:
        rep["label_walltime_min"] = round(lab["walltime_s"] / 60, 1)
    num = cp2k.parse_output(d / "stress_numerical" / "output.out")
    if lab["stress"] is not None and num["stress"] is not None:
        Sa = lab["stress"] * units.STRESS_TO_EV_A3[lab["stress_unit"]] * units.EV_A3_GPA
        Sn = num["stress"] * units.STRESS_TO_EV_A3[num["stress_unit"]] * units.EV_A3_GPA
        diff = float(np.abs(Sa - Sn).max())
        rep.update(stress_max_abs_diff_GPa=round(diff, 4), pressure_analytical_GPa=round(float(np.trace(Sa) / 3), 4),
                   pressure_numerical_GPa=round(float(np.trace(Sn) / 3), 4))
        ok &= diff < 0.05
    else:
        rep["stress_comparison"] = "not available yet"
        ok = False
    dbg = cp2k.parse_debug_polar(d / "debug_polar" / "output.out")
    if dbg is not None:
        big = np.abs(dbg[:, 1]) > 1.0
        rel = np.abs(dbg[:, 0] - dbg[:, 1])[big] / np.abs(dbg[:, 1])[big] * 100
        rep["polar_max_rel_err_pct"] = round(float(rel.max()), 3) if rel.size else None
        rep["polar_diag_au"] = [round(float(x), 2) for x in dbg[[0, 4, 8], 1]]
        ok &= bool(rel.size) and rel.max() < 2.0
    else:
        rep["polar_debug"] = "not available (still running, failed, or PERIODIC_EFIELD unsupported: see output)"
        ok = False
    if lab["dipole_debye"] is not None:
        cell = np.array([[float(x) for x in ln.split()[1:4]] for ln in
                         (d / "label" / "input.inp").read_text().splitlines() if ln.strip()[:2] in ("A ", "B ", "C ")])
        mu, ratio = geo.fold_dipole(lab["dipole_debye"] * units.DEBYE_EA, cell)
        rep["dipole_folded_eA"] = [round(float(x), 4) for x in mu]
        rep["dipole_over_half_quantum"] = round(ratio, 4)
    rep["verdict"] = "PASS: submit the labelling (`dtb label`)" if ok else "NOT PASSED: read the numbers above"
    c.log("**smoke test check**", **rep)
    c.update_state("smoketest", rep)
    return ok


# ============================================================================ labelling
def setup(c: Campaign, dry: bool = False, only_split: str | None = None):
    lab = c.root / "label"
    lab.mkdir(exist_ok=True)
    frames = _frames(c)
    names = []
    for fr in frames:
        if only_split and fr["info"]["split"] != only_split:
            continue
        fid = fr["info"]["frame_id"]
        fd = lab / fid
        fd.mkdir(exist_ok=True)
        if not (fd / "input.inp").exists():
            write_xyz(fd / "coord.xyz", fr["symbols"], fr["pos"], f"{fid} step={fr['info'].get('step')}")
            (fd / "input.inp").write_text(cp2k.make_input(c.cfg["dft"], fr["cell"], "label"))
            meta = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fr["info"].items()}
            meta["cell"] = fr["cell"].tolist()
            (fd / "meta.json").write_text(json.dumps(meta, indent=1))
        names.append(fid)
    todo = [n for n in names if not _done(lab / n)]
    nch = int(c.cfg["dft"]["chunks"])
    for k in range(nch):
        (lab / f"chunk_{k}.txt").write_text("\n".join(todo[k::nch]) + "\n")
    body = _pool_body(c, lab, "chunk_${SLURM_ARRAY_TASK_ID}.txt")
    sc = write_script(c.cfg["slurm"]["cpu"], lab / "label_job.sh", "dtb_label", body,
                      c.cfg["dft"]["time_limit"], lab, array=f"0-{nch - 1}")
    jid = submit(sc, dry=dry)
    c.log(f"**label submitted**: {len(todo)} frames to run ({len(names) - len(todo)} already done)",
          chunks=nch, frames_per_node=c.cfg["dft"]["frames_per_node"], job=jid, script=sc)
    c.update_state("label", {"job": jid, "frames": len(names), "to_run": len(todo)})


def _done(fd: Path) -> bool:
    o = fd / "output.out"
    return o.exists() and "PROGRAM ENDED" in o.read_text(errors="replace")[-5000:]


def status(c: Campaign):
    lab = c.root / "label"
    if not lab.exists():
        print("no label folder yet")
        return
    dirs = sorted(p for p in lab.iterdir() if p.is_dir())
    done = sum(_done(p) for p in dirs)
    started = sum((p / "output.out").exists() for p in dirs)
    times = []
    for p in dirs:
        if _done(p):
            r = cp2k.parse_output(p / "output.out")
            if r["walltime_s"]:
                times.append(r["walltime_s"] / 60)
    msg = f"{done} finished / {started} started / {len(dirs)} frames"
    if times:
        msg += f"; wall time per frame: median {np.median(times):.1f} min, max {np.max(times):.1f} min"
    print(msg)


# ============================================================================ collect
def collect(c: Campaign) -> dict:
    """CP2K outputs -> parsed/labels.xyz and per-split files with every label in MACE's units.

    REF_energy eV, REF_forces eV/A, REF_stress eV/A^3 (ASE sign, Voigt), REF_dipole e*A (Berry phase,
    folded to the branch nearest zero), REF_polarizability e*A^2/V (MACE-MDP / SPICE-alpha units,
    9 values row-major). polarizability_A3 is also written, for humans.
    """
    lab, out = c.root / "label", c.p("parsed_dir")
    out.mkdir(parents=True, exist_ok=True)
    d = c.cfg["dft"]
    st = {"frames": 0, "kept": 0, "not_finished": 0, "scf_failed": 0, "missing_forces": 0, "missing_stress": 0,
          "missing_dipole": 0, "missing_polar": 0, "max_force": 0, "dipole_branch_warning": 0}
    rows = {"train": [], "valid": [], "test": []}
    allf = open(out / "labels.xyz", "w")
    for fd in sorted(p for p in lab.iterdir() if p.is_dir() and p.name[0] in "fs"):
        st["frames"] += 1
        r = cp2k.parse_output(fd / "output.out")
        if not r["ended"] or r["energy"] is None:
            st["not_finished"] += 1
            continue
        if r["scf_failed"]:
            st["scf_failed"] += 1
            continue
        meta = json.loads((fd / "meta.json").read_text())
        _, syms, pos = next(read_xyz(fd / "coord.xyz"))
        cell = np.array(meta.pop("cell"))
        if r["forces"] is None or len(r["forces"]) != len(syms):
            st["missing_forces"] += 1
            continue
        if r["stress"] is None:
            st["missing_stress"] += 1
            continue
        F = r["forces"] * units.HA_BOHR_EV_A
        if np.abs(F).max() > d["max_force_ev_a"]:
            st["max_force"] += 1
            continue
        S = cp2k.cp2k_stress_to_ase(r["stress"], r["stress_unit"])
        info = {"REF_energy": r["energy"] * units.HA_EV, "REF_stress": cp2k.to_voigt(S),
                "pressure_GPa": float(-np.trace(S) / 3 * units.EV_A3_GPA)}
        if r["dipole_debye"] is not None:
            mu, ratio = geo.fold_dipole(r["dipole_debye"] * units.DEBYE_EA, cell)
            info["REF_dipole"] = mu
            info["dipole_over_half_quantum"] = ratio
            if ratio > 0.5:
                st["dipole_branch_warning"] += 1
                info["config_dipole_weight"] = 0.0         # MACE: ignore this frame's dipole
        else:
            st["missing_dipole"] += 1
        if r["polar_au"] is not None:
            P = 0.5 * (r["polar_au"] + r["polar_au"].T)
            info["REF_polarizability"] = (P * units.AU_POL_EA2V).ravel()
            info["polarizability_A3"] = (P * units.AU_POL_A3).ravel()
        else:
            st["missing_polar"] += 1
        for k in ("frame_id", "config_type", "split", "step", "time_ns", "T_target", "volume", "parent_id"):
            if k in meta:
                info[k] = meta[k]
        info["walltime_min"] = round((r["walltime_s"] or 0) / 60, 2)
        write_extxyz_frame(allf, syms, pos, cell, {"REF_forces": F}, info)
        rows[meta.get("split", "train")].append((syms, pos, cell, F, info))
        st["kept"] += 1
    allf.close()
    for split, frs in rows.items():
        with open(out / f"{split}.xyz", "w") as f:
            for syms, pos, cell, F, info in frs:
                write_extxyz_frame(f, syms, pos, cell, {"REF_forces": F}, info)
        st[f"{split}_frames"] = len(frs)
    (out / "collect.json").write_text(json.dumps(st, indent=1))
    c.log("**collect**: CP2K outputs -> extxyz", dir=out, **st)
    c.update_state("collect", st)
    if st["kept"]:
        summarize(out)
    return st


def summarize(out: Path):
    """Quick numbers + parity-free sanity plots of the labels (pressure, dipole, alpha vs T)."""
    fr = list(read_extxyz(out / "labels.xyz"))
    T = np.array([f["info"].get("T_target", np.nan) for f in fr])
    V = np.array([f["info"].get("volume", np.nan) for f in fr])
    Pg = np.array([f["info"]["pressure_GPa"] for f in fr])
    iso = np.array([np.trace(np.asarray(f["info"]["polarizability_A3"]).reshape(3, 3)) / 3
                    if "polarizability_A3" in f["info"] else np.nan for f in fr])
    mu = np.array([np.linalg.norm(f["info"]["REF_dipole"]) if "REF_dipole" in f["info"] else np.nan for f in fr])
    txt = (f"{len(fr)} frames; pressure {np.nanmean(Pg):+.3f} +- {np.nanstd(Pg):.3f} GPa "
           f"(min {np.nanmin(Pg):+.2f}, max {np.nanmax(Pg):+.2f}); |mu| {np.nanmean(mu):.3f} e*A; "
           f"alpha_iso {np.nanmean(iso):.1f} A^3 per cell ({np.nanmean(iso) / 2:.1f} per molecule)")
    (out / "summary.txt").write_text(txt + "\n")
    print(txt)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        ax[0].scatter(V, Pg, s=4, c=T, cmap="viridis")
        ax[0].set(xlabel="cell volume (A^3)", ylabel="DFT static pressure (GPa)")
        ax[1].scatter(T, iso, s=4)
        ax[1].set(xlabel="T target (K)", ylabel="alpha_iso per cell (A^3)")
        ax[2].scatter(T, mu, s=4)
        ax[2].set(xlabel="T target (K)", ylabel="|dipole| per cell (e*A)")
        fig.savefig(out / "labels_overview.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)
