"""Choose the frames to label from the MACE-OFF23 fix-deform cooling run, and build strained copies.

Fix-deform frames (config [select]):
  * the ramp (300 -> 53 K, 197 389 frames every 50 fs) is cut into temperature bins;
  * every bin gets the same number of frames: half evenly spaced in time (typical thermal
    configurations), half by farthest-point sampling on the rotor/twist features (rare states:
    rotor hops, biphenyl excursions), never two frames of one bin closer than `min_gap_fs`;
  * features per frame: cos 3phi, sin 3phi for the four tBu rotors (threefold: a 120 deg rotation is
    the same state with the methyls relabelled) and cos phi, sin phi for the two biphenyl twists, read
    from the tbu_rotations*.dat files already computed from this trajectory.

Strained frames (config [strain]): copies of selected frames with the cell deformed, so the stress
is learnt away from the fix-deform volumes as well (9MA lesson: the expanded side must be labelled).
  * rigid-molecule strains: whole molecules move with their centres of mass, intramolecular geometry
    untouched; isotropic linear strain in [iso_min, iso_max] + random deviatoric part;
  * affine strains: every atom follows the cell, small amplitude (bond-length response to strain).

Every frame keeps its source step, time and temperature. Train/valid/test are assigned by time
block, so correlated neighbouring frames never end up on both sides of a split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import geometry as geo
from .io import iter_lammps_dump, read_table, write_extxyz_frame
from .project import Campaign


# ---------------------------------------------------------------------------- features
def load_features(c: Campaign) -> dict:
    s = c.cfg["select"]
    tbu = read_table(c.path(s["tbu_dat"]))          # step time_ns T_target tBu1..4 (deg)
    bip = read_table(c.path(s["biphenyl_dat"]))     # step time_ns T_target bip1 bip2 (deg)
    if not np.array_equal(tbu[:, 0], bip[:, 0]):
        raise ValueError("tbu and biphenyl .dat files do not have the same steps")
    ph_t = np.radians(tbu[:, 3:7])
    ph_b = np.radians(bip[:, 3:5])
    feats = np.hstack([np.cos(3 * ph_t), np.sin(3 * ph_t), np.cos(ph_b), np.sin(ph_b)])
    return {"step": tbu[:, 0].astype(int), "time_ns": tbu[:, 1], "T": tbu[:, 2], "feat": feats,
            "tbu_deg": tbu[:, 3:7], "bip_deg": bip[:, 3:5]}


def _fps(feat: np.ndarray, times_fs: np.ndarray, k: int, chosen: list[int], gap_fs: float, rng) -> list[int]:
    """Farthest-point sampling (Euclidean on features) with a minimum time gap to every chosen frame."""
    n = len(feat)
    out = list(chosen)
    if not out:
        out.append(int(rng.integers(n)))
    dmin = np.min(np.linalg.norm(feat[:, None] - feat[out][None], axis=-1), axis=1)
    while len(out) < len(chosen) + k:
        blocked = np.zeros(n, bool)
        for j in out:
            blocked |= np.abs(times_fs - times_fs[j]) < gap_fs
        cand = np.where(~blocked)[0]
        if len(cand) == 0:
            break
        j = int(cand[np.argmax(dmin[cand])])
        out.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(feat - feat[j], axis=1))
    return out[len(chosen):]


def choose_frames(c: Campaign) -> dict:
    s = c.cfg["select"]
    F = load_features(c)
    rng = np.random.default_rng(s["seed"])
    T, t_fs = F["T"], F["time_ns"] * 1e6
    edges = np.arange(np.floor(T.min() / s["bin_K"]) * s["bin_K"], T.max() + s["bin_K"], s["bin_K"])
    nb = len(edges) - 1
    counts = np.histogram(T, edges)[0]
    want = np.floor(s["n_frames"] * counts / counts.sum()).astype(int)
    want[np.argsort(-counts)[: s["n_frames"] - want.sum()]] += 1          # distribute the remainder
    picked = []
    for b in range(nb):
        idx = np.where((T >= edges[b]) & (T < edges[b + 1]))[0]
        if len(idx) == 0 or want[b] == 0:
            continue
        n_even = int(round(want[b] * s["even_fraction"]))
        even = idx[np.linspace(0, len(idx) - 1, n_even + 2)[1:-1].round().astype(int)] if n_even else np.array([], int)
        local_even = [int(np.where(idx == e)[0][0]) for e in even]
        fps = _fps(F["feat"][idx], t_fs[idx], want[b] - len(local_even), local_even, s["min_gap_fs"], rng)
        picked += [(int(idx[j]), "even") for j in local_even] + [(int(idx[j]), "fps") for j in fps]
    picked.sort()
    return {"F": F, "picked": picked, "edges": edges, "want": want}


# ---------------------------------------------------------------------------- split
def split_of(time_ns: float, sp: dict) -> str:
    b = int(time_ns // sp["block_ns"])
    if b % sp["period"] in sp["test_blocks"]:
        return "test"
    if b % sp["period"] in sp["valid_blocks"]:
        return "valid"
    return "train"


# ---------------------------------------------------------------------------- strain
def _random_eps(rng, iso_lo, iso_hi, dev_sigma, dev_max) -> np.ndarray:
    iso = rng.uniform(iso_lo, iso_hi)
    d = rng.normal(0.0, dev_sigma, 6).clip(-dev_max, dev_max)
    E = np.array([[d[0], d[5], d[4]], [d[5], d[1], d[3]], [d[4], d[3], d[2]]])
    E -= np.trace(E) / 3 * np.eye(3)                      # deviatoric part only
    return iso * np.eye(3) + E


def make_strained(c: Campaign, frames: list[dict], mols, rng) -> list[dict]:
    st = c.cfg["strain"]
    n = st["n_frames"]
    if n <= 0:
        return []
    src = np.linspace(0, len(frames) - 1, n).round().astype(int)
    out = []
    for k, i in enumerate(src):
        fr = frames[i]
        rigid = k % 4 != 3                                # 3 of 4 rigid-molecule, 1 of 4 affine
        for attempt in range(8):
            scale = 0.5 ** attempt
            if rigid:
                eps = _random_eps(rng, st["iso_min"], st["iso_max"], st["dev_sigma"], st["dev_max"]) * scale
                pos, cell = geo.strain_rigid_molecules(fr["pos"], fr["cell"], mols, fr["symbols"], eps)
            else:
                eps = _random_eps(rng, -st["affine_max"], st["affine_max"], st["affine_max"] / 2, st["affine_max"]) * scale
                pos, cell = geo.strain_affine(fr["pos"], fr["cell"], eps)
            dmin = geo.min_intermolecular_distance(pos, cell, mols)
            if dmin >= st["min_contact_A"]:
                break
        else:
            continue
        info = dict(fr["info"])
        info.update({"config_type": "strained_rigid" if rigid else "strained_affine",
                     "strain_voigt": [eps[0, 0], eps[1, 1], eps[2, 2], eps[1, 2], eps[0, 2], eps[0, 1]],
                     "volume": geo.cell_volume(cell), "min_contact_A": dmin, "frame_id": f"s{k:04d}",
                     "parent_id": fr["info"]["frame_id"]})
        out.append({"symbols": fr["symbols"], "pos": pos, "cell": cell, "info": info})
    return out


# ---------------------------------------------------------------------------- main
def run(c: Campaign):
    s, sp = c.cfg["select"], c.cfg["split"]
    outdir = c.p("selection_dir")
    outdir.mkdir(parents=True, exist_ok=True)
    ch = choose_frames(c)
    F, picked = ch["F"], ch["picked"]
    steps = {int(F["step"][i]): (i, how) for i, how in picked}
    c.log(f"**select**: {len(picked)} fix-deform frames chosen; reading them from the dump",
          dump=c.path(s["dump"]), bins=f"{len(ch['edges']) - 1} x {s['bin_K']} K", per_bin=list(map(int, ch["want"])))

    frames, mols, broken = {}, None, []
    for fr in iter_lammps_dump(c.path(s["dump"]), want_steps=set(steps)):
        if mols is None:
            mols = geo.molecules(fr["symbols"], fr["pos"], fr["cell"])
            sizes = sorted(len(m) for m in mols)
            if sizes != [46] * (len(fr["symbols"]) // 46):
                raise RuntimeError(f"expected whole C20H26 molecules, got sizes {sizes}")
            nb = geo.bond_graph(fr["symbols"], fr["pos"], fr["cell"])
        ok, why = geo.check_molecules(fr["symbols"], fr["pos"], fr["cell"], mols)
        if not ok:
            broken.append((fr["step"], why))
            continue
        frames[fr["step"]] = fr                            # a repeated step (dump append) keeps the last copy
    missing = sorted(set(steps) - set(frames))
    out = []
    for k, step in enumerate(sorted(frames)):
        fr = frames[step]
        i, how = steps[step]
        pos = geo.make_whole(fr["symbols"], fr["pos"], fr["cell"], mols, nb)
        info = {"frame_id": f"f{k:04d}", "config_type": "fixdeform", "source": "mace-off23-small_fixdeform",
                "step": step, "time_ns": float(F["time_ns"][i]), "T_target": float(F["T"][i]), "pick": how,
                "volume": geo.cell_volume(fr["cell"]), "tbu_deg": F["tbu_deg"][i], "biphenyl_deg": F["bip_deg"][i]}
        info["split"] = split_of(info["time_ns"], sp)
        out.append({"symbols": fr["symbols"], "pos": pos, "cell": fr["cell"], "info": info})

    rng = np.random.default_rng(c.cfg["strain"]["seed"])
    strained = make_strained(c, out, mols, rng)
    allf = out + strained
    with open(outdir / "frames.extxyz", "w") as f:
        for fr in allf:
            write_extxyz_frame(f, fr["symbols"], fr["pos"], fr["cell"], None, fr["info"])
    (outdir / "molecules.json").write_text(json.dumps(mols))

    V = np.array([fr["info"]["volume"] for fr in allf])
    summ = {"fixdeform_frames": len(out), "strained_frames": len(strained), "total": len(allf),
            "broken_skipped": broken, "missing_steps": missing[:20], "n_missing": len(missing),
            "split_counts": {k: sum(fr["info"]["split"] == k for fr in allf) for k in ("train", "valid", "test")},
            "T_range_K": [float(min(fr["info"]["T_target"] for fr in out)), float(max(fr["info"]["T_target"] for fr in out))],
            "V_range_A3": [float(V.min()), float(V.max())],
            "picks": {h: sum(fr["info"]["pick"] == h for fr in out) for h in ("even", "fps")}}
    (outdir / "selection.json").write_text(json.dumps(summ, indent=1))
    c.log("**select done**: frames written", file=outdir / "frames.extxyz", **{k: v for k, v in summ.items() if k != "broken_skipped"},
          broken_skipped=len(broken))
    c.update_state("select", summ)
    try:
        plot(outdir, F, out, strained)
    except Exception as e:                                # plotting is optional
        print("plot skipped:", e)
    return summ


def plot(outdir: Path, F: dict, out: list[dict], strained: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    T = np.array([f["info"]["T_target"] for f in out])
    ax[0, 0].hist(F["T"], bins=50, color="0.8", label="all frames (scaled)", weights=np.full(len(F["T"]), len(T) / len(F["T"])))
    ax[0, 0].hist(T, bins=50, histtype="step", color="C0", label="selected")
    ax[0, 0].set(xlabel="T target (K)", ylabel="frames", title="temperature coverage")
    ax[0, 0].legend()
    tb = np.array([f["info"]["tbu_deg"] for f in out])
    for r in range(4):
        ax[0, 1].scatter(T, tb[:, r], s=3, label=f"tBu {r + 1}")
    ax[0, 1].set(xlabel="T target (K)", ylabel="rotor angle (deg)", title="tBu rotor angles of selected frames")
    bp = np.array([f["info"]["biphenyl_deg"] for f in out])
    ax[1, 0].scatter(T, bp[:, 0], s=3)
    ax[1, 0].scatter(T, bp[:, 1], s=3)
    ax[1, 0].set(xlabel="T target (K)", ylabel="twist (deg)", title="biphenyl twists of selected frames")
    V = [f["info"]["volume"] for f in out]
    ax[1, 1].scatter(T, V, s=3, label="fix-deform")
    if strained:
        ax[1, 1].scatter([f["info"]["T_target"] for f in strained], [f["info"]["volume"] for f in strained],
                         s=6, marker="x", label="strained")
    ax[1, 1].set(xlabel="T target (K)", ylabel="cell volume (A^3)", title="volumes to be labelled")
    ax[1, 1].legend()
    fig.savefig(outdir / "selection.png", dpi=150)
