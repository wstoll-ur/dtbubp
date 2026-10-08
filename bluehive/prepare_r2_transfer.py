"""Validate separate node snapshots, collect E/F/stress, identify remaining frames.

No calculation or remote mutation. Original node files remain in node_snapshots.
"""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from dtbubp import cp2k, label, units
from dtbubp.io import read_extxyz, read_xyz, write_extxyz_frame, write_xyz
from dtbubp.project import Campaign


CAMPAIGN = "02_dft_dataset_r2_pbe-d3bj-c9_dzvpsr"


def canonical_input(text):
    return " ".join(re.sub(r"[!#].*", "", text).split())


def prepare(root):
    root = Path(root).resolve()
    c = Campaign(root / "Calculations" / CAMPAIGN / "config.toml")
    if cp2k.level_of(c.cfg["dft"]) != {
        "functional": "PBE", "dispersion": "D3BJ", "c9": True,
        "basis": "DZVP-MOLOPT-SR-GTH",
    } or c.cfg["dft"].get("polarizability", True):
        raise ValueError("Expected the r2 PBE-D3(BJ)+C9 deck without polarizability")
    frames = list(read_extxyz(c.p("selection_dir") / "frames.extxyz"))
    by_id = {f["info"]["frame_id"]: f for f in frames}
    if len(by_id) != len(frames):
        raise ValueError("Duplicate selected frame IDs")
    lab = c.root / "label"
    lab.mkdir()
    accepted = {}
    rejected = []
    node_counts = {}
    for node in sorted((root / "node_snapshots").iterdir()):
        counts = {"outputs": 0, "ended": 0, "usable": 0}
        for fd in sorted(node.iterdir()):
            if not fd.is_dir() or not (fd / "output.out").exists():
                continue
            fid = fd.name
            if fid not in by_id:
                raise ValueError(f"Unexpected frame {node.name}/{fid}")
            counts["outputs"] += 1
            r = cp2k.parse_output(fd / "output.out")
            counts["ended"] += int(r["ended"])
            reason = None
            if not r["ended"] or r["energy"] is None:
                reason = "unfinished_or_missing_energy"
            elif r["scf_failed"]:
                reason = "scf_not_converged"
            elif r["forces"] is None or np.asarray(r["forces"]).shape != (len(by_id[fid]["symbols"]), 3):
                reason = "missing_or_wrong_forces"
            elif r["stress"] is None or r["stress_unit"] not in units.STRESS_TO_EV_A3:
                reason = "missing_or_unknown_stress"
            elif not all(np.isfinite(x).all() for x in [r["energy"], r["forces"], r["stress"]]):
                reason = "nonfinite_label"
            elif np.abs(r["forces"] * units.HA_BOHR_EV_A).max() > c.cfg["dft"]["max_force_ev_a"]:
                reason = "force_guard"
            if reason:
                rejected.append({"node": node.name, "frame_id": fid, "reason": reason})
                continue
            f = by_id[fid]
            meta = json.loads((fd / "meta.json").read_text())
            _, symbols, positions = next(read_xyz(fd / "coord.xyz"))
            if (symbols != f["symbols"] or not np.allclose(positions, f["pos"], atol=1e-7, rtol=0)
                    or not np.allclose(meta["cell"], f["cell"], atol=1e-7, rtol=0)
                    or meta.get("frame_id") != fid or meta.get("split") != f["info"].get("split")):
                raise ValueError(f"Geometry/provenance mismatch: {node.name}/{fid}")
            expected = cp2k.make_input(c.cfg["dft"], f["cell"], "label")
            if canonical_input((fd / "input.inp").read_text()) != canonical_input(expected):
                raise ValueError(f"DFT input mismatch: {node.name}/{fid}; review before mixing labels")
            counts["usable"] += 1
            if fid in accepted:
                old = cp2k.parse_output(lab / fid / "output.out")
                if (abs(old["energy"] - r["energy"]) * units.HA_EV > 1e-4
                        or not np.allclose(old["forces"], r["forces"], atol=1e-4 / units.HA_BOHR_EV_A, rtol=0)
                        or not np.allclose(cp2k.cp2k_stress_to_ase(old["stress"], old["stress_unit"]),
                                          cp2k.cp2k_stress_to_ase(r["stress"], r["stress_unit"]),
                                          atol=1e-3 / units.EV_A3_GPA, rtol=0)):
                    raise ValueError(f"Disagreeing duplicate label: {fid}")
                continue
            target = lab / fid
            target.mkdir()
            for filename in ("input.inp", "coord.xyz", "meta.json", "output.out"):
                shutil.copy2(fd / filename, target / filename)
            accepted[fid] = node.name
        node_counts[node.name] = counts
    remaining = [f for f in frames if f["info"]["frame_id"] not in accepted]
    with (c.root / "remaining.extxyz").open("w") as handle:
        for f in remaining:
            write_extxyz_frame(handle, f["symbols"], f["pos"], f["cell"], f["arrays"], f["info"])
            fid = f["info"]["frame_id"]
            fd = lab / fid
            fd.mkdir()
            write_xyz(fd / "coord.xyz", f["symbols"], f["pos"], fid)
            (fd / "input.inp").write_text(cp2k.make_input(c.cfg["dft"], f["cell"], "label"))
            meta = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in f["info"].items()}
            meta["cell"] = f["cell"].tolist()
            (fd / "meta.json").write_text(json.dumps(meta, indent=2))
    (c.root / "remaining_ids.txt").write_text("".join(f["info"]["frame_id"] + "\n" for f in remaining))
    collected = label.collect(c)
    if collected["kept"] != len(accepted):
        raise ValueError("Collector count differs from validated label count")
    manifest = {"campaign": CAMPAIGN, "selected": len(frames), "accepted": len(accepted),
                "remaining": len(remaining), "nodes": node_counts, "accepted_source": accepted,
                "rejected_outputs": rejected, "collect": collected,
                "note": "Read-only snapshot; BlueHive may continue. No DGX job launched."}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    hashes = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            hashes.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n")
    (root / "SHA256SUMS").write_text("".join(hashes))
    print(json.dumps({k: manifest[k] for k in ("selected", "accepted", "remaining", "nodes")}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: prepare_r2_transfer.py SNAPSHOT_DIRECTORY")
    prepare(sys.argv[1])
