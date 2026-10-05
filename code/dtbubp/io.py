"""File formats: LAMMPS text dumps, plain xyz, extended xyz, simple column tables.

numpy only, so selection and collection run on any login node without ASE.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np

TYPE_TO_SYMBOL = {1: "C", 2: "H"}       # the fixdeform dumps: dump_modify element C H


# ============================================================================ LAMMPS dump
def cell_from_bounds(lines: list[str], triclinic: bool) -> np.ndarray:
    """LAMMPS 'ITEM: BOX BOUNDS' lines -> 3x3 cell matrix, rows = a, b, c (Angstrom).

    LAMMPS writes the bounding box (xlo_bound ...), not xlo, for triclinic boxes; undo that.
    """
    v = np.array([ln.split()[:3] if triclinic else ln.split()[:2] + ["0"] for ln in lines], float)
    xy, xz, yz = v[:, 2]
    xlo = v[0, 0] - min(0.0, xy, xz, xy + xz)
    xhi = v[0, 1] - max(0.0, xy, xz, xy + xz)
    ylo = v[1, 0] - min(0.0, yz)
    yhi = v[1, 1] - max(0.0, yz)
    zlo, zhi = v[2, 0], v[2, 1]
    return np.array([[xhi - xlo, 0.0, 0.0], [xy, yhi - ylo, 0.0], [xz, yz, zhi - zlo]])


def iter_lammps_dump(path: Path, want_steps: set[int] | None = None) -> Iterator[dict]:
    """Yield frames {step, cell, ids, symbols, pos} of a LAMMPS text dump, atoms sorted by id.

    The fixdeform dump has no `dump_modify sort id`, so atom order changes between frames:
    every frame is sorted by id here. If `want_steps` is given, frames whose step is not in it
    are skipped without being parsed (fast scan of a 750 MB file).
    """
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(f.readline())
            f.readline()                                   # ITEM: NUMBER OF ATOMS
            n = int(f.readline())
            boxhdr = f.readline()
            box = [f.readline() for _ in range(3)]
            cols = f.readline().split()[2:]
            if want_steps is not None and step not in want_steps:
                for _ in range(n):
                    f.readline()
                continue
            body = [f.readline() for _ in range(n)]
            if len(body[-1].split()) < len(cols):          # truncated last frame
                return
            data = np.array(" ".join(body).split(), float).reshape(n, len(cols))
            ids = data[:, cols.index("id")].astype(int)
            order = np.argsort(ids)
            if "element" in cols:
                raise ValueError("element column not supported; dump type instead")
            types = data[order, cols.index("type")].astype(int)
            ix = [cols.index(k) for k in ("x", "y", "z")] if "x" in cols else [cols.index(k) for k in ("xu", "yu", "zu")]
            yield {"step": step, "cell": cell_from_bounds(box, "xy" in boxhdr), "ids": ids[order],
                   "symbols": [TYPE_TO_SYMBOL[t] for t in types], "pos": data[order][:, ix]}


# ============================================================================ plain xyz
def read_xyz(path: Path) -> Iterator[tuple[str, list[str], np.ndarray]]:
    """Yield (comment, symbols, Nx3) for each frame of a plain xyz file."""
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.strip():
                continue
            n = int(line.split()[0])
            comment = f.readline().rstrip("\n")
            syms, xyz = [], np.empty((n, 3))
            for i in range(n):
                t = f.readline().split()
                if len(t) < 4:
                    return
                syms.append(t[0])
                xyz[i] = [float(t[1]), float(t[2]), float(t[3])]
            yield comment, syms, xyz


def write_xyz(path: Path, syms, xyz, comment: str = ""):
    with open(path, "w") as f:
        f.write(f"{len(syms)}\n{comment}\n")
        for s, x in zip(syms, xyz):
            f.write(f"{s:<2s} {x[0]:16.10f} {x[1]:16.10f} {x[2]:16.10f}\n")


# ============================================================================ extended xyz
def _kv_header(line: str) -> dict:
    out = {}
    for m in re.finditer(r'(\w+)=("([^"]*)"|\S+)', line):
        out[m.group(1)] = m.group(3) if m.group(3) is not None else m.group(2)
    return out


def _parse_value(v: str):
    toks = v.split()
    try:
        vals = [float(t) for t in toks]
    except ValueError:
        return v
    if len(vals) == 1 and len(toks) == 1 and re.fullmatch(r"[-+]?\d+", toks[0]):
        return int(toks[0])
    return np.array(vals) if len(vals) > 1 else vals[0]


def read_extxyz(path: Path) -> Iterator[dict]:
    """Yield frames {info, symbols, pos, cell, arrays} of an extended-xyz file (our own writer's subset)."""
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.strip():
                continue
            n = int(line.split()[0])
            raw = _kv_header(f.readline())
            props = raw.pop("Properties", "species:S:1:pos:R:3").split(":")
            cols = [(props[i], props[i + 1], int(props[i + 2])) for i in range(0, len(props), 3)]
            rows = [f.readline().split() for _ in range(n)]
            arrays, c = {}, 0
            for name, typ, w in cols:
                vals = [r[c:c + w] for r in rows]
                arrays[name] = np.array(vals, dtype=str if typ == "S" else float)
                c += w
            syms = list(arrays.pop("species")[:, 0])
            pos = arrays.pop("pos")
            cell = np.array(raw.pop("Lattice").split(), float).reshape(3, 3) if "Lattice" in raw else None
            raw.pop("pbc", None)
            info = {k: _parse_value(v) for k, v in raw.items()}
            yield {"info": info, "symbols": syms, "pos": pos, "cell": cell, "arrays": arrays}


def write_extxyz_frame(f, syms, pos, cell, arrays: dict | None = None, info: dict | None = None):
    """One extended-xyz frame. arrays = {name: (N,3) array}; info values: scalars, strings or arrays."""
    lat = " ".join(f"{v:.10f}" for v in np.asarray(cell).ravel())
    props = "species:S:1:pos:R:3" + "".join(f":{k}:R:{np.asarray(v).reshape(len(syms), -1).shape[1]}"
                                            for k, v in (arrays or {}).items())
    extra = []
    for k, v in (info or {}).items():
        if isinstance(v, (list, tuple, np.ndarray)):
            extra.append(f'{k}="' + " ".join(f"{x:.12e}" for x in np.asarray(v, float).ravel()) + '"')
        elif isinstance(v, bool):
            extra.append(f"{k}={'T' if v else 'F'}")
        elif isinstance(v, (int, np.integer)):
            extra.append(f"{k}={int(v)}")
        elif isinstance(v, (float, np.floating)):
            extra.append(f"{k}={float(v):.12e}")
        else:
            s = str(v)
            extra.append(f'{k}="{s}"' if " " in s else f"{k}={s}")
    f.write(f'{len(syms)}\nLattice="{lat}" Properties={props} ' + " ".join(extra) + ' pbc="T T T"\n')
    cols = [np.asarray(pos, float)] + [np.asarray(v, float).reshape(len(syms), -1) for v in (arrays or {}).values()]
    M = np.hstack(cols)
    for s, row in zip(syms, M):
        f.write(f"{s:<2s}" + "".join(f" {x:16.10f}" for x in row) + "\n")


# ============================================================================ tables
def read_table(path: Path) -> np.ndarray:
    """Whitespace columns, '#' comments skipped."""
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        try:
            rows.append([float(x) for x in line.split()])
        except ValueError:
            continue
    w = min(len(r) for r in rows)
    return np.array([r[:w] for r in rows])
