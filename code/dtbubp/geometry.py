"""Periodic geometry: minimum image, molecules, making molecules whole, strains, dipole branches."""
from __future__ import annotations

from collections import deque

import numpy as np

BOND_CUT = {("C", "C"): 1.75, ("C", "H"): 1.30, ("H", "C"): 1.30}   # Angstrom; no H-H bonds
MASS = {"C": 12.011, "H": 1.008}


def mic(vec: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Minimum-image vectors (..., 3); cell rows = lattice vectors."""
    f = vec @ np.linalg.inv(cell)
    f -= np.round(f)
    return f @ cell


def distance_matrix(pos: np.ndarray, cell: np.ndarray) -> np.ndarray:
    return np.linalg.norm(mic(pos[:, None] - pos[None], cell), axis=-1)


def bond_graph(syms: list[str], pos: np.ndarray, cell: np.ndarray) -> list[list[int]]:
    d = distance_matrix(pos, cell)
    n = len(syms)
    nb = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cut = BOND_CUT.get((syms[i], syms[j]))
            if cut and d[i, j] < cut:
                nb[i].append(j)
                nb[j].append(i)
    return nb


def molecules(syms, pos, cell) -> list[list[int]]:
    """Connected components of the bond graph (atom indices, each sorted)."""
    nb = bond_graph(syms, pos, cell)
    seen, mols = set(), []
    for s in range(len(syms)):
        if s in seen:
            continue
        comp, q = [], deque([s])
        seen.add(s)
        while q:
            i = q.popleft()
            comp.append(i)
            for j in nb[i]:
                if j not in seen:
                    seen.add(j)
                    q.append(j)
        mols.append(sorted(comp))
    return mols


def make_whole(syms, pos, cell, mols, nb=None) -> np.ndarray:
    """Unwrap each molecule (BFS along bonds, minimum image), then put its centre of mass in the cell.

    Whole molecules matter for the dipole: with neutral, whole molecules the cell dipole is the sum of
    molecular dipoles and does not depend on how atoms were wrapped.
    """
    nb = nb or bond_graph(syms, pos, cell)
    out = np.array(pos, float).copy()
    m = np.array([MASS[s] for s in syms])
    inv = np.linalg.inv(cell)
    for mol in mols:
        root = mol[0]
        done = {root}
        q = deque([root])
        while q:
            i = q.popleft()
            for j in nb[i]:
                if j not in done:
                    out[j] = out[i] + mic(pos[j] - pos[i], cell)
                    done.add(j)
                    q.append(j)
        com = (m[mol, None] * out[mol]).sum(0) / m[mol].sum()
        shift = -np.floor(com @ inv) @ cell
        out[mol] += shift
    return out


def check_molecules(syms, pos, cell, mols_ref) -> tuple[bool, str]:
    """Same molecules as the reference, and every intramolecular bond still a bond (no fragmentation)."""
    mols = molecules(syms, pos, cell)
    if sorted(map(tuple, mols)) != sorted(map(tuple, mols_ref)):
        return False, f"molecule graph changed: {len(mols)} components, sizes {sorted(len(x) for x in mols)}"
    return True, ""


def min_intermolecular_distance(pos, cell, mols) -> float:
    lab = np.empty(len(pos), int)
    for k, mol in enumerate(mols):
        lab[mol] = k
    d = distance_matrix(pos, cell)
    d[lab[:, None] == lab[None]] = np.inf
    return float(d.min())


def strain_rigid_molecules(pos_whole, cell, mols, syms, eps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deform the cell by (1 + eps) and move whole molecules rigidly with their centres of mass.

    Intramolecular geometry is untouched, so only intermolecular distances change (what the stress
    of a molecular crystal is about). eps: symmetric 3x3 small-strain tensor.
    """
    F = np.eye(3) + eps
    new_cell = cell @ F.T                       # rows a_i -> F a_i
    m = np.array([MASS[s] for s in syms])
    out = pos_whole.copy()
    for mol in mols:
        com = (m[mol, None] * pos_whole[mol]).sum(0) / m[mol].sum()
        out[mol] += com @ F.T - com
    return out, new_cell


def strain_affine(pos, cell, eps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F = np.eye(3) + eps
    return pos @ F.T, cell @ F.T


def fold_dipole(mu: np.ndarray, cell: np.ndarray) -> tuple[np.ndarray, float]:
    """Berry-phase dipole (e*A) is defined modulo e*R (R a lattice vector). Return the branch closest
    to zero, and |mu| / (half the shortest quantum) as a warning measure (< 0.5 is comfortable).

    Justification: DtBuBP crystals are built from neutral, non-polar molecules in centrosymmetric
    cells, so the physical cell dipole is a small fluctuation, far below a polarization quantum.
    """
    c = np.linalg.solve(cell.T, mu)            # mu = sum_i c_i a_i
    c -= np.round(c)
    # also try neighbouring branches: for skewed cells the fractional rounding is not the shortest
    best = None
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                v = (c + [dx, dy, dz]) @ cell
                if best is None or np.linalg.norm(v) < np.linalg.norm(best):
                    best = v
    qmin = min(np.linalg.norm(a) for a in cell)
    return best, float(np.linalg.norm(best) / (0.5 * qmin))


def cell_volume(cell) -> float:
    return float(abs(np.linalg.det(cell)))


def lattice_parameters(cell) -> tuple[float, ...]:
    a, b, c = (np.linalg.norm(v) for v in cell)
    al = np.degrees(np.arccos(np.dot(cell[1], cell[2]) / (b * c)))
    be = np.degrees(np.arccos(np.dot(cell[0], cell[2]) / (a * c)))
    ga = np.degrees(np.arccos(np.dot(cell[0], cell[1]) / (a * b)))
    return a, b, c, al, be, ga
