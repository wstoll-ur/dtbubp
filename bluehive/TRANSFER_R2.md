# r2 transfer from Vermont nodes to Box and DGX

Run `bash bluehive/package_r2.sh` on bluehive3 from an updated checkout of
`wstoll-ur/dtbubp`. Requires Python >= 3.10 with numpy, ssh, rsync, tar and
sha256sum. Uses the existing node SSH access. It creates a unique snapshot under
`~/dtbubp-transfer/` and a `.tar.gz` archive with a companion `.sha256` file.
It never starts or stops a calculation or changes node files.

The archive contains:

- separate raw snapshots from bhx0125, bhx0124 and bhx0123, including CP2K
  input, coordinates, metadata and output tables; no wavefunctions;
- the exact original 2200 selected geometries, r2 configuration and parser code;
- merged accepted outputs and `parsed/{labels,train,valid,test}.xyz` with energy
  in eV, forces in eV/angstrom, and ASE-sign stress in eV/angstrom^3;
- `remaining.extxyz`, `remaining_ids.txt`, and ready per-frame inputs for frames
  lacking an accepted label;
- root `manifest.json` with node counts, rejected-output reasons and source
  provenance, plus `SHA256SUMS` for file integrity.

The parser requires normal termination, converged SCF, finite energy/forces/stress,
the expected number of forces and the existing force guard. It compares coordinates,
cell and split to the original frame, and compares the DFT input to the r2 deck.
Disagreeing duplicate labels or mismatched provenance stop packaging for review.
Snapshot files remain available for diagnosis when packaging fails.

Copy the archive and its checksum from BlueHive to the Mac's Box project, then
from the Mac to DGX using the existing `dgx` SSH alias. Keep scientific outputs
out of GitHub. Only these tools and notes are committed.

On Linux, verify the archive before unpacking:

```bash
sha256sum -c r2-transfer.XXXXXXXX.tar.gz.sha256
tar -xzf r2-transfer.XXXXXXXX.tar.gz
cd r2-transfer.XXXXXXXX
sha256sum -c SHA256SUMS
cat manifest.json
```

On macOS use `shasum -a 256 -c` in place of `sha256sum -c`.

The snapshot can be taken while bhx0123 still runs. Its accepted count may exceed
the user's 1689 count by the time files are copied; partially written outputs
become pending frames. Before launching pending work on DGX, stop only the
specific DtBuBP chunk launcher on bhx0123, or take a final refreshed snapshot
after it finishes. Do not stop unrelated CP2K processes. The existing
`vermont.sh status` / `vermont.sh kill TAG` workflow can identify and stop the
registered job when its registry is available in the original checkout.

DGX execution remains a separate step: validate r2 energies, forces and stresses
against a completed BlueHive frame using the DGX binary, then run the exact pending
IDs with the same PBE-D3(BJ)+C9/DZVP-SR reference. Do not use the Berkeley project's
BLYP configuration. Preserve each cell, atom order and original split. Successful
new DGX outputs should enter the same per-frame collection layout with explicit
backend provenance. No DGX reference calculation has been launched by packaging.
