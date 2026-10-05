"""dtb: DFT labels (E, F, stress, dipole, polarizability) for DtBuBP frames, and MACE fine-tuning.

    python -m dtbubp.cli --config CONFIG <command>

  select                  choose the fix-deform frames + strained copies   (local: needs the dump)
  smoketest [--check]     one frame: label deck / numerical stress / finite-field polarizability
  label [--split S]       write every frame's CP2K input and submit the packed array job
  status                  labelling progress and wall time per frame
  collect                 CP2K outputs -> parsed/{labels,train,valid,test}.xyz
  train potential|dielectric|both
  evaluate potential|dielectric --model PATH      (runs inside the training jobs too)
  inputs                  print the label deck for the first frame (inspection)
  npt-setup [--force]     fixed-T CP2K-MACE NPT tests (config of 04_npt_*/); only temperatures not started yet
  npt-analyze             V/molecule vs experiment (V_cell/Z of the phase at each T), XRD structures, cells
  npt-diagnose            static MACE checks in Python: heads, DFT vs MACE pressure, E(V) scan, last NPT frames
  eos-scan / eos-dft / eos-analyze   relaxed volume scan with MACE r0, DFT on the same structures, comparison
  bench-setup [--only a,b] / bench-analyze   level-of-theory benchmark: CP2K CELL_OPT per functional/basis
Add --dry-run to label / smoketest / train to write everything without submitting.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .project import Campaign


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dtb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("cmd", choices=["select", "smoketest", "label", "status", "collect", "train", "evaluate", "inputs", "npt-setup", "npt-analyze", "npt-diagnose", "eos-scan", "eos-dft", "eos-analyze", "bench-setup", "bench-analyze"])
    ap.add_argument("what", nargs="?", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="npt-setup: also redo temperatures that already started")
    ap.add_argument("--split", default=None, choices=[None, "train", "valid", "test"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--only", default=None, help="bench-setup: comma-separated level names")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    c = Campaign(Path(a.config))

    if a.cmd == "select":
        from . import select
        select.run(c)
    elif a.cmd == "smoketest":
        from . import label
        ok = label.smoketest(c, check=a.check, dry=a.dry_run)
        if a.check:
            sys.exit(0 if ok else 1)
    elif a.cmd == "label":
        from . import label
        label.setup(c, dry=a.dry_run, only_split=a.split)
    elif a.cmd == "status":
        from . import label
        label.status(c)
    elif a.cmd == "collect":
        from . import label
        label.collect(c)
    elif a.cmd == "train":
        from . import train
        if a.what in ("potential", "both"):
            train.potential(c, dry=a.dry_run)
        if a.what in ("dielectric", "both"):
            train.dielectric(c, dry=a.dry_run)
        if a.what not in ("potential", "dielectric", "both"):
            ap.error("train potential|dielectric|both")
    elif a.cmd == "evaluate":
        from . import evaluate
        if not a.model:
            ap.error("--model is required")
        getattr(evaluate, a.what)(c, Path(a.model), a.device)
    elif a.cmd == "npt-setup":
        from . import npt
        npt.setup(c, dry=a.dry_run, force=a.force)
    elif a.cmd == "npt-analyze":
        from . import npt
        npt.analyze(c)
    elif a.cmd == "npt-diagnose":
        from . import diagnose
        diagnose.run(c, device=a.device)          # cuda if available (run it in a GPU job), else cpu
    elif a.cmd in ("eos-scan", "eos-dft", "eos-analyze"):
        from . import eos
        if a.cmd == "eos-scan":
            eos.scan(c, device=a.device)
        elif a.cmd == "eos-dft":
            eos.dft(c, dry=a.dry_run)
        else:
            eos.analyze(c)
    elif a.cmd in ("bench-setup", "bench-analyze"):
        from . import bench
        if a.cmd == "bench-setup":
            bench.setup(c, dry=a.dry_run, only=a.only.split(",") if a.only else None)
        else:
            bench.analyze(c)
    elif a.cmd == "inputs":
        from . import cp2k
        from .io import read_extxyz
        fr = next(read_extxyz(c.p("selection_dir") / "frames.extxyz"))
        print(cp2k.make_input(c.cfg["dft"], fr["cell"], a.what or "label"))


if __name__ == "__main__":
    main()
