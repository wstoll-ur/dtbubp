"""Benchmark set-up (inputs, job script) and analysis on fake CP2K CELL_OPT output."""
from __future__ import annotations

import json
import subprocess

import numpy as np

from dtbubp import bench, geometry as geo
from dtbubp.io import write_xyz
from dtbubp.project import Campaign

from test_pipeline import read_lmp, HERE, CODE


def test_bench(tmp_path):
    cell0, types, pos0 = read_lmp(HERE / "data" / "cell298.lmp")
    syms = ["C" if t == 1 else "H" for t in types]
    root = tmp_path / "Calculations"
    (root / "02_dft_dataset").mkdir(parents=True)
    (root / "02_dft_dataset" / "config.toml").write_text((CODE / "config_template.toml").read_text())
    st = root / "04_npt_ramp_78-300K" / "fixedT_r0" / "eos" / "V430"
    st.mkdir(parents=True)
    write_xyz(st / "coord.xyz", syms, pos0, "x")
    (st / "meta.json").write_text(json.dumps({"V_per_molecule": 430.0, "n_molecules": 2, "cell": cell0.tolist()}))
    b = root / "01_dft_benchmark"
    b.mkdir()
    (b / "config.toml").write_text((CODE / "config_bench_template.toml").read_text())
    c = Campaign(b / "config.toml")
    sc = bench.setup(c, dry=True)
    assert subprocess.run(["bash", "-n", str(sc)]).returncode == 0
    s = sc.read_text()
    assert "--array=0-11" in s and "EXT_RESTART" in s and "--ntasks=112" in s
    d = b / "cellopt"
    names = (d / "levels.txt").read_text().split()
    assert len(names) == 12
    ctrl = (d / "01_blyp_d3_dzvpsr" / "input.inp").read_text()
    assert "RUN_TYPE CELL_OPT" in ctrl and "CALCULATE_C9" not in ctrl and "GTH-BLYP-q4" in ctrl and "DFTD3\n" in ctrl
    assert "{{" not in ctrl and "DZVP-MOLOPT-SR-GTH" in ctrl
    c9 = (d / "08_revpbe_d3bj_c9_tzv2p" / "input.inp").read_text()
    assert "PARAMETRIZATION REVPBE" in c9 and "DFTD3(BJ)" in c9 and "CALCULATE_C9_TERM     T" in c9
    assert "GTH-PBE-q1" in c9 and "TZV2P-MOLOPT-GTH" in c9 and "REFERENCE_FUNCTIONAL  revPBE" in c9
    assert "CALCULATE_C9" not in (d / "04_blyp_d3bj_tzv2p" / "input.inp").read_text()
    rv = (d / "09_rvv10_tzv2p" / "input.inp").read_text()
    assert "RVV10" in rv and "PAIR_POTENTIAL" not in rv
    assert bench.setup(c, dry=True, only=["03_blyp_d3_tzv2p"]) and (d / "levels.txt").read_text().split() == ["03_blyp_d3_tzv2p"]
    bench.setup(c, dry=True)
    # fake runs: one converged, one aborted, one running, rest queued
    V0 = geo.cell_volume(cell0)
    for n, target, done in (("01_blyp_d3_dzvpsr", 377.0, "conv"), ("03_blyp_d3_tzv2p", 395.0, "run")):
        rd = d / n
        with open(rd / "bench-1.cell", "w") as f:
            f.write("#   Step   Time [fs]  Ax Ay Az Bx By Bz Cx Cy Cz Volume [Angstrom^3]\n")
            for k, s_ in enumerate(np.linspace(1, (2 * target / V0) ** (1 / 3), 12)):
                cc = cell0 * s_
                f.write(f"{k} 0.0 " + " ".join(f"{x:.6f}" for x in cc.ravel()) + f" {geo.cell_volume(cc):.6f}\n")
        txt = "".join(f" OPT| Step number {k}\n OPT| Total energy [hartree]   -300.{k:03d}\n"
                      f" OPT| Internal pressure [bar]   {-5000 + 400 * k}\n OPT| Used time [s]   60.0\n" for k in range(12))
        if done == "conv":
            txt += " *** GEOMETRY OPTIMIZATION COMPLETED ***\n PROGRAM ENDED AT now\n"
        if n == "03_blyp_d3_tzv2p":          # an old aborted run at the top of the same file (CP2K appends)
            txt = " PROGRAM STARTED AT x\n *   [ABORT]   *\n * found an unknown keyword *\n PROGRAM STARTED AT y\n" + txt
        (rd / "output.out").write_text(txt)
    (d / "09_rvv10_tzv2p" / "output.out").write_text(
        " PROGRAM STARTED AT x\n *******\n * [ABORT]                       *\n *  \\___/  Unknown subsection GGA_X_RPW86 *\n")
    out = bench.analyze(c)
    assert out["01_blyp_d3_dzvpsr"]["status"] == "converged" and abs(out["01_blyp_d3_dzvpsr"]["V_per_molecule"] - 377) < 0.01
    assert out["03_blyp_d3_tzv2p"]["status"] == "running" and out["03_blyp_d3_tzv2p"]["steps"] == 12
    assert out["09_rvv10_tzv2p"]["status"] == "FAILED" and "RPW86" in out["09_rvv10_tzv2p"]["error"]
    assert out["10_pbe_d3bj_dzvpsr"]["status"] == "queued"
    assert (d / "bench_summary.txt").exists()
