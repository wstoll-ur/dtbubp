"""Label deck level of theory comes from [dft] (r1 relabel at PBE-D3(BJ)/TZV2P); defaults = r0 deck."""
import numpy as np

from dtbubp import cp2k
from dtbubp.slurm import write_script

DFT = {"cutoff": 600, "rel_cutoff": 60, "eps_default": "1.0E-16", "eps_scf": "1.0E-08", "max_scf": 50,
       "xc_smoothing": True, "linres": {"preconditioner": "FULL_ALL", "eps": "1.0E-08", "max_iter": 300}}
CELL = np.diag([8.0, 10.0, 11.0])


def test_default_level_is_r0():
    t = cp2k.make_input(DFT, CELL)
    assert "&XC_FUNCTIONAL BLYP" in t and "TYPE                  DFTD3\n" in t
    assert t.count("BASIS_SET  DZVP-MOLOPT-SR-GTH") == 2 and "GTH-BLYP-q4" in t and "GTH-BLYP-q1" in t
    assert "CALCULATE_C9_TERM" not in t


def test_pbe_d3bj_tzv2p():
    t = cp2k.make_input(dict(DFT, functional="PBE", dispersion="D3BJ", basis="TZV2P-MOLOPT-GTH"), CELL)
    assert "&XC_FUNCTIONAL PBE" in t and "DFTD3(BJ)" in t and "REFERENCE_FUNCTIONAL  PBE" in t
    assert t.count("BASIS_SET  TZV2P-MOLOPT-GTH") == 2 and "GTH-PBE-q4" in t and "GTH-PBE-q1" in t
    assert "BLYP" not in t.split("do not edit by hand.")[1]
    assert "&POLAR" in t and "PERIODIC   T" in t


def test_launch_and_optional_account(tmp_path):
    assert cp2k.launch({}, 28).endswith("--mpi=pmi2 --cpu-bind=cores cp2k.popt")
    l = cp2k.launch({"cp2k_exe": "cp2k.psmp", "mpi_flags": "--mpi=pmix"}, 24, "--exact")
    assert l == "srun --exact --ntasks=24 --cpus-per-task=1 --mpi=pmix cp2k.psmp"
    s = write_script({"partition": "standard", "account": "", "extra": ["--nodes=1"]}, tmp_path / "j.sh", "x",
                     "echo", "01:00:00", tmp_path).read_text()
    assert "--partition=standard" in s and "--account" not in s and "--qos" not in s
