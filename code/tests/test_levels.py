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


def test_apptainer_launcher_and_lanes(tmp_path):
    d = {"launcher": "apptainer", "container": "/scratch/x/cp2k.sif"}
    l = cp2k.launch(d, 28, cpus="28-55")
    assert l == ("apptainer exec --cleanenv --env OMP_NUM_THREADS=1 /scratch/x/cp2k.sif "
                 "mpirun -np 28 --bind-to core --cpu-set 28-55 cp2k.psmp")
    from dtbubp.label import _pool_body

    class C:
        cfg = {"dft": dict(d, frames_per_node=2, cores_per_node=56, mem_per_node_gb=400, frame_timeout_min=60),
               "env": {"dft": "module load apptainer"}}
    b = _pool_body(C(), tmp_path, "chunk_0.txt")
    assert "--cpu-set $lo-$hi" in b and "lo=$(( $1 * R ))" in b and "K=${DTB_LANES:-2}; R=${DTB_RANKS:-28}" in b
    assert "srun" not in b and "-np $R" in b


def test_mpirun_launcher():
    assert cp2k.launch({"launcher": "mpirun"}, 12, cpus="12-23") == "mpirun -np 12 --bind-to core --cpu-set 12-23 cp2k.psmp"
    d = {"launcher": "mpirun", "cp2k_exe": "cp2k.popt", "mpi_pin": "-genv I_MPI_PIN_PROCESSOR_LIST {cpus}"}
    assert cp2k.launch(d, 24, cpus="0-23") == "mpirun -np 24 -genv I_MPI_PIN_PROCESSOR_LIST 0-23 cp2k.popt"


def test_multinode_launcher():
    d = {"launcher": "mpirun", "hosts": "a:24,b:24", "mpi_prefix": "/x/ompi"}
    assert cp2k.launch(d, 48, cpus="0-47") == "mpirun --prefix /x/ompi -np 48 --host a:24,b:24 --map-by core --bind-to core -x PATH -x LD_LIBRARY_PATH -x OMP_NUM_THREADS cp2k.psmp"


def test_efs_only_deck_with_wfn_chain(tmp_path):
    d = dict(DFT, functional="PBE", dispersion="D3BJ", c9=True, dipole=True, polarizability=False, wfn_chain=True)
    t = cp2k.make_input(d, CELL)
    assert "&LINRES" not in t and "&MOMENTS" in t and "CALCULATE_C9_TERM     T" in t
    assert "SCF_GUESS  RESTART" in t and "WFN_RESTART_FILE_NAME  guess.wfn" in t and "&RESTART ON" in t
    t0 = cp2k.make_input(dict(DFT, dipole=False, polarizability=False), CELL)
    assert "&MOMENTS" not in t0 and "&LINRES" not in t0 and "SCF_GUESS  ATOMIC" in t0 and "&RESTART OFF" in t0
    from dtbubp.label import _pool_body

    class C:
        cfg = {"dft": dict(d, launcher="mpirun", frames_per_node=2, cores_per_node=24, mem_per_node_gb=60,
                           frame_timeout_min=60), "env": {"dft": "true"}}
    b = _pool_body(C(), tmp_path, "chunk_0.txt")
    assert "if true; then" in b and "cp \"$prev\" $f/guess.wfn" in b and "input_atomic.inp" in b
    (tmp_path / "pool.sh").write_text(b)
    import subprocess
    assert subprocess.run(["bash", "-n", str(tmp_path / "pool.sh")]).returncode == 0
