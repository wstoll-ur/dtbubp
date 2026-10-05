"""MACE training jobs: (a) the potential (E, F, stress) fine-tuned from MACE-OFF23-small, exported for CP2K;
(b) the dielectric model (dipole, polarizability), fine-tuned from MACE-MDP (or trained from scratch).

Both use the split written by `dtb collect` (train.xyz / valid.xyz / test.xyz), never a random split
(neighbouring fix-deform frames are 50 fs apart and strongly correlated).
"""
from __future__ import annotations

from pathlib import Path

from .project import Campaign
from .slurm import submit, write_script


def _args(lst) -> str:
    return " \\\n  ".join(lst)


def potential(c: Campaign, dry: bool = False) -> str:
    t, paths = c.cfg["train_potential"], c.cfg["paths"]
    d = c.p("potential_dir")
    d.mkdir(parents=True, exist_ok=True)
    data = c.p("parsed_dir")
    name = t["name"]
    base = [f'--name="{name}"', f'--foundation_model="{c.p("foundation_potential")}"',
            f'--train_file="{data / "train.xyz"}"', f'--valid_file="{data / "valid.xyz"}"',
            f'--test_file="{data / "test.xyz"}"', f'--seed={t["seed"]}'] + t["args"]
    body = c.cfg["env"]["mace"].strip() + f"""
export PYTHONPATH={c.p("code_dir")}:$PYTHONPATH
cd {d}
mace_run_train \\
  {_args(base)}
M={name}_stagetwo.model; [ -f $M ] || M={name}.model
[ -f $M ] || {{ echo "training produced no model"; exit 1; }}
rm -f checkpoints/*_epoch-*.pt 2>/dev/null
python -m dtbubp.cli --config {c.config_path} evaluate potential --model {d}/$M || echo "evaluation failed"
deactivate 2>/dev/null
{c.cfg['env']['cp2k_mace'].strip()}
python {c.p('create_cp2k_model')} {d}/$M --dtype float64 || exit 1
ls -l {d}/$M-cp2k.pth && echo "CP2K model: {d}/$M-cp2k.pth"
"""
    sc = write_script(c.cfg["slurm"]["gpu"], d / "train_job.sh", "dtb_pot", body, t["time_limit"], d)
    jid = submit(sc, dry=dry)
    c.log("**potential training submitted**", job=jid, dir=d, foundation=c.p("foundation_potential"))
    return jid


def dielectric(c: Campaign, dry: bool = False) -> str:
    t = c.cfg["train_dielectric"]
    d = c.p("dielectric_dir")
    d.mkdir(parents=True, exist_ok=True)
    data = c.p("parsed_dir")
    name = t["name"]
    base = [f'--name="{name}"', f'--train_file="{data / "train.xyz"}"', f'--valid_file="{data / "valid.xyz"}"',
            f'--test_file="{data / "test.xyz"}"', f'--seed={t["seed"]}'] + t["args"]
    if t["mode"] == "finetune_mdp":            # needs mace-torch from the main branch (>= July 2026)
        base += [f'--foundation_model="{c.p("foundation_dielectric")}"', "--finetune_dipoles_polarizabilities",
                 "--multiheads_finetuning=False"]
    elif t["mode"] == "scratch":
        base += t["scratch_args"]
    else:
        raise ValueError(f"train_dielectric.mode = {t['mode']!r}: use 'scratch' or 'finetune_mdp'")
    body = c.cfg["env"]["mace"].strip() + f"""
export PYTHONPATH={c.p("code_dir")}:$PYTHONPATH
cd {d}
mace_run_train \\
  {_args(base)}
M={name}_stagetwo.model; [ -f $M ] || M={name}.model
[ -f $M ] || {{ echo "training produced no model"; exit 1; }}
rm -f checkpoints/*_epoch-*.pt 2>/dev/null
python -m dtbubp.cli --config {c.config_path} evaluate dielectric --model {d}/$M || echo "evaluation failed"
"""
    sc = write_script(c.cfg["slurm"]["gpu"], d / "train_job.sh", "dtb_diel", body, t["time_limit"], d)
    jid = submit(sc, dry=dry)
    c.log(f"**dielectric training submitted** ({t['mode']})", job=jid, dir=d)
    return jid
