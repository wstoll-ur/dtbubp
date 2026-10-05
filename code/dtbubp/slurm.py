"""Slurm job scripts (Leonardo, BlueHive): [slurm.*] account/partition/qos/extra; account and qos optional."""
from __future__ import annotations

import subprocess
from pathlib import Path


def write_script(slurm: dict, path: Path, name: str, body: str, time: str, workdir: Path,
                 array: str | None = None) -> Path:
    out = f"{path.parent}/{name}_%A_%a.out" if array else f"{path.parent}/{name}_%j.out"
    head = ["#!/bin/bash", f"#SBATCH --job-name={name}"]
    head += [f"#SBATCH --{k}={slurm[k]}" for k in ("account", "partition", "qos") if slurm.get(k)]
    head += [f"#SBATCH --time={time}", f"#SBATCH --output={out}"] + [f"#SBATCH {x}" for x in slurm.get("extra", [])]
    if array:
        head.append(f"#SBATCH --array={array}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(head) + "\n" + f"cd {workdir}\n" + body.strip() + "\n")
    path.chmod(0o755)
    return path


def submit(script: Path, dependency: str | None = None, dry: bool = False) -> str:
    cmd = ["sbatch", "--parsable"] + ([f"--dependency={dependency}"] if dependency else []) + [str(script)]
    if dry:
        print("  [dry-run]", " ".join(cmd))
        return "DRY"
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"sbatch failed for {script}: {r.stderr.strip()}")
    return r.stdout.strip().split(";")[0]
