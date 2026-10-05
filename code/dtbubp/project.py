"""Campaign folder: config.toml, automatic LOG.md and state.json (same pattern as 9MA npt_al)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

try:
    import tomllib                       # python >= 3.11
except ModuleNotFoundError:
    try:
        import tomli as tomllib          # python 3.10 with tomli
    except ModuleNotFoundError:          # no TOML parser: use the JSON copy written on every successful TOML read
        tomllib = None


class Campaign:
    """One labelling + fine-tuning campaign. Every relative path in config.toml is relative to it."""

    def __init__(self, config: Path):
        self.config_path = Path(config).resolve()
        self.root = self.config_path.parent
        # .<config stem>.json = parsed copy of config.toml + its sha256, for pythons without a TOML parser
        # (it is committed to git, so a clone on another cluster can use it; stale copies are refused).
        js = self.root / f".{self.config_path.stem}.json"
        raw = self.config_path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if tomllib is not None:
            self.cfg = tomllib.loads(raw.decode())
            try:
                old = json.loads(js.read_text()) if js.exists() else {}
                if old.get("toml_sha256") != sha:
                    js.write_text(json.dumps({"toml_sha256": sha, "cfg": self.cfg}, indent=1))
            except (OSError, ValueError):
                pass
        else:
            old = json.loads(js.read_text()) if js.exists() else {}
            if old.get("toml_sha256") != sha:
                raise SystemExit(f"no TOML parser in this python and {js} is missing or older than {self.config_path}: "
                                 "use python >= 3.11, or pip install tomli in your own venv")
            self.cfg = old["cfg"]

    # ---------------------------------------------------------------- paths
    def path(self, p) -> Path:
        p = Path(os.path.expandvars(os.path.expanduser(str(p))))
        return Path(os.path.normpath(p if p.is_absolute() else self.root / p))

    def p(self, key: str) -> Path:
        return self.path(self.cfg["paths"][key])

    # ---------------------------------------------------------------- log + state
    def log(self, msg: str, **kv):
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.root / "LOG.md", "a") as f:
            f.write(f"- `{now}` {msg}\n")
            for k, v in kv.items():
                f.write(f"    - {k}: {v}\n")
        print(msg, *(f"\n  {k}: {v}" for k, v in kv.items()))

    def state(self) -> dict:
        p = self.root / "state.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def update_state(self, key: str, value):
        s = self.state()
        s[key] = value
        (self.root / "state.json").write_text(json.dumps(s, indent=1, default=str))
