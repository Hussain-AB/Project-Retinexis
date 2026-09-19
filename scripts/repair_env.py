#!/usr/bin/env python3
"""Repair a venv whose installed packages are missing their dependencies.

    python scripts/repair_env.py            # dry run: show what is broken
    python scripts/repair_env.py --write    # install the missing packages

Why this exists rather than just `pip install -r requirements.txt`:

pip only resolves dependencies for packages it is actually installing. Every
package in requirements.txt is already "satisfied" in this venv, so pip marks
them done and never looks at THEIR dependencies -- which is exactly what is
missing here. transformers 5.3.0 is installed and cannot import, because
`packaging` is absent, and `pip install -r requirements.txt` will not fix it.

`pip check` reports those broken edges. This reads them, installs what is
missing, and repeats until the graph is consistent or it stops making progress.
Run on a LOGIN node -- compute nodes have no outbound network.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

# "wandb 0.25.1 requires platformdirs, which is not installed."
# "transformers 5.3.0 requires packaging>=20.0, which is not installed."
# "datasets 4.7.0 requires dill<0.4.1,>=0.3.0, which is not installed."
# The requirement may carry version specifiers, and those specifiers may
# themselves contain commas -- so capture everything up to the trailing clause
# and take the distribution name off the front.
MISSING = re.compile(r"^\S+ \S+ requires (.+?), which is not installed\.?$")
NAME = re.compile(r"^[A-Za-z0-9._-]+")
# "spam 1.0 has requirement eggs>=2, but you have eggs 1.0."
CONFLICT = re.compile(r"^(\S+) \S+ has requirement (\S+), but you have (\S+ \S+)\.?$")

MAX_ROUNDS = 6


def pip(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pip", *args],
                          capture_output=True, text=True)


def scan() -> tuple[set[str], list[str]]:
    out = pip("check")
    missing: set[str] = set()
    conflicts: list[str] = []
    for line in (out.stdout + out.stderr).splitlines():
        line = line.strip()
        m = MISSING.match(line)
        if m:
            name = NAME.match(m.group(1))
            if name:
                missing.add(name.group(0))
            continue
        if CONFLICT.match(line):
            conflicts.append(line)
    return missing, conflicts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="install (default: dry run)")
    args = ap.parse_args()

    print(f"python: {sys.executable}\n")
    if sys.prefix == sys.base_prefix:
        print("WARNING: this is not a virtualenv. Activate myenv first:")
        print("  source scripts/ibex/env_setup.sh\n")

    seen: set[str] = set()
    for rnd in range(1, MAX_ROUNDS + 1):
        missing, conflicts = scan()
        if not missing:
            print(f"round {rnd}: no missing dependencies")
            if conflicts:
                print("\nVersion conflicts remain -- these need a human decision, "
                      "since resolving them can move a pinned package:")
                for c in conflicts:
                    print(f"  {c}")
            print("\npip check is clean of missing packages.")
            print("Freeze it:  python -m pip freeze > environment_frozen.txt")
            return 0

        new = sorted(missing - seen)
        print(f"round {rnd}: {len(missing)} missing -> {' '.join(sorted(missing))}")
        if not new:
            print("\nNo progress: the same packages are still missing after an install.")
            print("Install them by hand and re-run.")
            return 1
        seen |= missing

        if not args.write:
            print("\ndry run -- rerun with --write to install.")
            print(f"equivalent: python -m pip install {' '.join(sorted(missing))}")
            return 1

        print(f"  installing: {' '.join(new)}")
        res = pip("install", *new)
        tail = (res.stdout + res.stderr).strip().splitlines()[-3:]
        for t in tail:
            print(f"    {t}")
        if res.returncode != 0:
            print("\npip install failed. Full output:\n")
            print(res.stdout[-3000:], res.stderr[-3000:])
            return 1

    print(f"\nstopped after {MAX_ROUNDS} rounds -- still not consistent.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
