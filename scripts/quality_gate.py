"""Development quality gate: pylint + unit tests + coverage in one shot.

Every AI coding agent MUST run this before committing:

    python scripts/quality_gate.py

Exits non-zero unless:
  * pylint reports 10.00/10 on every tracked ``*.py`` file
    (config: ``[tool.pylint]`` in ``pyproject.toml``)
  * all unit tests pass with >= 90% coverage
    (config: ``[tool.pytest.ini_options]`` in ``pyproject.toml``)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(command: list[str], description: str) -> None:
    print(f"== {description} ==", flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        print(f"FAILED: {description}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    py_files = tracked.stdout.split()
    if not py_files:
        print("No tracked Python files found.", file=sys.stderr)
        raise SystemExit(1)

    run(
        [sys.executable, "-m", "pylint", *py_files],
        "pylint (score must be 10.00/10)",
    )
    run(
        [sys.executable, "-m", "pytest", "tests/unit_tests"],
        "pytest (all unit tests + coverage >= 90%)",
    )
    print("Quality gate passed.")


if __name__ == "__main__":
    main()
