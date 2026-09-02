#!/usr/bin/env python
"""Build script for offline benchmark package.

Creates a standalone, self-contained package suitable for upload
to a closed, air-gapped environment for running benchmarks.

Usage:
    python scripts/build_benchmark.py [options]

Quick (same platform as build machine):
    python scripts/build_benchmark.py --with-wheels --with-installers --zip

Cross-platform (e.g. Windows building for Linux x86_64 + CPython 3.11):
    python scripts/build_benchmark.py \
        --with-wheels --with-installers --zip \
        --wheel-platform linux_x86_64 --wheel-python-version 311

Cross-platform for Ascend ARM servers (aarch64):
    python scripts/build_benchmark.py \
        --with-wheels --with-installers --zip \
        --wheel-platform linux_aarch64 --wheel-python-version 311

Full help: see OFFLINE_PACKAGING.md (bilingual) at the repository root.
"""

from __future__ import annotations

import argparse
import platform as _platform_mod
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_PKG = ROOT / "langchain_ascend"
SRC_BENCH = ROOT / "benchmark"

INSTALL_SH = r"""#!/bin/bash
# ============================================================
# LangChain Ascend Affinity — Offline Benchmark Installer
# Target: Linux (Ascend NPU / MindIE / vLLM-Ascend)
# Usage:  bash install_offline.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
PIP="${PYTHON} -m pip"

echo "=============================================="
echo " LangChain Ascend Affinity — Offline Installer"
echo "=============================================="
echo "Python: $(${PYTHON} --version)"
echo "Pip:    $(${PIP} --version)"
echo

# Check Python version (need >= 3.11 for deepagents 0.7.6+)
PY_MAJOR=$(${PYTHON} -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(${PYTHON} -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    echo "ERROR: Python >= 3.11 required (deepagents 0.7.6+ needs it)"
    echo "  Current: $(${PYTHON} --version)"
    echo "  For oj-* only (openjiuwen) without deepagents, install manually"
    echo "  after lowering the required versions in requirements.txt."
    exit 1
fi

# Step 1: Install dependencies from local wheels
echo "[1/3] Installing dependencies from wheels/ ..."
if [ -d "wheels" ] && [ -n "$(ls -A wheels 2>/dev/null)" ]; then
    echo "  Using --no-index --find-links=wheels"
    if ! ${PIP} install --no-index --find-links=wheels \
        -r requirements.txt 2>&1 | tail -20; then
        echo
        echo "WARNING: Some wheels may not match your platform."
        echo "  Pure Python wheels (py3-none-any) work cross-platform."
        echo "  C-extension wheels (numpy, orjson, pydantic-core, etc.)"
        echo "  must match OS + Python version + CPU architecture."
        echo
        echo "  Fix: re-run build_benchmark.py --with-wheels on a machine"
        echo "  matching the target platform, or manually copy correct wheels"
        echo "  into wheels/. See OFFLINE_PACKAGING.md §3."
        exit 1
    fi
else
    echo "  WARNING: wheels/ directory missing or empty!"
    echo "  Nothing to install offline. Refusing to continue."
    exit 1
fi

# Step 2: Install the langchain-ascend-affinity package itself
echo
echo "[2/3] Installing langchain-ascend-affinity from source..."
if [ -f "pyproject.toml" ]; then
    ${PIP} install --no-build-isolation -e . 2>&1 | tail -5 || {
        echo "  Trying without editable mode..."
        ${PIP} install --no-build-isolation . 2>&1 | tail -5
    }
else
    echo "  pyproject.toml not found, skipping package install."
    echo "  You may need: export PYTHONPATH=\$PWD:\$PYTHONPATH"
fi

# Step 3: Verify installation
echo
echo "[3/3] Verifying installation..."
${PYTHON} -c "
import langchain_ascend
from langchain_ascend import AscendAffinityChatModel
print('  langchain_ascend OK:', langchain_ascend.__file__)
try:
    import deepagents
    print('  deepagents OK:', deepagents.__version__)
except ImportError:
    print('  deepagents MISSING - lc-* agents will not run')
try:
    import langchain_openai
    print('  langchain_openai OK')
except ImportError:
    print('  langchain_openai MISSING')
try:
    import openjiuwen  # pylint: disable=unused-import
    print('  openjiuwen OK (oj-* agents available)')
except ImportError:
    print('  openjiuwen MISSING - oj-* agents will be skipped (install from internal index)')
try:
    from benchmark import run_benchmark  # noqa: F401
    print('  benchmark harness OK')
except ImportError as e:
    print('  benchmark harness ERROR:', e)
"

echo
echo "=============================================="
echo " Installation complete!"
echo "=============================================="
echo
echo "Run benchmark:"
echo "  python benchmark/run_benchmark.py \\"
echo "    --engine-url http://<host>:<port>/v1 \\"
echo "    --model <model-name> \\"
echo "    --api-key <api-key>"
echo
echo "Or with environment variables:"
echo "  export ASCEND_ENGINE_URL=http://<host>:<port>/v1"
echo "  export ASCEND_MODEL=<model-name>"
echo "  export ASCEND_API_KEY=<api-key>"
echo "  python benchmark/run_benchmark.py"
"""

INSTALL_PS1 = r"""# ============================================================
# LangChain Ascend Affinity — Offline Benchmark Installer
# Target: Windows (LM Studio / ascend-sim benchmarking)
# Usage:  .\install_offline.ps1
# ============================================================

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonCmd = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$PipCmd = "$PythonCmd -m pip"

Write-Host "=============================================="
Write-Host " LangChain Ascend Affinity — Offline Installer"
Write-Host "=============================================="
Write-Host "Python: $(& $PythonCmd --version)"
Write-Host "Pip:    $(& $PythonCmd -m pip --version)"
Write-Host ""

# Check Python version
$pyVersion = & $PythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$pyMajor = [int]($pyVersion -split '\.')[0]
$pyMinor = [int]($pyVersion -split '\.')[1]
if ($pyMajor -lt 3 -or ($pyMajor -eq 3 -and $pyMinor -lt 11)) {
    Write-Host "ERROR: Python >= 3.11 required (deepagents 0.7.6+ needs it)" -ForegroundColor Red
    Write-Host "  Current: $pyVersion"
    exit 1
}

# Step 1: Install dependencies from local wheels
Write-Host "[1/3] Installing dependencies from wheels/ ..."
$wheelsPath = Join-Path $ScriptDir "wheels"
if (Test-Path $wheelsPath) {
    $hasWheels = (Get-ChildItem "$wheelsPath\*.whl" -ErrorAction SilentlyContinue).Count -gt 0
    if ($hasWheels) {
        Write-Host "  Using --no-index --find-links=wheels"
        & cmd /c "$PipCmd install --no-index --find-links=wheels -r requirements.txt"
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "WARNING: Some wheels may not match your platform." -ForegroundColor Yellow
            Write-Host "  See OFFLINE_PACKAGING.md section 3 for cross-platform wheel build steps."
            exit 1
        }
    } else {
        Write-Host "  WARNING: wheels/ exists but has no .whl files." -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "  WARNING: wheels/ directory missing!" -ForegroundColor Yellow
    exit 1
}

# Step 2: Install the langchain-ascend-affinity package itself
Write-Host ""
Write-Host "[2/3] Installing langchain-ascend-affinity from source..."
if (Test-Path "pyproject.toml") {
    & cmd /c "$PipCmd install --no-build-isolation -e ."
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Trying without editable mode..." -ForegroundColor Yellow
        & cmd /c "$PipCmd install --no-build-isolation ."
    }
} else {
    Write-Host "  pyproject.toml not found, skipping package install." -ForegroundColor Yellow
}

# Step 3: Verify installation
Write-Host ""
Write-Host "[3/3] Verifying installation..."
& $PythonCmd -c @"
import langchain_ascend
from langchain_ascend import AscendAffinityChatModel
print('  langchain_ascend OK:', langchain_ascend.__file__)
try:
    import deepagents
    print('  deepagents OK:', deepagents.__version__)
except ImportError:
    print('  deepagents MISSING - lc-* agents will not run')
try:
    import langchain_openai
    print('  langchain_openai OK')
except ImportError:
    print('  langchain_openai MISSING')
try:
    from benchmark import run_benchmark  # noqa: F401
    print('  benchmark harness OK')
except ImportError as e:
    print('  benchmark harness ERROR:', e)
"@

Write-Host ""
Write-Host "=============================================="
Write-Host " Installation complete!"
Write-Host "=============================================="
Write-Host ""
Write-Host "Run benchmark:"
Write-Host "  python benchmark/run_benchmark.py --engine-url http://<host>:<port>/v1 --model <model-name> --api-key <api-key>"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build offline benchmark package for air-gapped environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See OFFLINE_PACKAGING.md / OFFLINE_PACKAGING.zh-CN.md at the repo root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "ascend-benchmark-offline",
        help="Output directory (default: ./ascend-benchmark-offline)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        default=False,
        help="Also create a <output-dir>.zip archive",
    )
    parser.add_argument(
        "--with-wheels",
        action="store_true",
        default=False,
        help="Run `pip download` to fetch wheels into wheels/ after build",
    )
    parser.add_argument(
        "--with-installers",
        action="store_true",
        default=False,
        help="Write install_offline.sh and install_offline.ps1 into the package",
    )
    parser.add_argument(
        "--wheel-platform",
        type=str,
        default="current",
        help=(
            "Target platform for wheel download: 'current' (same machine), or a "
            "pip platform tag like 'linux_x86_64', 'linux_aarch64', 'win_amd64', "
            "'macosx_11_0_arm64' etc."
        ),
    )
    parser.add_argument(
        "--wheel-python-version",
        type=str,
        default="",
        help=(
            "CPython version for cross-platform download, e.g. '311' for "
            "Python 3.11. Required together with --wheel-platform for "
            "cross-platform builds. Defaults to the running Python version."
        ),
    )
    parser.add_argument(
        "--index-url",
        type=str,
        default="",
        help="Override PyPI index URL for wheel download (e.g. corporate mirror)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        default=False,
        help="Do NOT wipe output dir before build (preserves existing wheels)",
    )
    return parser.parse_args()


def _clean_pycache(path: Path) -> None:
    """Remove all __pycache__ directories under path."""
    for d in path.rglob("__pycache__"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def copy_package_structure(output: Path) -> None:
    """Copy langchain_ascend source to output dir."""
    out = output / "langchain_ascend"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SRC_PKG, out)
    _clean_pycache(out)


def copy_benchmark_files(output: Path) -> None:
    """Copy benchmark harness files to output dir, clearing runtime artefacts."""
    out = output / "benchmark"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SRC_BENCH, out)
    _clean_pycache(out)
    # Remove runtime artefacts (old reports, logs, cached data)
    reports_dir = out / "reports"
    if reports_dir.exists():
        for f in reports_dir.iterdir():
            if f.is_file():
                f.unlink()
    for pattern in ("*.log", "*.jsonl"):
        for f in out.glob(pattern):
            f.unlink(missing_ok=True)


def generate_requirements(output: Path) -> None:
    """Generate a pinned requirements.txt from the current environment.

    Falls back to a hand-authored (non-pinned) list if ``pip freeze``
    fails — in that case the user should run with ``--with-wheels`` on
    a machine whose environment mirrors the target.
    """
    # Prefer pinned: run `pip freeze` and filter down to actual project deps.
    req_lines: list[str] = []
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, check=True,
        )
        # Include everything — --with-wheels will download the whole set,
        # and an air-gapped machine must have everything pinned.
        req_lines.extend(
            line.rstrip() for line in freeze.stdout.splitlines()
            if line.strip() and not line.startswith("-e ") and "==" in line
        )
    except (subprocess.CalledProcessError, OSError):
        # Fallback: hand-authored range-based requirements.
        req_lines = [
            "# Fallback range-based requirements — generated without pip freeze.",
            "# For a reproducible air-gapped build, re-run build_benchmark.py",
            "# --with-wheels on a machine whose Python env has the benchmark deps installed.",
            "",
            "# Core project dependencies",
            "langchain-core>=1.0.0,<2.0.0",
            "pydantic>=2.0.0,<3.0.0",
            "",
            "# Benchmark-only dependencies",
            "deepagents>=0.2",
            "langchain>=1.0",
            "langchain-openai>=1.0",
            "",
            "# openjiuwen is a proprietary Huawei-internal package; install from",
            "# your internal index. Without it oj-baseline and oj-affinity are",
            "# skipped at runtime.",
            "# openjiuwen>=0.1",
        ]

    header = [
        "# Pinned dependencies for langchain-ascend-affinity offline benchmark",
        f"# Generated by scripts/build_benchmark.py on {_platform_mod.platform()}",
        f"# Python {sys.version.split()[0]}",
        "",
        "# NOTE: C-extension wheels (numpy, pydantic-core, tokenizers, etc.)",
        "# are platform-specific. If you built this package on a different",
        "# OS/architecture, re-download wheels on the target platform:",
        "#   pip download -r requirements.txt -d wheels --only-binary=:all: \\",
        "#       --platform linux_x86_64 --python-version 311",
        "# See OFFLINE_PACKAGING.md §3 for full cross-platform instructions.",
        "",
    ]

    with open(output / "requirements.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n".join(req_lines) + "\n")


def copy_pyproject(output: Path) -> None:
    """Copy the real pyproject.toml from the repo root."""
    src = ROOT / "pyproject.toml"
    if src.exists():
        shutil.copy2(src, output / "pyproject.toml")


def copy_docs(output: Path) -> None:
    """Copy documentation files from repo root."""
    docs = [
        "README.md", "README.zh-CN.md",
        "OFFLINE_PACKAGING.md", "OFFLINE_PACKAGING.zh-CN.md",
        "AGENTS.md", "LICENSE", "CHANGELOG.md", "REQUIREMENTS.md",
    ]
    for fname in docs:
        src = ROOT / fname
        dst = output / fname
        if src.exists():
            shutil.copy2(src, dst)


def copy_scripts(output: Path) -> None:
    """Copy scripts/ directory (quality gate + build tool)."""
    out = output / "scripts"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "scripts", out)
    _clean_pycache(out)


def write_installers(output: Path) -> None:
    """Write install_offline.sh (Bash) and install_offline.ps1 (PowerShell)."""
    sh_path = output / "install_offline.sh"
    ps1_path = output / "install_offline.ps1"
    with open(sh_path, "w", encoding="utf-8") as f:
        f.write(INSTALL_SH)
    # Make the Bash script executable (best-effort on Windows)
    try:
        sh_path.chmod(0o755)
    except OSError:
        pass
    with open(ps1_path, "w", encoding="utf-8") as f:
        f.write(INSTALL_PS1)


def _detect_python_version_tag() -> str:
    """Return the current CPython version tag, e.g. '311' for 3.11."""
    return f"{sys.version_info.major}{sys.version_info.minor}"


def download_wheels(
    output: Path,
    platform: str,
    python_version: str,
    index_url: str,
) -> None:
    """Download wheels for requirements.txt into output/wheels."""
    wheels_dir = output / "wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "pip", "download"]
    cmd += ["-r", str(output / "requirements.txt")]
    cmd += ["-d", str(wheels_dir)]
    cmd += ["--only-binary=:all:"]

    if platform != "current":
        cmd += ["--platform", platform]
        py_ver = python_version or _detect_python_version_tag()
        cmd += ["--python-version", py_ver]
        cmd += ["--implementation", "cp"]
        if py_ver == "313":
            cmd += ["--abi", "cp313"]
        elif py_ver == "312":
            cmd += ["--abi", "cp312"]
        elif py_ver == "311":
            cmd += ["--abi", "cp311"]
        elif py_ver == "310":
            cmd += ["--abi", "cp310"]
        # else: omit abi and let pip figure it out

    if index_url:
        cmd += ["--index-url", index_url]

    print(f"  pip download cmd: {' '.join(cmd[:8])}...")
    # check=False because we want to emit friendly troubleshooting instead
    # of raising CalledProcessError for cross-platform download failures.
    result = subprocess.run(cmd, text=True, check=False)
    if result.returncode != 0:
        print()
        print("WARNING: pip download did not exit cleanly.", file=sys.stderr)
        print(
            "  Cross-platform wheel download can fail on package mirrors that",
            file=sys.stderr,
        )
        print(
            "  don't expose full-platform indices. Workaround: build on a",
            file=sys.stderr,
        )
        print(
            "  machine matching the target OS/architecture.",
            file=sys.stderr,
        )
        print("  See OFFLINE_PACKAGING.md §3-B.", file=sys.stderr)
        return

    count = len(list(wheels_dir.glob("*.whl")))
    size_mb = sum(f.stat().st_size for f in wheels_dir.glob("*.whl")) / (1024 * 1024)
    print(f"  Wheels: {count} files, {size_mb:.1f} MB")


def print_tree(output: Path) -> None:
    """Print a 2-level tree of the package contents."""
    print("Contents:")
    for item in sorted(output.iterdir()):
        name = item.name
        if item.is_dir():
            size_mb = 0.0
            file_count = 0
            for f in item.rglob("*"):
                if f.is_file():
                    size_mb += f.stat().st_size
                    file_count += 1
            size_mb /= 1024 * 1024
            print(f"  {name}/  ({file_count} files, {size_mb:.1f} MB)")
        else:
            size_kb = item.stat().st_size / 1024
            print(f"  {name}  ({size_kb:.1f} KB)")


def main() -> int:
    args = parse_args()
    output = args.output_dir

    # Wipe output unless --no-clean
    if output.exists() and not args.no_clean:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Building offline benchmark package to: {output}")
    print(f"  Platform: {_platform_mod.platform()}  Python: {sys.version.split()[0]}")
    print()

    steps: list[tuple[str, callable]] = []
    steps.append(("Copying langchain_ascend package...", lambda: copy_package_structure(output)))
    steps.append(("Copying benchmark harness...", lambda: copy_benchmark_files(output)))
    steps.append(("Copying scripts/...", lambda: copy_scripts(output)))
    steps.append(("Generating pinned requirements.txt...", lambda: generate_requirements(output)))
    steps.append(("Copying pyproject.toml...", lambda: copy_pyproject(output)))
    steps.append(("Copying documentation...", lambda: copy_docs(output)))

    if args.with_installers:
        steps.append(("Writing install_offline.sh/.ps1...", lambda: write_installers(output)))

    for i, (label, fn) in enumerate(steps, start=1):
        print(f"[{i}/{len(steps)}] {label}")
        fn()

    if args.with_wheels:
        print(f"[{len(steps) + 1}/{len(steps) + 1}] Downloading wheels to wheels/...")
        download_wheels(
            output=output,
            platform=args.wheel_platform,
            python_version=args.wheel_python_version,
            index_url=args.index_url,
        )

    print()
    print_tree(output)

    if args.zip:
        zip_path = output.with_suffix(".zip")
        print(f"\nCreating zip archive: {zip_path}")
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(
            str(zip_path.with_suffix("")), "zip", root_dir=output
        )
        zip_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"Zip created at: {zip_path}  ({zip_mb:.1f} MB)")

    print()
    print("Build complete. Upload and install steps:")
    print("  1. Transfer (scp / usb / media) the package to the air-gapped machine.")
    print("  2. Unzip:   unzip ascend-benchmark-offline.zip")
    print("  3. Install: cd ascend-benchmark-offline && bash install_offline.sh")
    print("  4. Run:     python benchmark/run_benchmark.py --engine-url ... --model ...")
    print()
    print("For cross-platform wheel / troubleshooting: OFFLINE_PACKAGING.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
