# Offline Benchmark Packaging Guide

[English](OFFLINE_PACKAGING.md) | [简体中文](OFFLINE_PACKAGING.zh-CN.md)

This document explains how to build a **self-contained, offline-capable**
package of `langchain-ascend-affinity` + its benchmark harness, suitable for
upload into **air-gapped / no-internet** environments (e.g. secured data
centres running Ascend NPU clusters).

---

## 1. When you need this

Use this flow when:

- The target environment has **no outbound internet access** (no PyPI, no
  internal mirrors, or mirror access is denied by policy).
- You want **byte-identical, reproducible dependency versions** — no surprise
  upgrades when someone runs `pip install` six months later.
- You want **zero interactive prompts** at install time — the benchmark should
  install and run with one script.

When you do **not** need this:

- Target environment has pip access to PyPI or a corporate mirror → use the
  normal flow in `benchmark/README.md`.
- You only plan to verify the library itself (not the 4-agent benchmark).

---

## 2. Quick start (one command)

```bash
# From the repository root:
python scripts/build_benchmark.py \
  --with-wheels \
  --with-installers \
  --zip \
  --output-dir ./ascend-benchmark-offline
```

The script produces:

```
ascend-benchmark-offline/
├── langchain_ascend/          # Core library (edit: one model swap)
├── benchmark/                 # 4-agent harness + tasks + metrics
├── scripts/                   # quality_gate.py, build_benchmark.py
├── wheels/                    # Pre-downloaded .whl files
├── requirements.txt           # Pinned dependency versions
├── pyproject.toml             # Package metadata (editable install)
├── install_offline.sh         # Linux installer (Bash)
├── install_offline.ps1        # Windows installer (PowerShell)
├── README.md                  # Project quick start
├── README.zh-CN.md            # 项目快速开始（中文）
├── AGENTS.md / LICENSE
└── OFFLINE_PACKAGING.md       # This document
```

`--zip` additionally produces `ascend-benchmark-offline.zip` next to the
output directory, ready for upload.

### CLI options

| Option | Default | Meaning |
|---|---|---|
| `--output-dir DIR` | `./ascend-benchmark-offline` | Output directory |
| `--zip` | off | Also compress into `<dir-name>.zip` |
| `--with-wheels` | off | Run `pip download` into `wheels/` after building the package |
| `--with-installers` | off | Write `install_offline.sh` + `.ps1` into the package |
| `--wheel-platform PLATFORM` | `current` | Target platform for wheels. `current` = same machine you run build on. See §3 for cross-platform. |
| `--wheel-python-version VER` | auto-detected | E.g. `311` for CPython 3.11. Pass with `--wheel-platform` for cross-platform builds. |
| `--index-url URL` | system default | Override the PyPI index used for wheel download (e.g. a corporate mirror). |
| `--no-clean` | off | Do **not** wipe the output directory before rebuilding. |

---

## 3. Cross-platform wheel downloads

### Why wheels are platform-specific

Of the ~110 packages in a full build, roughly 80 are **pure Python**
(`py3-none-any.whl`) — these work on **any** OS / CPU architecture.

The remaining ~30 are **C / Rust / Cython extensions** compiled for a
specific OS + Python version + CPU triple. Typical offenders:

- `pydantic-core` (Rust) — pydantic's parser
- `numpy` (C) — numerical arrays
- `orjson` / `ujson` / `msgpack` (C/Rust) — fast JSON
- `tokenizers` / `tiktoken` (Rust) — BPE tokenisers
- `pillow` (C) — image I/O
- `lxml` (C) — XML parser
- `regex`, `rpds-py`, `zstandard`, `httptools` …

### Three scenarios

#### Scenario A — Build machine == target machine (same OS / Python / arch)

```bash
python scripts/build_benchmark.py --with-wheels --with-installers --zip
```

Done. `--wheel-platform current` is the default.

#### Scenario B — Build on Windows, deploy to Linux (very common)

You have two paths:

**Path 1 — Preferred: rebuild on any internet-connected Linux machine.**

Same architecture as the target (x86_64 or aarch64 for Ascend ARM servers):

```bash
# On the Linux machine (same Python version as target, ideally 3.11+):
git clone <this-repo>
python scripts/build_benchmark.py --with-wheels --with-installers --zip
# Transfer ascend-benchmark-offline.zip to the air-gapped machine.
```

**Path 2 — Cross-download on Windows.**

This is slower and sometimes PyPI mirrors (e.g. Huawei internal mirror)
don't expose cross-platform indices for all packages. Use the official
PyPI index explicitly:

```bash
# Target: Linux x86_64 + CPython 3.11
python scripts/build_benchmark.py \
  --with-wheels \
  --with-installers \
  --zip \
  --wheel-platform linux_x86_64 \
  --wheel-python-version 311 \
  --index-url https://pypi.org/simple/
```

Or for Ascend ARM (aarch64):

```bash
python scripts/build_benchmark.py \
  --with-wheels \
  --with-installers \
  --zip \
  --wheel-platform linux_aarch64 \
  --wheel-python-version 311 \
  --index-url https://pypi.org/simple/
```

If cross-download fails for a handful of Rust crates (`pydantic-core`,
`tokenizers`), fall back to Path 1 — spin up a quick cloud Linux VM
(1 CPU, 2 GB RAM, 10 min) and build there.

#### Scenario C — No internet at all, even on the build machine

1. Download the `*.whl` files on **any** machine with internet using:
   ```bash
   pip download -r requirements.txt -d wheels \
     --only-binary=:all: --platform linux_x86_64 --python-version 311
   ```
2. Copy the `wheels/` folder into the package output directory manually.
3. `install_offline.sh` will pick them up automatically via
   `--find-links wheels`.

---

## 4. Installing on the air-gapped target

### Linux (Bash) — typical Ascend NPU host

```bash
unzip ascend-benchmark-offline.zip
cd ascend-benchmark-offline
bash install_offline.sh
```

The installer:
1. Checks Python ≥ 3.11 (required by `deepagents 0.7.6`).
2. Installs dependencies from `wheels/` (no internet, `--no-index --find-links=wheels`).
3. Installs the local `langchain-ascend-affinity` package in editable mode.
4. Verifies that the import chain works and prints a status report.

Then run the benchmark:

```bash
python benchmark/run_benchmark.py \
  --engine-url http://<engine-host>:<port>/v1 \
  --model <model-name> \
  --api-key <api-key>
```

### Windows (PowerShell) — e.g. LM Studio + ascend-sim dev station

```powershell
Expand-Archive ascend-benchmark-offline.zip
cd ascend-benchmark-offline
.\install_offline.ps1
```

### Troubleshooting the install

**Problem: "No matching distribution" for a C-extension package.**

Root cause: the wheels in `wheels/` don't match the target platform.
Typical message:

```
ERROR: Could not find a version that satisfies the requirement numpy==2.3.5
(from versions: none)
```

Fix: re-download wheels on a machine that matches the target platform
(§3, Scenario B-Path 1), or manually drop in the correct `.whl` files.

**Problem: "Metadata-generation-failed" for `pydantic-core` / `tokenizers`.**

Root cause: pip fell back to source build (no binary wheel matched), but
maturin / rustc are not installed.

Fix: same as above — get the correct binary wheels for the target platform.

**Problem: `openjiuwen` missing.**

`openjiuwen` is a Huawei-internal package not published on PyPI. The
installer reports it as missing; the `oj-baseline` and `oj-affinity`
agents are then auto-skipped by the benchmark runner. `lc-baseline` and
`lc-affinity` (the LangChain pair) still run. To run all four agents,
install `openjiuwen` from your internal package index:

```bash
pip install openjiuwen --index-url https://your-internal-index/simple/
```

---

## 5. Package contents reference

```
ascend-benchmark-offline/
├── langchain_ascend/
│   ├── __init__.py                       # Exports: AscendAffinityChatModel
│   ├── prefix_tracker.py                 # Prefix-diff scheduler
│   └── llms/
│       ├── chat_ascend.py                # Main model (salt + release)
│       ├── affinity_pipeline.py          # Pipeline stages
│       ├── agent_hint.py                 # evict/offload/prefetch (opt-in)
│       ├── serialization.py              # Request/response serialisers
│       └── transport.py                  # HTTP transport + retry
├── benchmark/
│   ├── run_benchmark.py                  # Entry point (setup → probe → run)
│   ├── agents.py                         # lc-baseline + lc-affinity builders
│   ├── oj_adapter.py                     # oj-baseline + oj-affinity builders
│   ├── tasks.py                          # Financial task set + tools + longrun
│   ├── metrics.py                        # TTFT/E2E/KV-hit/TPOT/decode aggregation
│   ├── probe.py                          # Engine probe (release endpoint / salt tolerance / streaming usage)
│   ├── reporting.py                      # JSON + Markdown lab-sheet report
│   ├── requirements.txt                  # Benchmark-only dep subset
│   ├── PRINCIPLES.md                     # Benchmark methodology
│   ├── README.md / README.zh-CN.md       # Benchmark quick start
│   └── reports/                          # Report output directory (empty initially)
├── scripts/
│   ├── build_benchmark.py                # This build tool
│   └── quality_gate.py                   # Pylint + pytest gate
├── wheels/                               # 110 pre-downloaded wheels (~62 MB)
├── requirements.txt                      # Pinned versions
├── pyproject.toml                        # For editable install
├── install_offline.sh                    # Linux installer
├── install_offline.ps1                   # Windows installer
├── README.md / README.zh-CN.md           # Project quick start
├── AGENTS.md                             # Project maintenance rules
└── LICENSE
```

---

## 6. Upgrading / rebuilding

To update the package after a code change:

```bash
# 1. Update the code in the repo
# 2. Reinstall benchmark deps in case versions drifted
pip install -r benchmark/requirements.txt

# 3. Regenerate the package (and wheels)
python scripts/build_benchmark.py --with-wheels --with-installers --zip --no-clean

# 4. Upload the new zip
```

**Note**: `--no-clean` preserves the existing `wheels/` directory so you
don't re-download 62 MB of already-valid wheels every time. Remove it if
you want a fully fresh build (including deleting stale wheels).

---

## 7. FAQ

**Q: Can I strip the package to run only `lc-*` agents, dropping `openjiuwen`?**

Yes. Remove:
- `benchmark/oj_adapter.py` (optional, the runner skips missing deps anyway)
- `benchmark/requirements.txt` openjiuwen line (already commented)
- Pass `--agents lc` to `run_benchmark.py` at runtime.
Expected package size reduction: zero, `openjiuwen` wheel was never bundled.

**Q: The package is 62 MB. Can I make it smaller?**

Yes. Options (biggest wins first):

1. Drop `transformers`, `tokenizers`, `huggingface_hub` — they are
   **not** required by the affinity library or the benchmark runner.
   They are pulled in transitively by `langchain-core`'s optional deps but
   never used. Remove them from `requirements.txt` and from `wheels/`.
   Saves ~12 MB.
2. Drop `pillow`, `lxml`, `openpyxl`, `reportlab`, `python-docx`,
   `python-pptx`, `pdfplumber` etc. — data-processing packages that
   neither the library nor the benchmark touches. Saves another ~8 MB.
3. Run `pip install --dry-run --report` to find unused transitive deps.
4. Keep only `--agents lc` and drop `langchain-anthropic`,
   `langchain-google-genai` from the requirements.

**Q: What is the minimum viable package?**

Core library only (no benchmark):

```
langchain_ascend/ + requirements.txt
  -> langchain-core, pydantic, typing_extensions, annotated-types,
     jiter, anyio, sniffio, idna, certifi, h11, h2, httpx, httpcore
  ≈ 8 wheels, < 5 MB
```

**Q: Can the installer auto-run quality_gate.py after install?**

Not by default. Quality gate runs **unit tests** that assert the library
works correctly on the current platform — in an air-gapped environment
with only wheels for a *different* platform this would mislead. Run it
manually **after** verifying wheels match the target:

```bash
python scripts/quality_gate.py
```
