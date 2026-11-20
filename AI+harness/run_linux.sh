#!/usr/bin/env bash
set -euo pipefail
HARNESS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_CONDA="${CONDA_EXE:-$HOME/anaconda3/bin/conda}"
if [[ ! -x "$HARNESS_CONDA" ]]; then
    echo "未找到 Anaconda，请通过 CONDA_EXE 指定 conda 可执行文件。" >&2
    exit 2
fi
HARNESS_BASE="$("$HARNESS_CONDA" info --base)"
HARNESS_PYTHON="$HARNESS_BASE/envs/AI+harness/bin/python"
if [[ ! -x "$HARNESS_PYTHON" ]]; then
    echo "未找到 Anaconda AI+harness 环境，请先运行 bash scripts/setup_linux.sh" >&2
    exit 2
fi
cd -- "$HARNESS_ROOT"
export PYTHONIOENCODING=utf-8
if [[ $# -eq 0 ]]; then
    set -- run
fi
exec "$HARNESS_PYTHON" "$HARNESS_ROOT/run.py" "$@"
