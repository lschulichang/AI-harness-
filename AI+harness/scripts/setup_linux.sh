#!/usr/bin/env bash
set -euo pipefail
HARNESS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS_CONDA="${CONDA_EXE:-$HOME/anaconda3/bin/conda}"
if [[ ! -x "$HARNESS_CONDA" ]]; then
    echo "未找到 Anaconda，请通过 CONDA_EXE 指定 conda 可执行文件。" >&2
    exit 2
fi
HARNESS_BASE="$("$HARNESS_CONDA" info --base)"
HARNESS_ENV="$HARNESS_BASE/envs/AI+harness"
if [[ ! -x "$HARNESS_ENV/bin/python" ]]; then
    "$HARNESS_CONDA" create --prefix "$HARNESS_ENV" python=3.11 -y
fi
cd -- "$HARNESS_ROOT"
"$HARNESS_ENV/bin/python" -m pip install -r requirements-harness-dev.txt
"$HARNESS_ENV/bin/python" -m pip check
echo "环境已就绪：$HARNESS_ENV；可运行 bash run_linux.sh --help"
