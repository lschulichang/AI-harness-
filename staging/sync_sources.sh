#!/usr/bin/env bash
# Sync the current trainers from dev and keep only their competition entry points.
set -euo pipefail
target=$(cd -- "$(dirname -- "$0")" && pwd)
project=$(cd -- "$target/../dev" && pwd)
python3 - "$project" "$target" <<'PY'
import ast
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

project, target = map(Path, sys.argv[1:])
files = {}


def remove_definitions(source, *names, owner=None):
    body = ast.parse(source).body
    if owner is not None:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == owner)
    definitions = {node.name: node for node in body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    lines = source.splitlines(keepends=True)
    spans = []
    for name in names:
        node = definitions[name]
        start = min([node.lineno, *(item.lineno for item in node.decorator_list)]) - 1
        end = node.end_lineno
        while end < len(lines) and not lines[end].strip():
            end += 1
        spans.append((start, end))
    for start, end in sorted(spans, reverse=True):
        del lines[start:end]
    return "".join(lines)


def replace(source, old, new=""):
    if source.count(old) != 1:
        raise ValueError("Development source changed. Review the staging rule for: " + old)
    return source.replace(old, new, 1)


def add(relative, destination=None, *, drop=(), methods=None, edits=()):
    src = project / relative
    destination = destination or relative
    if not (drop or methods or edits):
        files[destination] = src.read_bytes()
        return
    source = src.read_text(encoding="utf-8")
    if drop:
        source = remove_definitions(source, *drop)
    for owner, names in (methods or {}).items():
        source = remove_definitions(source, *names, owner=owner)
    for old, new in edits:
        source = replace(source, old, new)
    files[destination] = (source.rstrip() + "\n").encode("utf-8")


for module in ("__init__", "storage", "portability", "feature_drift",
               "gas_drift", "gas_metrics", "gas_training", "history_evaluation"):
    add(f"harness_core/{module}.py")

add("harness_core/runner.py", drop=("run_workflow",), edits=(
    ("校验运行配置，分发 A 气体分类及 B、C 寿命回放。", "比赛训练闭环共用的历史窗口。"),
    ("from pathlib import Path\n\nfrom .config import RunConfig\nfrom .credentials import get_secret\nfrom .storage import read_json\n\n\n", ""),
))
add("harness_core/config.py", drop=("project_root",), methods={"RunConfig": ("__post_init__", "from_saved_options")}, edits=(
    ("from pathlib import Path\n\nPROVIDER_DEFAULTS = {\n"
     "    \"pytrio\": (\"Qwen/Qwen3.8-27B\", \"https://pytrio.com/api/openai/v1\", \"PYTRIO_API_KEY\"),\n"
     "}\n\n\n", "\n\n"),
    ('    provider: str = "pytrio"\n', '    provider: str = "host_gateway_rpc"\n'),
    ("    base_url: str | None = None\n    api_key_env: str | None = None\n", ""),
    ('    swanlab_mode: str = "online"\n', '    swanlab_mode: str = "offline"\n'),
    ('    swanlab_workspace: str | None = "ls_chulichang"\n', "    swanlab_workspace: str | None = None\n"),
    ('    competition: bool = False\n    data_dir: str = ""\n', ""),
    ('    baseline_dir: str = ""\n', ""),
    ('        if self.provider not in PROVIDER_DEFAULTS:\n            raise ValueError("provider 只支持 PyTRiO")\n', ""),
    ('        if self.swanlab_mode not in {"offline", "online", "disabled"}:\n',
     '        if self.swanlab_mode not in {"offline", "disabled"}:\n'),
    ('        if self.competition:\n            raise ValueError("官方 SDK 尚未接入：练习入口不能作为正式比赛入口。请先实现官方数据适配器。")\n'
     '        if not self.base_url or not self.api_key_env:\n'
     '            raise ValueError("PyTRiO 需要配置 API 地址和密钥环境变量")\n', ""),
))
add("harness_core/llm_client.py", drop=("OpenAITransport",), edits=(
    ("from .credentials import get_secret\n\n\n", ""),
    ("                    if self.transport is None:\n"
     "                        self.transport = OpenAITransport(self.config)\n"
     "                    if isinstance(self.transport, OpenAITransport):\n"
     '                        response = self.transport(messages, max_tokens=reservation.get("reserved_output_tokens"))\n'
     "                    else:\n"
     "                        response = self.transport(messages)\n",
     "                    response = self.transport(messages)\n"),
))
for module, owner in (("gas_policy", "GasTrainer"), ("rul_policy", "RulTrainer")):
    add(f"harness_core/{module}.py", methods={owner: ("__init__",)}, edits=(
        ("from .llm_client import OpenAITransport, Planner", "from .llm_client import Planner"),
    ))
add("harness_core/tracking.py", methods={"Tracker": ("log_final",)}, edits=(
    ("from .credentials import get_secret\n", ""),
    ('                    on_online_error="offline", settings=swanlab.Settings(api_key=get_secret("SWANLAB_API_KEY"), probe={\n',
     "                    settings=swanlab.Settings(probe={\n"),
))

add("harness_core/gas_data.py", drop=("open_gas_batches", "evaluate"), edits=(
    ("from wrappers.gas_dataset import GasDriftData, load_gas_model",
     "from wrappers.gas_dataset import load_gas_model"),
    ('GAS_DATA_DIR = "data/gas+sensor+array+drift+dataset"\n\n\n', ""),
))
add("harness_core/gas_workflow.py", drop=("run_trainer_batches",), edits=(
    ("from statistics import mean\n", ""),
    ("from .gas_data import evaluate, evaluate_samples, gas_samples", "from .gas_data import evaluate_samples, gas_samples"),
    ("from .gas_metrics import evaluate as trainer_evaluate\n", ""),
    ("from .gas_policy import (\n    TRAINING_ACTIONS,\n    WINDOW_DRIFT_BATCHES,\n"
     "    WINDOW_TRAINER_CONFIG,\n    GasTrainer,\n)\n",
     "from .gas_policy import TRAINING_ACTIONS, WINDOW_TRAINER_CONFIG\n"),
    ("from .history_evaluation import gas_references, gas_scores, history_batches",
     "from .history_evaluation import gas_references, gas_scores"),
    ("from .resource_monitor import BatchMeter\n", ""),
    ('name not in {config.api_key_env, "PYTHONPATH"}', 'name != "PYTHONPATH"'),
))
add("harness_core/rul_workflow.py", drop=("evaluate_public_test", "write_summary", "run_rul"), edits=(
    ("from datetime import datetime\nfrom pathlib import Path\n", ""),
    ("import shutil\n", ""),
    ("from uuid import uuid4\n", ""),
    ("from threadpoolctl import threadpool_limits\n", ""),
    ("from wrappers.bearing_baseline import PACKAGE, BearingBaseline, digest\n"
     "from wrappers.bearing_pairs import REPLAY_DIR, PairedBearingBaseline\n"
     "from wrappers.cmapss_baseline import PACKAGE as C_PACKAGE\n"
     "from wrappers.cmapss_baseline import RAW_DATA, CmapssBaseline, test_scores\n",
     "from wrappers.rul_baseline import digest\n"),
    ("from .rul_policy import LIMITS, TRAINING_ACTIONS, RulTrainer, training_defaults",
     "from .rul_policy import LIMITS, TRAINING_ACTIONS, training_defaults"),
    ("from .history_evaluation import HISTORY_WINDOW, history_batches, rul_references, rul_scores",
     "from .history_evaluation import history_batches, rul_references, rul_scores"),
    ("from .config import project_root\n", ""),
    ("from .llm_client import request_usage_known\n", ""),
    ("from .portability import resolve_model_path, saved_data_path", "from .portability import resolve_model_path"),
    ("from .storage import Ledger, RunLock, atomic_json, read_json", "from .storage import read_json"),
    ("from .tracking import Tracker\n", ""),
    ('name not in {config.api_key_env, "PYTHONPATH"}', 'name != "PYTHONPATH"'),
    ("run_bearing_rul = run_rul  # 保留原 B 调用入口。\n", ""),
))

files["wrappers/__init__.py"] = b'"""Competition model adapters."""\n'
add("wrappers/gas_dataset.py", drop=("GasDriftData",), edits=(
    ('"""读取 UCI 气体传感器漂移数据的时间批次。"""', '"""读取平台提供的气体分类检查点。"""'),
    ('import hashlib\nimport json\nfrom functools import cached_property\nfrom pathlib import Path\n\n'
     'import numpy as np\nfrom sklearn.datasets import load_svmlight_file\n'
     'from sklearn.model_selection import train_test_split\n\nIMPORT_MANIFEST = "gas-import.json"\n\n\n', ""),
    ("def load_gas_model(checkpoint=None):", "def load_gas_model(checkpoint):"),
    ('    if checkpoint is None:\n        checkpoint = Path(__file__).resolve().parents[1] / "baseline/gas/checkpoint.npz"\n', ""),
))
add("wrappers/rul_baseline.py", drop=("read_rows",), methods={"RulBaseline": ("__init__",)}, edits=(
    ("import importlib\nimport importlib.util\n", ""), ("import sys\n", ""),
))
add("wrappers/bearing_baseline.py", methods={"BearingBaseline": ("identity",)}, edits=(
    ("from .rul_baseline import digest as digest\n", ""),
    ("from .rul_baseline import read_rows as read_rows\n", ""),
))
add("wrappers/cmapss_baseline.py", drop=("test_scores",), methods={"CmapssBaseline": ("identity", "test_manifest", "test_samples")}, edits=(
    ("from pathlib import Path\n\nimport numpy as np\n\nfrom .rul_baseline import RulBaseline, digest",
     "from .rul_baseline import RulBaseline"),
    ('RAW_DATA = "data/CMAPSSData"\n', ""),
))

for bundle in ("gas", "cpu-b-rul-20261110-v1", "cpu-c-guide-20261110-v1"):
    for name in ("__init__", "model", "data", "train", "predict"):
        add(f"baseline/{bundle}/baselines/{name}.py")
add("baseline/cpu-b-rul-20261110-v1/baselines/configs/B.json")
add("baseline/cpu-c-guide-20261110-v1/baselines/configs/C.json")
sdk = "contest-harness-sdk-1.0.2"
add(f"{sdk}/template/harness.py", "harness.py")
for name in ("pyproject.toml", "README.md"):
    add(f"{sdk}/{name}", f"sdk/{name}")
for src in sorted((project / sdk / "contest_harness").glob("*.py")):
    add(f"{sdk}/contest_harness/{src.name}", f"sdk/contest_harness/{src.name}")

for name, content in files.items():
    path = target / name
    if target not in path.resolve().parents:
        raise ValueError("Staging path escapes its directory: " + str(path))
    if path.suffix == ".py":
        compile(content, name, "exec")

for name, content in files.items():
    path = target / name
    if path.is_file() and path.read_bytes() == content:
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".sync-", delete=False) as pending:
        pending.write(content)
    os.chmod(pending.name, 0o644)
    os.replace(pending.name, path)

print(f"Synced {len(files)} competition build files from {project}")
PY
