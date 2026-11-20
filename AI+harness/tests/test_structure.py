"""验证真实数据入口和独立的模型、数据适配层。"""

import subprocess
import sys
from pathlib import Path

from wrappers import GasDriftData


def test_unified_entrypoint_and_original_module_both_show_help():
    root = Path(__file__).resolve().parent.parent
    for args in ([str(root / "run.py"), "--help"], ["-m", "harness_demo", "--help"]):
        result = subprocess.run([sys.executable, *args], cwd=root, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0
        assert "{run,status}" in result.stdout


def test_wrappers_live_outside_core_package():
    root = Path(__file__).resolve().parent.parent
    for name in ("adapters.py", "data.py", "wrappers.py"):
        assert not (root / "harness_demo" / name).exists()
    assert GasDriftData.__module__ == "wrappers.gas_drift"


def test_container_includes_external_wrappers():
    root = Path(__file__).resolve().parent.parent
    assert "COPY wrappers ./wrappers" in (root / "Dockerfile").read_text()
    assert "!wrappers/**" in (root / ".dockerignore").read_text()


def test_only_mainline_modes_are_exposed():
    from harness_demo.__main__ import parser

    options = parser().parse_args(["run", "--strategy", "llm"])
    assert options.strategy == "llm"
    root = Path(__file__).resolve().parent.parent
    assert not (root / "harness_demo/gas_feature_drift.py").exists()
    assert not (root / "wrappers/models.py").exists()
    assert not (root / "wrappers/datasets.py").exists()
