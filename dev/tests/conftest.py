"""自动验收永远不读取用户保存的真实服务密钥。"""

from pathlib import Path

import numpy as np
import pytest

from harness_core import credentials
from wrappers.gas_dataset import GasDriftData


@pytest.fixture(autouse=True)
def isolate_local_secrets(monkeypatch):
    monkeypatch.setattr(credentials, "_local_values", lambda: {})


@pytest.fixture(scope="session")
def gas_data():
    def create(root):
        root.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        for batch, name in enumerate(GasDriftData.batch_names):
            size = 250 if batch == 0 else 100
            x = rng.standard_normal((size, 2))
            direction = -1 if batch >= 4 else 1
            y = 1 + (direction * x[:, 0] + .4 * x[:, 1] > 0).astype(int)
            rows = [f"{label} 1:{a:.8f} 2:{b:.8f}\n" for label, (a, b) in zip(y, x)]
            (root / name).write_text("".join(rows), encoding="utf-8")
        return GasDriftData(root, seed=42)

    return create


@pytest.fixture(scope="session")
def dataset(tmp_path_factory, gas_data):
    return gas_data(tmp_path_factory.mktemp("gas_batches"))


@pytest.fixture(scope="session")
def official_bundle():
    bundle = Path(__file__).resolve().parents[1] / "baseline/gas"
    if not (bundle / "checkpoint.npz").is_file():
        pytest.skip("需要官方气体 Baseline 包")
    return bundle
