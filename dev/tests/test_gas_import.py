"""Check format conversion preserves the model's input and the replay split."""

import csv
import json

import numpy as np
import pytest
from harness_core.config import project_root
from harness_core.gas_data import open_gas_batches
from harness_core.gas_import import convert, detect, draft_mapping, mapped_rows
from harness_core.gas_runner import replay_batches


def test_shuffled_csv_preserves_gas_batch_and_predictions(tmp_path, official_bundle):
    from baseline.gas.baselines.model import Model

    source = project_root() / "data/gas+sensor+array+drift+dataset"
    if not (source / "batch3.dat").is_file():
        pytest.skip("需要项目气体批次数据")
    from sklearn.datasets import load_svmlight_file

    original = open_gas_batches(source, seed=42)
    x, y = load_svmlight_file(str(source / "batch3.dat"), n_features=128, zero_based=False)
    columns = [f"sensor_{index}" for index in range(1, 129)]
    data_path = tmp_path / "source.csv"
    with data_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["gas", *reversed(columns)])
        for label, features in zip(y, x.toarray()):
            writer.writerow([int(label), *reversed(features)])
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"feature_columns": columns, "label_column": "gas"}))
    raw = data_path.read_bytes()
    output = tmp_path / "converted"
    convert(raw, raw.decode(), "csv", mapping, output, 3)
    converted = open_gas_batches(output, seed=42)
    assert replay_batches(converted, {}) == (2,)
    model = Model.load(official_bundle / "checkpoint.npz")
    for split in ("train", "validation", "test"):
        before = original.load(2, split)
        after = converted.load(2, split)
        for expected, actual in zip(before, after):
            np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(model.predict(before[0]), model.predict(after[0]))

    with (output / "batch3.dat").open("a") as stream:
        stream.write("1 1:0\n")
    with pytest.raises(ValueError, match="摘要不一致"):
        open_gas_batches(output, seed=42).manifest


def test_jsonl_missing_label_is_prediction_only(tmp_path):
    columns = [f"values.{index}" for index in range(128)]
    mapping = {"feature_columns": columns, "id_column": "key"}
    raw = ('\n'.join(json.dumps({"key": value, "values": list(range(128))})
                     for value in ("a", "b"))).encode()
    assert detect(raw.decode()) == "jsonl"
    rows = list(mapped_rows(raw.decode(), "jsonl", mapping, labels_required=False))
    assert rows[0] == ("a", list(range(128)), None)
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(mapping))
    out = tmp_path / "should-not-be-created"
    with pytest.raises(ValueError, match="需要真实标签"):
        convert(raw, raw.decode(), "jsonl", path, out, 3)
    assert not out.exists()


def test_draft_transmits_schema_and_field_paths_only():
    def transport(messages):
        assert json.loads(messages[1]["content"]) == {"fields": ["x"], "schema": "unknown"}
        return {"content": '{"mapping":{"feature_columns":[null]},"unresolved":["missing definition"]}'}

    assert draft_mapping(["x"], "unknown", "pytrio", None, transport)["unresolved"]
