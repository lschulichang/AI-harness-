"""按整根轴承配对 B 批次，复用公开 _4 特征并从原始 _5 CSV 计算配对特征。"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


# Physical condition, training bearing, validation bearing in release order.
PAIRS = (
    ("35Hz12kN", "Bearing1_4", "Bearing1_5"),
    ("37.5Hz11kN", "Bearing2_5", "Bearing2_4"),
    ("40Hz10kN", "Bearing3_4", "Bearing3_5"),
)
PROTOCOL = "bearing_paired_condition_replay"


NAMES = (
    "rms", "peak", "peak_to_peak", "mean_abs", "kurtosis", "skewness",
    "crest_factor", "shape_factor", "impulse_factor", "margin_factor",
    "spectral_centroid", "spectral_rms", "spectral_kurtosis", "dominant_freq",
    "band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k",
)

def features(signal):
    x = np.asarray(signal, dtype=np.float64)
    absolute = np.abs(x)
    mean, std = x.mean(), x.std()
    rms = np.sqrt(np.mean(x * x))
    peak, mean_abs = absolute.max(), absolute.mean()
    if min(std, rms, mean_abs, np.mean(np.sqrt(absolute))) <= 0:
        raise ValueError("zero vibration scale in a raw snapshot")
    freq = np.fft.rfftfreq(x.size, d=1 / 25600)
    power = (np.abs(np.fft.rfft(x)) / x.size) ** 2
    power[0] = 0
    if power.sum() <= 0 or power.std() <= 0:
        raise ValueError("zero spectral scale in a raw snapshot")
    values = (
        rms, peak, np.ptp(x), mean_abs,
        np.mean((x - mean) ** 4) / std ** 4,
        np.mean((x - mean) ** 3) / std ** 3,
        peak / rms, rms / mean_abs, peak / mean_abs,
        peak / np.mean(np.sqrt(absolute)) ** 2,
        np.sum(freq * power) / np.sum(power),
        np.sqrt(np.mean(power)),
        np.mean((power - power.mean()) ** 4) / power.std() ** 4,
        freq[np.argmax(power)],
        power[(freq >= 0) & (freq < 2000)].sum(),
        power[(freq >= 2000) & (freq < 6000)].sum(),
        power[(freq >= 6000) & (freq <= min(12800, freq[-1]))].sum(),
    )
    result = dict(zip(NAMES, map(float, values)))
    if len(result) != 17 or not all(np.isfinite(v) for v in result.values()):
        raise ValueError("invalid vibration features")
    return result

def convert(csv_path, bearing, index, total, raw_root):
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    if raw.shape != (32768, 2) or not np.isfinite(raw).all():
        raise ValueError(f"invalid snapshot: {csv_path}")
    return {
        "id": f"xjtu_{bearing}_s{index:06d}",
        "inputs": {"features": {
            "horizontal": features(raw[:, 0]), "vertical": features(raw[:, 1])}},
        "answer": total - index,
        "meta": {"bearing": bearing, "snapshot": index,
                 "source": "XJTU-SY_Bearing_Datasets/" + csv_path.relative_to(raw_root).as_posix()},
    }


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def snapshots(raw_root, condition, bearing):
    directory = raw_root / condition / bearing
    paths = sorted(directory.glob("*.csv"), key=lambda path: int(path.stem))
    if not paths or [int(path.stem) for path in paths] != list(range(1, len(paths) + 1)):
        raise ValueError(f"轴承 CSV 编号为空、不连续或有重复：{directory}")
    return paths


def prepare(baseline, raw_root, output):
    report = json.loads((baseline / "training.json").read_text(encoding="utf-8"))
    info = json.loads((baseline / "baseline-info.json").read_text(encoding="utf-8"))
    if (report.get("profile"), report.get("task"), report.get("target_unit")) != ("B", "regression", "minutes"):
        raise ValueError("需要 B 轴承 RUL 基线")
    paths = {"train_sha256": baseline / "public/train.jsonl",
             "validation_sha256": baseline / "public/validation.jsonl",
             "checkpoint_sha256": baseline / "checkpoint.npz"}
    hashes = {key: digest(path) for key, path in paths.items()}
    if any(report[key] != value for key, value in hashes.items()) or info["checkpoint_sha256"] != hashes["checkpoint_sha256"]:
        raise ValueError("公开基线数据或检查点摘要不匹配")
    seen = {row["meta"]["bearing"] for row in rows(paths["train_sha256"])}
    public = rows(paths["validation_sha256"])
    train, validation, batches = [], [], []
    for index, (condition, training_bearing, validation_bearing) in enumerate(PAIRS, 1):
        if training_bearing in seen or validation_bearing in seen:
            raise ValueError("回放轴承在初始模型训练集内")
        published, computed = f"Bearing{index}_4", f"Bearing{index}_5"
        source = sorted((row for row in public if row["meta"]["bearing"] == published),
                        key=lambda row: row["meta"]["snapshot"])
        source_paths = snapshots(raw_root, condition, published)
        n = len(source_paths)
        if (len(source) != n or [row["meta"]["snapshot"] for row in source] != list(range(1, n + 1))
                or [row["answer"] for row in source] != list(range(n - 1, -1, -1))):
            raise ValueError(f"公开特征与原始时间轴不一致：{published}")
        for sample in {0, n - 1}:
            converted = convert(source_paths[sample], published, sample + 1, n, raw_root)
            for channel in ("horizontal", "vertical"):
                actual = converted["inputs"]["features"][channel]
                expected = source[sample]["inputs"]["features"][channel]
                if actual.keys() != expected.keys() or not np.allclose(
                        [actual[name] for name in expected], list(expected.values()), rtol=1e-10, atol=1e-12):
                    raise ValueError(f"原始振动特征与公开包定义不一致：{published}")
        raw_paths = snapshots(raw_root, condition, computed)
        bearing_rows = {published: source,
                        computed: [convert(path, computed, i, len(raw_paths), raw_root)
                                   for i, path in enumerate(raw_paths, 1)]}
        training = bearing_rows[training_bearing]
        evaluation = bearing_rows[validation_bearing]
        train.extend(training)
        validation.extend(evaluation)
        batches.append({"condition": f"condition{index}", "train_bearing": training_bearing,
                        "train_rows": len(training), "validation_bearing": validation_bearing,
                        "validation_rows": len(evaluation)})
        print(f"{condition}: {training_bearing} {len(training)} / {validation_bearing} {len(evaluation)}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".bearing-pairs-", dir=output.parent) as temporary:
        staging = Path(temporary) / "data"
        staging.mkdir()
        for name, samples in (("train", train), ("validation", validation)):
            with (staging / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                for row in samples:
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        manifest = {"protocol": PROTOCOL, "checkpoint_sha256": hashes["checkpoint_sha256"],
                    "train_sha256": digest(staging / "train.jsonl"),
                    "validation_sha256": digest(staging / "validation.jsonl"), "batches": batches}
        (staging / "replay.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"输出目录已存在：{args.output}")
    prepare(args.baseline, args.raw_root, args.output)
