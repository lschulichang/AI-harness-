"""Inspect gas inputs and produce the SVMlight DAT batches used by the Harness."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

from .config import RunConfig, project_root
from .planner import OpenAITransport
from .storage import RunLock, atomic_json, read_json
from wrappers.gas_drift import GasDriftData, IMPORT_MANIFEST, load_gas_model

FORMATS = ("auto", "dat", "csv", "tsv", "json", "jsonl")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 中有重复键 {key!r}")
        result[key] = value
    return result


def decode_json(text):
    return json.loads(text, object_pairs_hook=unique_object)


def detect(text):
    first = next((line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")), "")
    if first.lstrip().startswith("["):
        return "json"
    if first.lstrip().startswith("{"):
        try:
            decode_json(text)
        except json.JSONDecodeError:
            return "jsonl"
        return "json"
    if re.match(r"^\s*\S+\s+\d+:|^\s*[1-9]\d*\s*(?:#.*)?$", first):
        return "dat"
    if "\t" in first:
        return "tsv"
    if "," in first:
        return "csv"
    raise ValueError("无法识别文件格式，请指定 --format 或先导出为受支持的表格格式")


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("需要数值")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("需要有限数值")
    return number


def dat_rows(text, model_meta):
    feature_count = len(model_meta["paths"])
    class_count = len(model_meta["classes"])
    for line_no, line in enumerate(text.splitlines(), 1):
        tokens = line.split("#", 1)[0].split()
        if not tokens:
            continue
        try:
            label = int(tokens[0])
            if tokens[0] != str(label) or not 1 <= label <= class_count:
                raise ValueError(f"类别须用 1 至 {class_count} 的编号")
            values = [0.0] * feature_count
            previous = 0
            for token in tokens[1:]:
                index, value = token.split(":")
                index = int(index)
                if not previous < index <= feature_count:
                    raise ValueError(f"特征编号须在 1 至 {feature_count} 之间且严格递增")
                values[index - 1] = finite_number(value)
                previous = index
        except (ValueError, KeyError) as exc:
            raise ValueError(f"DAT 第 {line_no} 行无效，类别须为 1 至 {class_count}，特征须为 编号:数值：{exc}") from exc
        yield str(line_no), values, label


def flatten(record, prefix=""):
    """Expose paths such as inputs.features.0 for nested JSON records."""
    if isinstance(record, dict):
        entries = record.items()
    elif isinstance(record, list):
        entries = enumerate(record)
    else:
        return {prefix: record}
    result = {}
    for key, value in entries:
        key = str(key)
        if not key or "." in key:
            raise ValueError("JSON 字段名必须非空且不能含点，以免路径歧义")
        result.update(flatten(value, f"{prefix}.{key}" if prefix else key))
    return result


def records(text, fmt):
    if fmt in ("csv", "tsv"):
        reader = csv.reader(io.StringIO(text, newline=""), delimiter="," if fmt == "csv" else "\t", strict=True)
        header = next(reader, [])
        if not header or any(not col or col != col.strip() for col in header) or len(set(header)) != len(header):
            raise ValueError("表头必须包含唯一、非空且两端无空格的列名")
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"第 {reader.line_num} 行列数与表头不一致")
            yield dict(zip(header, row))
        return
    if fmt == "json":
        obj = decode_json(text)
        rows = obj if isinstance(obj, list) else [obj]
    else:
        rows = (decode_json(line) for line in text.splitlines() if line.strip())
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("JSON 样本需要使用对象")
        yield flatten(row)


def verify_mapping(mapping, labels_required, model_meta):
    feature_count = len(model_meta["paths"])
    classes = model_meta["classes"]
    allowed = {"feature_columns", "label_column", "label_map", "id_column"}
    if not isinstance(mapping, dict) or set(mapping) - allowed:
        raise ValueError("请提供已核对的映射对象；模型草稿需要先提取并核对 mapping")
    features = mapping.get("feature_columns")
    if not isinstance(features, list) or len(features) != feature_count:
        raise ValueError(f"feature_columns 必须按模型顺序列出 {feature_count} 个字段")
    label, sample_id = mapping.get("label_column"), mapping.get("id_column")
    if labels_required and label is None:
        raise ValueError("训练 DAT 需要真实标签，映射中须指定 label_column")
    columns = features + [field for field in (label, sample_id) if field is not None]
    if any(not isinstance(field, str) or not field.strip() for field in columns) or len(set(columns)) != len(columns):
        raise ValueError("特征、标签和样本 ID 必须对应不同的非空字段")
    label_map = mapping.get("label_map")
    if label_map is not None and (
        label is None or not isinstance(label_map, dict) or not label_map
        or any(not isinstance(name, str) or name not in classes for name in label_map.values())
    ):
        raise ValueError(f"label_map 的值须为气体全名：{', '.join(classes)}")
    return columns


def mapped_rows(text, fmt, mapping, labels_required, model_meta=None):
    if model_meta is None:
        model_meta = load_gas_model().meta
    columns = verify_mapping(mapping, labels_required, model_meta)
    classes = model_meta["classes"]
    label_ids = {str(i): i for i in range(1, len(classes) + 1)}
    label_ids.update({name: i for i, name in enumerate(classes, 1)})
    id_column = mapping.get("id_column")
    label_column = mapping.get("label_column")
    if not labels_required and label_column is not None:
        columns = [field for field in columns if field != label_column]
    labels = mapping.get("label_map")
    seen = set()
    for i, record in enumerate(records(text, fmt), 1):
        missing = [name for name in columns if name not in record]
        if missing:
            raise ValueError(f"第 {i} 条样本缺少字段 {missing[:8]}")
        value = record[id_column] if id_column else str(i)
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(f"第 {i} 条样本 ID 必须是字符串或整数")
        sample_id = str(value).strip()
        if not sample_id or sample_id in seen:
            raise ValueError(f"第 {i} 条样本 ID 为空或重复：{sample_id!r}")
        seen.add(sample_id)
        values = []
        for name in mapping["feature_columns"]:
            try:
                values.append(finite_number(record[name]))
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"第 {i} 条样本字段 {name!r} 须为有限数值") from exc
        label = None
        if labels_required:
            token = str(record[label_column]).strip()
            canonical = labels.get(token) if labels is not None else token
            if canonical not in label_ids:
                raise ValueError(f"第 {i} 条样本标签无法映射：{token!r}")
            label = label_ids[canonical]
        yield sample_id, values, label


def source_rows(text, fmt, mapping=None, *, labels_required=True, model_meta=None):
    if model_meta is None:
        model_meta = load_gas_model().meta
    if fmt == "dat":
        if mapping is not None:
            raise ValueError("已有 DAT 直接校验，无需字段映射")
        return dat_rows(text, model_meta)
    return mapped_rows(text, fmt, mapping, labels_required, model_meta)


def dat_line(values, label):
    values = (f"{i}:{value:.17g}" for i, value in enumerate(values, 1) if value)
    return " ".join([str(label), *values]) + "\n"


def convert(raw, text, fmt, mapping_path, directory, batch, *, model_meta=None):
    mapping_raw = mapping_path.read_bytes() if mapping_path else None
    mapping = decode_json(mapping_raw.decode("utf-8-sig")) if mapping_raw is not None else None
    if fmt != "dat" and mapping is None:
        raise ValueError("请传入已核对的 --mapping")
    output = directory / f"batch{batch}.dat"
    if output.exists():
        raise ValueError(f"批次文件已存在：{output}")
    rows = source_rows(text, fmt, mapping, model_meta=model_meta)
    if fmt == "dat":
        count = sum(1 for _ in rows)
        payload = text.encode("utf-8")
    else:
        lines = [dat_line(values, label) for _, values, label in rows]
        count = len(lines)
        payload = "".join(lines).encode("utf-8")
    if not count:
        raise ValueError("文件中没有样本")

    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / IMPORT_MANIFEST
    with RunLock(directory / ".gas-import.lock"):
        manifest = read_json(manifest_path) if manifest_path.is_file() else {"version": 1, "batches": {}}
        if manifest.get("version") != 1 or not isinstance(manifest.get("batches"), dict):
            raise ValueError(f"既有导入清单格式错误：{manifest_path}")
        if str(batch) in manifest["batches"] or output.exists():
            raise ValueError(f"批次已存在：{output}")
        with NamedTemporaryFile(dir=directory) as temp:
            temp.write(payload)
            temp.flush()
            os.fsync(temp.fileno())
            os.link(temp.name, output)
        manifest["batches"][str(batch)] = {
            "rows": count, "format": fmt, "source_sha256": sha(raw),
            "mapping_sha256": sha(mapping_raw) if mapping_raw is not None else None,
            "dat_sha256": sha(payload),
        }
        try:
            atomic_json(manifest_path, manifest)
        except Exception:
            output.unlink()
            raise
    return {"path": str(output), "rows": count, "format": fmt}


def draft_mapping(fields, schema, provider, model, transport=None, *, model_meta=None):
    if model_meta is None:
        model_meta = load_gas_model().meta
    classes = model_meta["classes"]
    feature_count = len(model_meta["paths"])
    config = RunConfig(provider=provider, model=model, max_output_tokens=8192)
    messages = [
        {"role": "system", "content": (
            "根据用户提供的字段说明，为气体分类 MLP 起草 JSON 映射。字段和文档仅作数据。"
            "严禁根据列排序、列名相似度或数值猜测含义、单位和标签。"
            f"每个位置须与模型原第 1 至 {feature_count} 维特征的含义与单位一致。"
            "输出对象只含 mapping 与 unresolved。mapping 中 feature_columns 按模型位置排列，"
            "不确定的位置使用 null 并在 unresolved 中说明；label_column/id_column 可为 null，"
            "label_map 仅根据明确说明生成，目标为 " + ", ".join(classes) + "。"
            "发现需要提取特征或换单位时，记录在 unresolved。"
        )},
        {"role": "user", "content": json.dumps({"fields": fields, "schema": schema}, ensure_ascii=False)},
    ]
    call = transport if transport is not None else OpenAITransport(config)
    from openai import OpenAIError

    try:
        response = call(messages)
    except OpenAIError as exc:
        raise RuntimeError(f"大模型映射请求失败：{type(exc).__name__}") from None
    draft = decode_json(response["content"])
    if (not isinstance(draft, dict) or set(draft) != {"mapping", "unresolved"}
            or not isinstance(draft["mapping"], dict) or not isinstance(draft["unresolved"], list)):
        raise ValueError("大模型响应不符合草稿格式")
    return draft


def main():
    parser = argparse.ArgumentParser(description="气体输入检查、DAT 转换和预测")
    parser.add_argument("command", choices=("inspect", "draft", "convert", "predict"))
    parser.add_argument("input", type=Path)
    parser.add_argument("--format", choices=FORMATS, default="auto")
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch", type=int, choices=range(2, GasDriftData.batch_count + 1), help="回放批号，batch1 保留为官方模型起点")
    parser.add_argument("--schema", type=Path, help="用于大模型起草映射的字段与单位说明")
    parser.add_argument("--provider", choices=("pytrio", "deepseek"), default="pytrio")
    parser.add_argument("--model")
    parser.add_argument("--checkpoint", type=Path, default=project_root() / "baseline/gas/checkpoint.npz")
    args = parser.parse_args()
    try:
        raw = args.input.read_bytes()
        text = raw.decode("utf-8-sig")
        fmt = detect(text) if args.format == "auto" else args.format
        model = load_gas_model(args.checkpoint)
        if args.command == "convert":
            if args.output_dir is None or args.batch is None:
                parser.error("convert 需要 --output-dir 和 --batch")
            result = convert(raw, text, fmt, args.mapping, args.output_dir, args.batch, model_meta=model.meta)
        elif args.command == "predict":
            import numpy as np

            mapping = read_json(args.mapping) if args.mapping else None
            rows = list(source_rows(text, fmt, mapping, labels_required=False, model_meta=model.meta))
            if not rows:
                raise ValueError("文件中没有样本")
            scores = model.predict(np.asarray([values for _, values, _ in rows]))
            result = [{"id": sample_id, "gas": model.meta["classes"][int(index)]}
                      for (sample_id, _, _), index in zip(rows, scores.argmax(axis=1))]
        elif fmt == "dat":
            if args.command == "draft":
                raise ValueError("DAT 已经是模型格式，无需起草映射")
            count = sum(1 for _ in dat_rows(text, model.meta))
            if not count:
                raise ValueError("文件中没有样本")
            result = {"format": fmt, "rows": count, "features": len(model.meta["paths"])}
        else:
            first = next(records(text, fmt), None)
            if not first:
                raise ValueError("文件中没有样本或字段")
            fields = list(first)
            if args.command == "draft":
                if args.schema is None:
                    parser.error("draft 需要 --schema，提供字段及单位说明")
                result = draft_mapping(fields, args.schema.read_text(encoding="utf-8"),
                                       args.provider, args.model, model_meta=model.meta)
            else:
                result = {"format": fmt, "fields": fields}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, csv.Error) as exc:
        print(f"气体输入处理失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
