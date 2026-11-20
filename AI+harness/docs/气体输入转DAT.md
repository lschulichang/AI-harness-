# 气体输入转为 DAT

转换、映射和预测按 --checkpoint 指定模型的特征维度与类别顺序处理。下面的 128 维和六类示例对应仓库自带的气体基线。转换后的数据用于回放时，请使用匹配的 --baseline-dir。

现有的 batch*.dat 可继续直接回放。外部文件通过 `harness_demo.gas_import` 识别格式，再按已确认的字段映射生成同样的 DAT。当前支持 DAT、带表头的 CSV/TSV、JSON 对象或对象数组、JSONL，输入文本使用 UTF-8。其他格式先使用相应工具导出其中一种。原始波形需要先提取与检查点匹配的特征。

在项目根目录运行。下文的 `python` 指项目 Conda 环境里的 Python。

~~~bash
python -m harness_demo.gas_import inspect 外部文件
python -m harness_demo.gas_import --help
~~~

如果清楚字段定义，可以直接写 `mapping.json`。其中：

- `feature_columns` 是含 128 个字段的数组，第一个对应模型第 1 维，第二个对应第 2 维，以此类推。JSON 使用如 `inputs.features.0` 的路径。
- `label_column` 指真实标签字段。已用气体全名或官方 1–6 编号时可省略 `label_map`。自定义编码用 `"label_map": {"E": "Ethanol"}` 这样的映射。
- `id_column` 可选。提供时检查 ID 是否为空或重复，预测结果以该 ID 返回。

可让大模型根据字段说明起草映射。此命令会向所选服务发送字段路径和说明文件的内容，使用项目已有 API 配置。

~~~bash
python -m harness_demo.gas_import draft 外部文件 --schema 字段说明.txt > 草稿.json
~~~

对照提供方的定义核对特征位置、单位和标签，处理 `unresolved` 中的问题，然后把 `mapping` 对象单独保存到 `mapping.json`。数值和列名的校验无法证明特征的物理含义相同。

~~~bash
python -m harness_demo.gas_import convert 外部文件 \
  --mapping mapping.json --batch 3 --output-dir data/gas-import
bash run_linux.sh run --dataset gas_mlp --data-dir data/gas-import \
  --strategy rules --swanlab-mode disabled
~~~

批号为 3–10，按实际批次指定，程序按该顺序回放。可以只导入一批，重复执行转换可加入后续批次。第 1、2 批保留给现有基线训练和公开验证。直接导入 DAT 时省略 `--mapping`。训练、验证和留出测试仍由 Harness 划分。运行恢复时会校验导入清单和 DAT 摘要，添加或修改批次后应新建运行。

无标签数据可预测，字段映射可以省略 `label_column`：

~~~bash
python -m harness_demo.gas_import predict 外部文件 --mapping mapping.json > 预测.json
~~~

`--checkpoint` 可指定已有气体检查点，默认使用官方基线。真实标签到达后，使用带标签的文件做训练或评估。
