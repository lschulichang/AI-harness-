"""将数据路径保存为项目相对路径，规范化并校验模型路径。"""

import os
from pathlib import Path, PureWindowsPath


def saved_data_path(path, project):
    path, project = Path(path).resolve(), Path(project).resolve()
    try:
        return path.relative_to(project).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_model_path(run_dir, reference):
    root = Path(run_dir).resolve()
    if os.name != "nt" and PureWindowsPath(str(reference)).drive:
        raise ValueError("模型应使用运行目录内的相对路径，不能使用 Windows 绝对路径")
    path = (root / str(reference).replace("\\", "/")).resolve()
    if not path.is_relative_to(root):
        raise ValueError("模型路径超出运行目录")
    return path
