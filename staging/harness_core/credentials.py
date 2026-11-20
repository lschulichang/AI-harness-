"""环境变量优先，其次读取项目根目录的本机密钥源码。"""

import ast
import os

from .config import project_root


def _local_values():
    path = project_root() / "local_secrets.py"
    if not path.exists():
        return {}
    values = {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.endswith("_API_KEY"):
                        values[target.id] = ast.literal_eval(node.value)
    except (SyntaxError, ValueError, TypeError):
        raise RuntimeError("local_secrets.py 格式有误，请使用字符串常量定义密钥") from None
    return values


def get_secret(name):
    value = os.environ.get(name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    value = _local_values().get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None
