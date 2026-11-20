"""统一运行入口；命令参数与 python -m harness_core 完全相同。"""

from harness_core.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
