"""统一运行入口；命令参数与 python -m harness_demo 完全相同。"""

from harness_demo.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
