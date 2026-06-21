from __future__ import annotations

import argparse
import sys
from typing import Any


class ChineseArgumentParser(argparse.ArgumentParser):
    """默认使用中文帮助标题和错误前缀的 ArgumentParser。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("add_help", False)
        super().__init__(*args, **kwargs)
        self._positionals.title = "位置参数"
        self._optionals.title = "选项"
        self.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "用法:", 1)

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "用法:", 1)

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: 错误: {message}\n")
