"""
统一的彩色分级日志工具。

用法：
    from utils.logger import get_logger
    logger = get_logger(__name__)

    logger.debug("调试信息")
    logger.info("正常日志")
    logger.warning("警告信息")
    logger.error("报错信息")
    logger.exception("报错信息（自动附带异常堆栈）")

特点：
    1. 不同级别使用不同颜色，终端中一眼区分正常日志与报错；
    2. 输出包含时间、级别、模块名、行号，便于定位；
    3. 输出重定向到文件（非 TTY）时自动去除颜色码；
    4. 可选同时写入日志文件。
"""

import logging
import sys

# ANSI 颜色码
_COLORS = {
    "RESET": "\033[0m",
    "GREY": "\033[38;5;244m",     # DEBUG   - 灰色
    "GREEN": "\033[32m",          # INFO    - 绿色
    "YELLOW": "\033[33m",         # WARNING - 黄色
    "RED": "\033[31m",            # ERROR   - 红色
    "BOLD_RED": "\033[1;31m",     # CRITICAL- 加粗红
    "CYAN": "\033[36m",           # 时间戳  - 青色
}

# 级别 -> 颜色映射
_LEVEL_COLOR = {
    logging.DEBUG: _COLORS["GREY"],
    logging.INFO: _COLORS["GREEN"],
    logging.WARNING: _COLORS["YELLOW"],
    logging.ERROR: _COLORS["RED"],
    logging.CRITICAL: _COLORS["BOLD_RED"],
}

# 已配置过的 logger 名称，避免重复添加 handler
_CONFIGURED = set()


class ColorFormatter(logging.Formatter):
    """带颜色的日志格式化器；当输出目标不是终端时自动禁用颜色。"""

    def __init__(self, fmt=None, datefmt=None, use_color=True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color

    def format(self, record):
        message = super().format(record)
        if self.use_color:
            color = _LEVEL_COLOR.get(record.levelno, _COLORS["RESET"])
            message = f"{color}{message}{_COLORS['RESET']}"
        return message


def _build_fmt():
    # 时间 | 级别(固定宽度) | 模块:行号 | 内容
    #return "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    # 级别(固定宽度) | 模块:行号 | 内容
    return "%(levelname)-8s | %(name)s:%(lineno)d | %(message)s"


def get_logger(name="stereo", level=logging.INFO, log_file=None,
               file_level=logging.DEBUG):
    """
    获取（或创建）一个配置好的 logger。

    参数:
        name       : logger 名称，一般传 __name__。
        level      : 控制台输出级别，默认 INFO。
        log_file   : 若指定，则同时把日志写入该文件。
        file_level : 写入文件的级别，默认 DEBUG（更详细）。

    返回:
        logging.Logger
    """
    logger = logging.getLogger(name)

    # 避免同一 logger 重复添加 handler
    if name in _CONFIGURED:
        return logger

    logger.setLevel(min(level, file_level) if log_file else level)
    logger.propagate = False  # 不向 root 传播，避免重复打印

    datefmt = "%Y-%m-%d %H:%M:%S"
    fmt = _build_fmt()

    # ---- 控制台 handler ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    use_color = sys.stdout.isatty()
    console_handler.setFormatter(ColorFormatter(fmt, datefmt, use_color=use_color))
    logger.addHandler(console_handler)

    # ---- 文件 handler（可选，不写颜色码）----
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(file_handler)

    _CONFIGURED.add(name)
    return logger
