#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一日志配置模块
提供控制台+文件双输出日志，替代分散的 print() 调用

用法:
    from logger import log
    log.info("...") / log.warning("...") / log.error("...")
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    from config import LOGS_DIR
except ImportError:  # 单独调试时的兜底
    LOGS_DIR = Path(__file__).resolve().parent / 'logs'


def setup_logger(name='lottery', level=logging.INFO):
    """
    创建并配置日志记录器

    控制台按 level 输出，文件固定记录 DEBUG 及以上并做轮转，
    避免 logs/lottery.log 无上限增长。

    参数:
        name: 日志记录器名称
        level: 控制台日志级别

    返回:
        logging.Logger 实例
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    # 记录器级别必须不高于文件 handler 级别，否则 DEBUG 永远不会落盘
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s', '%m-%d %H:%M:%S')

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_dir = Path(LOGS_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        log_dir / 'lottery.log', maxBytes=5 * 1024 * 1024,
        backupCount=5, encoding='utf-8'
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# 全局日志实例，其他模块直接 from logger import log
log = setup_logger()
