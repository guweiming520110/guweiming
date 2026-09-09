#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一数据库连接管理模块
提供安全的数据库连接上下文管理器，避免连接泄漏

全项目所有 SQLite 访问都应通过这里，禁止业务文件裸调 sqlite3.connect。
"""

import sqlite3
from contextlib import contextmanager

from config import DB_PATH, DB_TIMEOUT
from logger import log


@contextmanager
def get_connection(row_factory=None, timeout=None, commit=True):
    """
    安全的数据库连接上下文管理器

    用法:
        with get_connection() as conn:
            conn.execute("SELECT ...")

        with get_connection(sqlite3.Row) as conn:
            row = conn.execute("SELECT ...").fetchone()
            print(row['column_name'])

    参数:
        row_factory: 行工厂(如 sqlite3.Row)，None 表示使用默认 tuple
        timeout: 锁等待超时(秒)，默认读取 config.DB_TIMEOUT
        commit: 退出时是否提交事务；纯查询可设为 False 避免无谓写锁
    """
    conn = sqlite3.connect(str(DB_PATH), timeout=timeout or DB_TIMEOUT)
    if row_factory:
        conn.row_factory = row_factory
    try:
        # WAL 模式：读写不互斥，显著降低 "database is locked" 概率（数据库级持久属性）
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=%d" % int((timeout or DB_TIMEOUT) * 1000))
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_db_status():
    """返回库中各表的记录数，便于启动自检与排障"""
    status = {}
    try:
        with get_connection(commit=False) as conn:
            for table in ('lottery_records', 'prediction_records', 'kl8_records'):
                try:
                    row = conn.execute("SELECT COUNT(*) FROM %s" % table).fetchone()
                    status[table] = row[0] if row else 0
                except sqlite3.Error:
                    status[table] = None
    except sqlite3.Error as e:
        log.warning("数据库状态查询失败: %s", e)
    return status
