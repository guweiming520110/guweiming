#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据库表结构统一管理模块
集中定义所有表的 DDL，避免各文件重复建表逻辑

v2 变更:
  - prediction_records 增加 UNIQUE(lottery_type, target_period)，杜绝同期号重复记录
  - prediction_records 增加 created_at
  - 增加 actual_front 索引，加速"待验证记录"查询
  - 移除与 UNIQUE(period) 重复的 idx_kl8_period
"""

import sqlite3

from logger import log

SCHEMA_VERSION = 2

DDL_LOTTERY_RECORDS = """
CREATE TABLE IF NOT EXISTS lottery_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lottery_type VARCHAR(20) NOT NULL,
    period VARCHAR(20) NOT NULL,
    date VARCHAR(30) NOT NULL DEFAULT '',
    front_1 INTEGER, front_2 INTEGER, front_3 INTEGER,
    front_4 INTEGER, front_5 INTEGER, front_6 INTEGER,
    back_1 INTEGER, back_2 INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(lottery_type, period)
)
"""

DDL_PREDICTION_RECORDS = """
CREATE TABLE IF NOT EXISTS prediction_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lottery_type VARCHAR(20) NOT NULL,
    predict_time VARCHAR(30) NOT NULL DEFAULT '',
    target_period VARCHAR(20) NOT NULL DEFAULT '',
    play_type VARCHAR(50) NOT NULL DEFAULT '几何图案预测',
    front_numbers VARCHAR(200) NOT NULL DEFAULT '',
    back_numbers VARCHAR(100) NOT NULL DEFAULT '',
    front_probabilities VARCHAR(1000) NOT NULL DEFAULT '',
    back_probabilities VARCHAR(500) NOT NULL DEFAULT '',
    front_dan VARCHAR(50) NOT NULL DEFAULT '',
    front_tuo VARCHAR(200) NOT NULL DEFAULT '',
    back_dan VARCHAR(20) NOT NULL DEFAULT '',
    back_tuo VARCHAR(50) NOT NULL DEFAULT '',
    combination_count INTEGER NOT NULL DEFAULT 0,
    actual_front VARCHAR(100) NOT NULL DEFAULT '',
    actual_back VARCHAR(50) NOT NULL DEFAULT '',
    is_matched INTEGER NOT NULL DEFAULT 0,
    match_level VARCHAR(20) NOT NULL DEFAULT '',
    front_match_count INTEGER NOT NULL DEFAULT 0,
    back_match_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(lottery_type, target_period)
)
"""

DDL_KL8_RECORDS = """
CREATE TABLE IF NOT EXISTS kl8_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    period TEXT NOT NULL UNIQUE,
    date TEXT NOT NULL DEFAULT '',
    numbers TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_lottery_type_period ON lottery_records(lottery_type, period)",
    "CREATE INDEX IF NOT EXISTS idx_prediction_type_period ON prediction_records(lottery_type, target_period)",
    # 加速"未验证记录"扫描：run_all / verification 每次启动都要查
    "CREATE INDEX IF NOT EXISTS idx_prediction_actual_front ON prediction_records(actual_front)",
]


def init_db(conn):
    """
    初始化所有表结构（幂等操作，可安全重复调用）

    参数:
        conn: sqlite3.Connection 实例
    """
    conn.execute(DDL_LOTTERY_RECORDS)
    conn.execute(DDL_PREDICTION_RECORDS)
    conn.execute(DDL_KL8_RECORDS)
    for idx_sql in DDL_INDEXES:
        conn.execute(idx_sql)
    conn.commit()
    migrate_db(conn)


def _get_columns(conn, table):
    return [r[1] for r in conn.execute("PRAGMA table_info(%s)" % table).fetchall()]


def _has_unique_prediction_index(conn):
    """判断 prediction_records 是否已带 UNIQUE 约束"""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='prediction_records'"
    ).fetchone()
    return bool(row and 'UNIQUE' in (row[0] or '').upper())


def migrate_db(conn):
    """
    把已有库升级到 SCHEMA_VERSION。

    老库是在 UNIQUE 约束引入之前建的，而 CREATE TABLE IF NOT EXISTS 对已存在的表
    不生效，因此必须通过 user_version + 重建表的方式补齐约束。
    """
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0] or 0
    except sqlite3.Error:
        version = 0

    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_records'"
    ).fetchone()
    if not exists:
        conn.execute("PRAGMA user_version = %d" % SCHEMA_VERSION)
        return

    cols = _get_columns(conn, 'prediction_records')
    if cols and 'created_at' not in cols:
        # SQLite 的 ALTER TABLE ADD COLUMN 不允许非常量默认值（CURRENT_TIMESTAMP），
        # 只能先加列再回填
        conn.execute("ALTER TABLE prediction_records ADD COLUMN created_at TIMESTAMP")
        conn.execute(
            "UPDATE prediction_records SET created_at = CURRENT_TIMESTAMP "
            "WHERE created_at IS NULL OR created_at = ''"
        )
        cols = _get_columns(conn, 'prediction_records')

    if not _has_unique_prediction_index(conn):
        # 先清理历史重复数据（同彩种同期号只保留最早一条），否则建唯一索引会失败
        cur = conn.execute(
            "DELETE FROM prediction_records WHERE id NOT IN ("
            "  SELECT MIN(id) FROM prediction_records GROUP BY lottery_type, target_period"
            ")"
        )
        if cur.rowcount > 0:
            log.warning("迁移: 已清理 %d 条重复的预测记录", cur.rowcount)

        shared = [c for c in cols if c != 'id']
        collist = ', '.join(shared)
        conn.execute(
            DDL_PREDICTION_RECORDS.replace('prediction_records', 'prediction_records_new', 1)
        )
        conn.execute(
            "INSERT INTO prediction_records_new (%s) SELECT %s FROM prediction_records"
            % (collist, collist)
        )
        conn.execute("DROP TABLE prediction_records")
        conn.execute("ALTER TABLE prediction_records_new RENAME TO prediction_records")
        for idx_sql in DDL_INDEXES:
            if 'prediction_records' in idx_sql:
                conn.execute(idx_sql)
        log.info("迁移: prediction_records 已升级到 v%d 结构", SCHEMA_VERSION)

    conn.execute("PRAGMA user_version = %d" % SCHEMA_VERSION)
    conn.commit()
