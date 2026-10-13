#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verification 模块 + run_all 验证去重回归测试。
运行: pytest tests/test_verification.py -v
"""
import sqlite3
import pytest


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """在临时目录建一个带数据的内存数据库，monkeypatch DB_PATH。"""
    from db_utils import get_connection
    import db_utils as _db
    db_file = tmp_path / 'test_lottery.db'
    # db_utils 在模块加载时已绑定 config.DB_PATH，所有模块都经由 get_connection
    # 访问数据库，故只需改 db_utils.DB_PATH 一处即可全局生效。
    monkeypatch.setattr(_db, 'DB_PATH', db_file)
    # 重新初始化表结构
    with get_connection() as conn:
        from schema import init_db
        init_db(conn)
    return db_file


def test_auto_verify_pending_empty(tmp_db):
    """空库时 auto_verify_predictions 返回 0 不崩溃。"""
    from verification import auto_verify_predictions
    assert auto_verify_predictions(no_notify=True) == 0


def test_auto_verify_with_data(tmp_db):
    """有预测记录且开奖已入库时，验证应更新 actual_front/match_level。"""
    from db_utils import get_connection
    from notifier import save_prediction

    # 写入一条预测（dlt 26001）和对应开奖
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO lottery_records (lottery_type, period, date, front_1, front_2, front_3, front_4, front_5, back_1, back_2)
               VALUES ('dlt', '26001', '2026-01-05', 1, 2, 3, 4, 5, 1, 2)""")
    save_prediction('dlt', '26001', [1, 2, 3, 4, 5], [1, 2])

    from verification import auto_verify_predictions
    n = auto_verify_predictions(no_notify=True)
    assert n == 1

    with get_connection(commit=False) as conn:
        row = conn.execute(
            "SELECT actual_front, front_match_count, is_matched FROM prediction_records WHERE target_period='26001'"
        ).fetchone()
    assert row[0] == '1,2,3,4,5'       # 已回填实际开奖
    assert row[1] == 5                 # 前区全中
    assert row[2] == 1                 # is_matched


def test_auto_verify_partial_hit(tmp_db):
    """部分命中时 match_level 应为对应奖级。"""
    from db_utils import get_connection
    from notifier import save_prediction
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO lottery_records (lottery_type, period, date, front_1, front_2, front_3, front_4, front_5, back_1, back_2)
               VALUES ('dlt', '26002', '2026-01-07', 1, 2, 3, 4, 5, 1, 2)""")
    # 预测只中 3 前区 + 0 后区 → 九等奖
    save_prediction('dlt', '26002', [1, 2, 3, 10, 11], [9, 10])

    from verification import auto_verify_predictions
    assert auto_verify_predictions(no_notify=True) == 1
    with get_connection(commit=False) as conn:
        row = conn.execute(
            "SELECT match_level, front_match_count, back_match_count FROM prediction_records WHERE target_period='26002'"
        ).fetchone()
    assert row[1] == 3
    assert row[2] == 0
    assert '九等奖' in row[0]
