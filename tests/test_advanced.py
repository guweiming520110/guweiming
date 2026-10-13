#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D项测试补充：钉钉签名/去重/限流 + evaluate完整流程 + auto_tune调参。
运行: pytest tests/test_advanced.py -v
"""
import json
import time
import hashlib
import hmac
import base64
import urllib.parse
import numpy as np
import pandas as pd
import pytest


# ==================== 钉钉签名与去重 ====================

def test_sign_algorithm():
    """钉钉签名: HMAC-SHA256(timestamp\nsecret) 后用 base64，与官方规范一致。"""
    import notifier as nt
    # 用已知输入验证算法正确性
    ts = 1710000000000
    secret = 'test_secret'
    nt.SECRET = secret
    sign = nt._sign(ts)
    # 独立重算验证
    string_to_sign = f"{ts}\n{secret}"
    expected = base64.b64encode(
        hmac.new(secret.encode(), string_to_sign.encode(), hashlib.sha256).digest()
    ).decode()
    assert sign == expected
    # URL 含 timestamp 和 sign 且 sign 已 URL 编码
    nt.WEBHOOK_URL = 'https://oapi.dingtalk.com/robot/send?access_token=abc'
    url = nt._build_url(ts)
    assert 'timestamp=1710000000000' in url
    assert 'sign=' in url
    assert urllib.parse.unquote(url).split('sign=')[1] == expected


def test_dedup_normalizes_timestamps():
    """去重键应把消息内时间戳统一替换为占位符（否则去重失效）。"""
    import notifier as nt
    p1 = {'msgtype': 'text', 'text': {'content': 'a 2026-10-13 11:00:00'}}
    p2 = {'msgtype': 'text', 'text': {'content': 'a 2026-10-13 12:30'}}
    assert nt._dedup_key(p1) == nt._dedup_key(p2)


def test_rate_limit_dedup_behavior(monkeypatch):
    """限流与去重逻辑存在且参数正确（不实际发送）。"""
    import notifier as nt
    from config import DINGTALK_RATE_LIMIT, DINGTALK_RATE_WINDOW
    # 配置值合理：20条/60秒
    assert DINGTALK_RATE_LIMIT == 20
    assert DINGTALK_RATE_WINDOW == 60
    # 去重键：时间戳被归一化
    p1 = {'text': {'content': 'a 2026-10-13 11:00:00'}}
    p2 = {'text': {'content': 'a 2026-10-13 12:30'}}
    assert nt._dedup_key(p1) == nt._dedup_key(p2)
    # 去重缓存仅在成功后写入的语义：初始为空
    assert nt._last_send_cache['content'] == ''


# ==================== evaluate 完整流程 ====================

def test_evaluate_runs_all_types(monkeypatch):
    """evaluate.evaluate 对三彩种跑通（用测试数据 monkeypatch 数据加载）。"""
    from evaluate import evaluate

    # monkeypatch load_data 返回内存数据，避免依赖真实库
    rng = np.random.RandomState(1)
    frames = {}
    # dlt
    frames['dlt'] = pd.DataFrame({
        'period': [f"25{i:03d}" for i in range(1, 61)],
        'date': ['2025-01-01'] * 60,
        **{f'front_{k}': [int(rng.choice(range(1, 36))) for _ in range(60)] for k in range(1, 6)},
        **{f'back_{k}': [int(rng.choice(range(1, 13))) for _ in range(60)] for k in range(1, 3)},
    })
    # ssq
    frames['ssq'] = pd.DataFrame({
        'period': [f"25{i:03d}" for i in range(1, 61)],
        'date': ['2025-01-01'] * 60,
        **{f'front_{k}': [int(rng.choice(range(1, 34))) for _ in range(60)] for k in range(1, 7)},
        'back_1': [int(rng.choice(range(1, 17))) for _ in range(60)],
    })
    # kl8
    frames['kl8'] = pd.DataFrame({
        'period': [f"2026{i:03d}" for i in range(1, 61)],
        'date': ['2026-01-01'] * 60,
        'num_list': [sorted(rng.choice(range(1, 81), 20, replace=False).tolist()) for _ in range(60)],
    })

    import evaluate as ev
    monkeypatch.setattr(ev, 'load_data', lambda lt: frames[lt])
    # 用无 ML 模式（快），horizon 小
    ev.evaluate('all', horizon=10, with_ml=False)
    # 能跑通即通过（内部打印结果）


def test_hypergeom_stats_correct():
    """超几何期望/方差/z 检验数值正确性。"""
    from evaluate import hypergeom_mean, hypergeom_var, zscore
    # dlt: 抽5/35，开奖5，期望 5*5/35
    assert abs(hypergeom_mean(5, 5, 35) - 0.7142857) < 1e-5
    # 方差为正且随 N 增大减小
    assert hypergeom_var(5, 5, 35) < hypergeom_var(5, 5, 10)


# ==================== auto_tune 调参流程 ====================

def test_auto_tune_get_config():
    """调参配置含关键参数且范围合理。"""
    from auto_tune import get_tune_config
    cfg = get_tune_config()
    assert 'rarity_max' in cfg
    assert 'time_decay_weight' in cfg
    assert cfg['rarity_max']['low'] < cfg['rarity_max']['high']


def test_auto_tune_objective_returns_number(monkeypatch):
    """objective 返回数值（用最小数据量避免超时）。"""
    from auto_tune import objective
    import auto_tune as at

    # monkeypatch 数据加载为小样本
    rng = np.random.RandomState(3)
    df = pd.DataFrame({
        'period': [f"25{i:03d}" for i in range(1, 41)],
        'date': ['2025-01-01'] * 40,
        **{f'front_{k}': [int(rng.choice(range(1, 36))) for _ in range(40)] for k in range(1, 6)},
        **{f'back_{k}': [int(rng.choice(range(1, 13))) for _ in range(40)] for k in range(1, 3)},
    })
    monkeypatch.setattr(at, 'load_data_for_type', lambda lt: df)

    class _Trial:
        def suggest_float(self, name, low, high, step=None):
            return (low + high) / 2
        def suggest_int(self, name, low, high, step=None):
            return (low + high) // 2

    score = objective(_Trial(), lottery_type='dlt', backtest_n=5)
    # 数据不足(40 < 5*2+30=40 边界)时返回 0，否则返回数值
    assert isinstance(score, float)
