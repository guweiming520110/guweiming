#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
notifier（钉钉）与爬虫解析回归测试。
运行: pytest tests/test_notifier.py -v
"""
import pytest


# ==================== notifier 消息格式化 ====================

def test_format_prediction_message_basic():
    """预测消息格式含目标期号与号码。"""
    from notifier import format_prediction_message
    front = [(1, 20.0, {}), (2, 18.0, {})]
    msg = format_prediction_message('dlt', front, back_preds=None,
                                    front_report='report', target_period='26120')
    assert '26120' in msg
    assert '1' in msg and '2' in msg


def test_format_prediction_message_kl8():
    """kl8 预测消息含投注组合。"""
    from notifier import format_prediction_message
    from prize_calculator import generate_kl8_combinations
    preds = [(n, 25 - n, {}) for n in range(1, 25)]
    combos = generate_kl8_combinations(preds)
    msg = format_prediction_message('kl8', preds, kl8_combos=combos,
                                    front_report='', target_period='2026272')
    assert '2026272' in msg
    assert '单式A' in msg


def test_extract_backtest_from_report():
    """从报告中提取回测统计（M8 修复回归）。"""
    from notifier import _extract_backtest_from_report
    report = "回测结果（最近5期，5期有效）\n  平均命中: 1.20/5\n  平均Top10命中: 2.40\n  最大单期命中: 2"
    lines = _extract_backtest_from_report(report)
    assert any('平均命中' in l for l in lines)


def test_accuracy_stats_kl8_key():
    """kl8 统计字段使用 avg_hit_count（L5 修复）。"""
    from notifier import get_accuracy_stats
    stats = get_accuracy_stats()
    kl8 = stats.get('kl8', {})
    # 即使无验证数据也不崩溃，且不会引用旧字段名
    assert 'avg_hit_count' in kl8 or 'avg_top10_hit' not in kl8


# ==================== 爬虫解析 ====================

def test_fill_dlt_parser_empty():
    """空 HTML 返回 None 不崩溃。"""
    from fill_dlt_data import parse_dlt_numbers
    assert parse_dlt_numbers(None) is None
    assert parse_dlt_numbers('') is None


def test_fill_ssq_parser_empty():
    """空 HTML 返回 None 不崩溃。"""
    from fill_ssq_data import parse_ssq_numbers
    assert parse_ssq_numbers(None) is None
    assert parse_ssq_numbers('') is None


def test_kl8_parser_empty():
    """空 HTML 返回 None 不崩溃。"""
    from kl8_data import parse_kl8_numbers
    assert parse_kl8_numbers(None) is None
    assert parse_kl8_numbers('') is None


def test_http_utils_4xx_no_retry(monkeypatch):
    """M7: 403 应直接返回 None 不重试。"""
    import http_utils
    import requests

    def fake_get(*a, **kw):
        r = requests.Response()
        r.status_code = 403
        r.url = 'http://x'
        raise requests.exceptions.HTTPError('403', response=r)

    monkeypatch.setattr(requests, 'get', fake_get)
    assert http_utils.get_with_retry('http://x', max_retries=3) is None
