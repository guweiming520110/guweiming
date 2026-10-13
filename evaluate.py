#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测模型诚实评估基准（walk-forward + 统计显著性检验）

背景：
  彩票开奖是均匀随机事件，任何历史数据驱动的模型若不能显著优于随机基线，
  就不应进入融合环节（否则只会拉低结果，甚至"负学习"）。

本模块对每个彩种运行 walk-forward 回测（用前 N 期训练/预测第 N+1 期），
对比以下策略：
  - random_baseline : 纯随机（理论超几何期望 + 蒙特卡洛验证）
  - hot_strategy    : 热号（历史出现频率最高的 k 个）
  - cold_strategy   : 冷号（近期出现频率最低的 k 个）
  - pattern_engine  : 几何图案引擎（当前系统主力）
  - ml_set          : ML 集合预测（号码二分类多标签，见 ml_models.set_predictor）

输出：
  - 每策略平均命中/期 vs 理论随机
  - z 值（超几何方差做标准误，|z|>1.96 即 95% 置信显著）
  - 结论（显著优于/≈随机/显著劣于）

用法:
  python evaluate.py                 # 全部彩种
  python evaluate.py --type dlt      # 仅大乐透
  python evaluate.py --horizon 60    # 回测期数
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db_utils import get_connection


# 抑制 PatternLearner 训练过程的长输出（图案频率统计等），只保留评估表
import io as _io
import contextlib as _ctx
import os as _os
_DEVNULL = _os.devnull


def _quiet_train(lottery_type, train_df, num_periods):
    """静默版训练：丢弃图案引擎训练期的大量 print 输出。"""
    from pattern_learner.core import PatternLearner
    pl = PatternLearner(lottery_type)
    with _ctx.redirect_stdout(_io.StringIO()):
        pl.train(train_df, num_periods=num_periods)
    return pl


# ==================== 超几何分布统计 ====================

def hypergeom_mean(n_draw, n_hit_total, n_pool):
    """从 n_pool 个号码中抽 n_draw 个，开奖命中集合大小为 n_hit_total，命中数期望。"""
    return n_draw * n_hit_total / n_pool


def hypergeom_var(n_draw, n_hit_total, n_pool):
    """对应方差（不放回抽样）。"""
    p = n_hit_total / n_pool
    return n_draw * p * (1 - p) * (n_pool - n_draw) / (n_pool - 1)


def zscore(obs_mean, n_draw, n_hit_total, n_pool, n_periods):
    """观测均值相对随机基线的 z 值（标准误用超几何方差/期数）。"""
    var_period = hypergeom_var(n_draw, n_hit_total, n_pool)
    se = np.sqrt(var_period / n_periods)
    if se == 0:
        return 0.0
    return (obs_mean - hypergeom_mean(n_draw, n_hit_total, n_pool)) / se


# ==================== 数据加载 ====================

LOTTERY_META = {
    'dlt': {'table': 'lottery_records', 'front_cols': ['front_1', 'front_2', 'front_3', 'front_4', 'front_5'],
            'front_pool': 35, 'front_draw': 5, 'has_back': True,
            'back_cols': ['back_1', 'back_2'], 'back_pool': 12, 'back_draw': 2},
    'ssq': {'table': 'lottery_records', 'front_cols': ['front_1', 'front_2', 'front_3', 'front_4', 'front_5', 'front_6'],
            'front_pool': 33, 'front_draw': 6, 'has_back': True,
            'back_cols': ['back_1'], 'back_pool': 16, 'back_draw': 1},
    'kl8': {'table': 'kl8_records', 'front_cols': [], 'front_pool': 80, 'front_draw': 20,
            'has_back': False, 'back_cols': [], 'back_pool': 0, 'back_draw': 0},
}


def load_data(lottery_type):
    meta = LOTTERY_META[lottery_type]
    with get_connection(commit=False) as conn:
        if lottery_type == 'kl8':
            df = pd.read_sql("SELECT period, numbers FROM kl8_records ORDER BY period", conn)
            df['num_list'] = [[int(x) for x in s.split(',') if x.strip().isdigit()] for s in df['numbers']]
        else:
            cols = ['period'] + meta['front_cols'] + meta['back_cols']
            df = pd.read_sql(
                f"SELECT {', '.join(cols)} FROM lottery_records WHERE lottery_type=? ORDER BY period",
                conn, params=(lottery_type,))
    return df


def _actual_set(row, meta):
    if meta['front_cols']:
        return set(int(row[c]) for c in meta['front_cols'])
    return set(row['num_list'])


# ==================== 策略实现 ====================

def random_strategy(train, meta, lottery_type=None):
    return set(np.random.choice(range(1, meta['front_pool'] + 1), size=meta['front_draw'], replace=False))


def hot_strategy(train, meta, lottery_type=None):
    """历史全量频率最高的 k 个号码。"""
    freq = np.zeros(meta['front_pool'] + 1)
    if meta['front_cols']:
        for _, r in train.iterrows():
            for c in meta['front_cols']:
                freq[int(r[c])] += 1
    else:
        for _, r in train.iterrows():
            for n in r['num_list']:
                freq[n] += 1
    return set(np.argsort(freq)[-meta['front_draw']:])


def cold_strategy(train, meta, lottery_type=None):
    """近 20 期出现频率最低的 k 个号码。"""
    freq = np.zeros(meta['front_pool'] + 1)
    recent = train.tail(20)
    if meta['front_cols']:
        for _, r in recent.iterrows():
            for c in meta['front_cols']:
                freq[int(r[c])] += 1
    else:
        for _, r in recent.iterrows():
            for n in r['num_list']:
                freq[n] += 1
    return set(np.argsort(freq)[:meta['front_draw']])


def pattern_engine_strategy(train, meta, lottery_type):
    """几何图案引擎 Top-k（静默训练）。"""
    pl = _quiet_train(lottery_type, train, min(50, len(train)))
    preds = pl.predict(top_k=meta['front_draw'] * 2)
    return set(n for n, s, d in preds[:meta['front_draw']])


def ml_set_strategy(train, meta, lottery_type):
    """ML 集合预测（号码二分类多标签）。"""
    from ml_models import SetNumberPredictor
    sp = SetNumberPredictor(lottery_type)
    sp.train(train)
    return set(sp.predict_topk())


# ==================== walk-forward 回测 ====================

def walk_forward(df, meta, lottery_type, strategy, horizon):
    """返回每期命中数序列。"""
    hits = []
    n = len(df)
    for t in range(horizon):
        train = df.iloc[:n - horizon + t]
        if len(train) < 30:
            continue
        try:
            pred = strategy(train, meta, lottery_type)
        except Exception:
            continue
        actual = _actual_set(df.iloc[n - horizon + t], meta)
        hits.append(len(pred & actual))
    return np.array(hits)


def _run_one(df, meta, lottery_type, horizon, with_ml=True):
    rows = []
    n_pool = meta['front_pool']
    n_draw = meta['front_draw']

    # 理论随机基线
    theo_mean = hypergeom_mean(n_draw, n_draw, n_pool)
    rows.append({
        'strategy': '理论随机',
        'avg_hit': theo_mean,
        'z': 0.0,
        'verdict': '基准',
    })

    # 蒙特卡洛随机实测（200 轮求均值作对照）
    mc_hits = []
    for _ in range(200):
        arr = walk_forward(df, meta, lottery_type, random_strategy, horizon)
        mc_hits.append(arr.mean())
    rows.append({
        'strategy': '随机(蒙特卡洛)',
        'avg_hit': np.mean(mc_hits),
        'z': 0.0,
        'verdict': '基准',
    })

    strategies = [('热号', hot_strategy), ('冷号', cold_strategy), ('图案引擎', pattern_engine_strategy)]
    if with_ml:
        strategies.append(('ML集合', ml_set_strategy))

    for name, fn in strategies:
        arr = walk_forward(df, meta, lottery_type, fn, horizon)
        if len(arr) == 0:
            continue
        obs = arr.mean()
        z = zscore(obs, n_draw, n_draw, n_pool, len(arr))
        if z > 1.96:
            verdict = '✅ 显著优于随机'
        elif z < -1.96:
            verdict = '❌ 显著劣于随机'
        else:
            verdict = '➖ ≈随机(不显著)'
        rows.append({'strategy': name, 'avg_hit': obs, 'z': z, 'verdict': verdict})

    return rows


def evaluate(lottery_type='all', horizon=60, with_ml=True):
    types = ['dlt', 'ssq', 'kl8'] if lottery_type == 'all' else [lottery_type]
    for lt in types:
        if lt not in LOTTERY_META:
            print(f"未知彩种: {lt}")
            continue
        print("\n" + "=" * 62)
        print(f"  {lt.upper()} walk-forward 回测（{horizon} 期）")
        print("=" * 62)
        df = load_data(lt)
        meta = LOTTERY_META[lt]
        if len(df) < 50:
            print(f"  数据不足({len(df)}期)，跳过")
            continue
        print(f"  数据量: {len(df)} 期")
        rows = _run_one(df, meta, lt, min(horizon, len(df) - 30), with_ml=with_ml)
        print(f"\n  {'策略':<18}{'平均命中/期':>12}{'z值':>8}  结论")
        print(f"  {'-'*18}{'-'*12}{'-'*8}  {'-'*20}")
        for r in rows:
            print(f"  {r['strategy']:<18}{r['avg_hit']:>10.3f}{r['z']:>8.2f}  {r['verdict']}")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='预测模型诚实评估基准')
    parser.add_argument('--type', default='all', help='彩种: dlt/ssq/kl8/all (默认 all)')
    parser.add_argument('--horizon', type=int, default=60, help='回测期数 (默认 60)')
    parser.add_argument('--no-ml', action='store_true', help='跳过 ML 集合预测(较慢)')
    args = parser.parse_args()
    evaluate(args.type, args.horizon, with_ml=not args.no_ml)
