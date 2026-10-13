#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P4: 自动调参与集成优化

使用 Optuna + 回测驱动，自动搜索图案引擎评分公式的最优参数组合。

用法:
    python auto_tune.py                          # 默认（大乐透前区）
    python auto_tune.py --type dlt_back           # 大乐透后区
    python auto_tune.py --type ssq                # 双色球前区
    python auto_tune.py --trials 100              # 100次试验
    python auto_tune.py --show-best               # 仅显示已保存的最佳参数
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 默认试次数
DEFAULT_TRIALS = 50


def get_tune_config():
    """获取调参配置：参数名、类型、范围"""
    return {
        # 稀有度
        'rarity_min': {'type': 'float', 'low': 0.1, 'high': 1.0, 'step': 0.1},
        'rarity_max': {'type': 'float', 'low': 3.0, 'high': 8.0, 'step': 0.5},
        # 跨期加成
        'span_bonus_factor': {'type': 'float', 'low': 0.2, 'high': 1.0, 'step': 0.1},
        'span_bonus_cap': {'type': 'float', 'low': 1.5, 'high': 4.0, 'step': 0.25},
        # 水平惩罚
        'horiz_penalty_max': {'type': 'float', 'low': 0.3, 'high': 1.0, 'step': 0.1},
        # 多样性
        'diversity_factor': {'type': 'float', 'low': 0.01, 'high': 0.15, 'step': 0.01},
        # 遗漏值
        'omission_factor': {'type': 'float', 'low': 0.02, 'high': 0.25, 'step': 0.02},
        'omission_max_mult': {'type': 'float', 'low': 0.5, 'high': 3.0, 'step': 0.5},
        # 时间衰减
        'time_decay': {'type': 'float', 'low': 0.8, 'high': 0.99, 'step': 0.01},
        'time_decay_weight': {'type': 'float', 'low': 0.0, 'high': 0.8, 'step': 0.1},
    }


def load_data_for_type(lottery_type='dlt'):
    """加载指定彩种的数据"""
    from generate_charts import load_data, load_kl8_data
    if lottery_type in ('kl8',):
        return load_kl8_data(limit=100)
    elif lottery_type in ('dlt', 'dlt_back'):
        df = load_data(lottery_type='dlt')
        return df.tail(100) if df is not None and len(df) > 100 else df
    elif lottery_type in ('ssq', 'ssq_back'):
        df = load_data(lottery_type='ssq', limit=100)
        return df
    return None


def objective(trial, lottery_type='dlt', backtest_n=20):
    """
    Optuna 目标函数：运行回测，返回平均命中数

    最大化: avg_hit (平均每期命中个数)

    注意: 这里只在 validation 窗口(回测的最后 backtest_n 期)上评估，
    最终 best_score 存在乐观偏差(同一窗口既选参又打分)，真正的泛化能力
    需用更靠后的独立窗口验证(见 evaluate_generalization)。
    """
    from pattern_learner.core import SCORE_PARAMS
    from pattern_learner.core import PatternLearner

    # 从当前参数复制
    params = dict(SCORE_PARAMS)
    tune_config = get_tune_config()

    # 替换为试验参数
    for pname, pcfg in tune_config.items():
        if pname in params:
            if pcfg['type'] == 'float':
                params[pname] = trial.suggest_float(pname, pcfg['low'], pcfg['high'], step=pcfg.get('step'))
            elif pcfg['type'] == 'int':
                params[pname] = trial.suggest_int(pname, pcfg['low'], pcfg['high'], step=pcfg.get('step', 1))

    # 写入全局 SCORE_PARAMS（backtest 内部读取它）
    # 用 try/finally 保证本次 trial 结束后恢复原始参数，避免全局态污染
    import pattern_learner.core as core_mod
    original_params = dict(core_mod.SCORE_PARAMS)
    try:
        for k, v in params.items():
            core_mod.SCORE_PARAMS[k] = v

        # 加载数据
        df = load_data_for_type(lottery_type)
        if df is None or len(df) < 30:
            return 0.0

        # 运行回测
        learner = PatternLearner(lottery_type)
        result = learner.backtest(df, test_periods=backtest_n)

        if result is None:
            return 0.0

        # 目标值：平均命中 + 平均Top10命中(折半) → 平衡命中数量和深度
        score = result['avg_hit'] + result.get('avg_top10_hit', 0) * 0.5
        return score
    finally:
        # 恢复原始全局参数，确保各 trial 之间互不干扰
        core_mod.SCORE_PARAMS.clear()
        core_mod.SCORE_PARAMS.update(original_params)


def tune(trials=DEFAULT_TRIALS, lottery_type='dlt', backtest_n=20):
    """执行自动调参"""
    import optuna
    from pattern_learner.core import SCORE_PARAMS

    print(f"\n{'='*55}")
    print(f"  🎯 自动调参 · {lottery_type.upper()}")
    print(f"  {trials} 次试验 · 回测 {backtest_n} 期")
    print(f"{'='*55}\n")

    # 记录初始参数
    print(f"初始参数:")
    for k, v in SCORE_PARAMS.items():
        cfgs = get_tune_config()
        tunable = '(可调)' if k in cfgs else '(固定)'
        print(f"  {k:24s} = {v!r:8s}  {tunable}")

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=10),
    )

    study.optimize(
        lambda t: objective(t, lottery_type=lottery_type, backtest_n=backtest_n),
        n_trials=trials,
        callbacks=[_progress_callback],
        show_progress_bar=True,
    )

    print(f"\n{'='*55}")
    print(f"  ✅ 调参完成")
    print(f"  最佳试验: #{study.best_trial.number}")
    print(f"  最佳评分: {study.best_trial.value:.4f}  (validation 窗口, 存在乐观偏差)")
    print(f"{'='*55}")

    # 显示最佳参数
    print(f"\n最佳参数:")
    for k, v in sorted(study.best_trial.params.items()):
        print(f"  {k:24s} = {v!r}")

    # 用独立窗口评估泛化能力（缓解同窗口既选参又打分的过拟合）
    gen_score = evaluate_generalization(study, lottery_type=lottery_type, backtest_n=backtest_n)
    # 挂到 study 上，由调用方(CLI)保存时统一写入，避免 tune 内部重复写文件
    try:
        study.set_user_attr('holdout_score', gen_score)
    except Exception:
        pass

    return study


def evaluate_generalization(study, lottery_type='dlt', backtest_n=20):
    """
    用一段独立(非调参)窗口评估最佳参数的泛化能力。

    调参时 objective 在 df 最后 backtest_n 期上既优化又打分，best_score
    因此偏乐观。这里把最新 backtest_n 期(调参窗口)从数据中剔除，改在其
    紧邻的更早一段独立窗口上回测，得到一个未参与选参的泛化评分。
    受限于历史数据量(约100期)，无法取"更靠后"的窗口，故用相邻的
    非重叠窗口作为留出集；该指标仅供参考，用于判断 best_score 的可信度。
    """
    from pattern_learner.core import PatternLearner
    import pattern_learner.core as core_mod

    df = load_data_for_type(lottery_type)
    if df is None or len(df) < backtest_n * 2 + 30:
        print(f"\n  ℹ️ 数据不足，跳过独立窗口泛化评估")
        return None

    # 应用最佳参数（临时写入全局，评估后恢复）
    original_params = dict(core_mod.SCORE_PARAMS)
    try:
        for k, v in study.best_trial.params.items():
            if k in core_mod.SCORE_PARAMS:
                core_mod.SCORE_PARAMS[k] = v

        # 剔除最新的调参窗口，在紧邻的更早一段独立窗口上回测
        holdout_df = df.iloc[:len(df) - backtest_n]
        learner = PatternLearner(lottery_type)
        result = learner.backtest(holdout_df, test_periods=backtest_n)
    finally:
        core_mod.SCORE_PARAMS.clear()
        core_mod.SCORE_PARAMS.update(original_params)

    if result is None:
        print(f"\n  ℹ️ 独立窗口回测无有效结果")
        return None

    gen_score = result['avg_hit'] + result.get('avg_top10_hit', 0) * 0.5
    print(f"\n{'='*55}")
    print(f"  🧪 独立窗口泛化评估 (留出集, 未参与选参)")
    print(f"  泛化评分: {gen_score:.4f}   validation评分: {study.best_trial.value:.4f}")
    if gen_score < study.best_trial.value:
        print(f"  ⚠️ 泛化评分低于 validation 评分，best_score 存在过拟合")
    print(f"{'='*55}")
    return gen_score


def _progress_callback(study, trial):
    """进度回调"""
    if trial.number % 10 == 0 and trial.number > 0:
        n_trials = getattr(study, 'n_trials', study.trials[-1].number + 1) if study.trials else '?'
        print(f"  ⏳ 已完成 {trial.number}/{n_trials} 试验, 当前最佳: {study.best_value:.4f}")


def save_best_params(study, lottery_type='dlt', holdout_score=None):
    """将最佳参数保存到 models/best_params.json

    说明: 仅写入 JSON 产物，不修改 core.py 源码。运行时由
    apply_best_params() 读取该 JSON 覆盖 core.SCORE_PARAMS。
    holdout_score: 独立留出窗口的泛化评分（若有）。调参窗口 validation 分
    存在乐观偏差，报告时以 holdout 分为最终泛化参考。
    """
    from pattern_learner.core import SCORE_PARAMS

    best_params = dict(SCORE_PARAMS)
    # 应用试验找到的最佳值
    for k, v in study.best_trial.params.items():
        if k in best_params:
            best_params[k] = v

    # 保存到 JSON
    save_path = Path(__file__).resolve().parent / 'models' / 'best_params.json'
    os.makedirs(str(save_path.parent), exist_ok=True)

    existing = {}
    if save_path.exists():
        with open(str(save_path), 'r') as f:
            existing = json.load(f)

    existing[lottery_type] = {
        'params': best_params,
        'best_score': study.best_trial.value,
        'holdout_score': holdout_score,
        'timestamp': datetime.now().isoformat(),
    }

    with open(str(save_path), 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 最佳参数已保存到 {save_path}")
    if holdout_score is not None:
        print(f"   留出集泛化评分: {holdout_score:.4f} (validation: {study.best_trial.value:.4f})")
        if holdout_score < study.best_trial.value:
            print(f"   ⚠️ 泛化评分低于调参窗口，参数存在过拟合，谨慎采用")
    return best_params


def load_best_params(lottery_type='dlt'):
    """加载指定彩种的最佳参数（如果存在）"""
    save_path = Path(__file__).resolve().parent / 'models' / 'best_params.json'
    if not save_path.exists():
        print(f"  ℹ️ 未找到已保存的最佳参数 ({save_path})")
        return None

    with open(str(save_path), 'r', encoding='utf-8') as f:
        all_params = json.load(f)

    if lottery_type not in all_params:
        print(f"  ℹ️ 未找到 {lottery_type} 的最佳参数")
        return None

    info = all_params[lottery_type]
    print(f"  📊 {lottery_type} 最佳参数 (评分={info['best_score']:.4f}, {info['timestamp']})")
    return info['params']


def show_best():
    """显示已保存的所有最佳参数"""
    save_path = Path(__file__).resolve().parent / 'models' / 'best_params.json'
    if not save_path.exists():
        print("❌ 未找到最佳参数文件 (models/best_params.json)")
        return

    with open(str(save_path), 'r', encoding='utf-8') as f:
        all_params = json.load(f)

    print(f"\n{'='*55}")
    print(f"  📊 已保存的最佳参数")
    print(f"{'='*55}")
    for lt, info in all_params.items():
        print(f"\n  [{lt.upper()}]  评分={info['best_score']:.4f}  ({info['timestamp']})")
        for k, v in info['params'].items():
            print(f"    {k:24s} = {v!r}")


def update_core_params(params):
    """将最佳参数写入 core.py 的 SCORE_PARAMS"""
    from pattern_learner import core as core_mod

    for k, v in params.items():
        if k in core_mod.SCORE_PARAMS:
            core_mod.SCORE_PARAMS[k] = v

    print(f"✅ 已更新 core.py SCORE_PARAMS ({len(params)} 项)")
    return core_mod.SCORE_PARAMS


def apply_best_params(lottery_type='dlt', verbose=True):
    """
    供 run_all.py / predict.py 调用：读取 models/best_params.json 中对应彩种的
    最佳参数并覆盖 core.SCORE_PARAMS。若无产物则保持默认参数不变。

    返回: True 表示成功应用了调参产物，False 表示未找到(使用默认参数)。
    """
    save_path = Path(__file__).resolve().parent / 'models' / 'best_params.json'
    if not save_path.exists():
        return False

    try:
        with open(str(save_path), 'r', encoding='utf-8') as f:
            all_params = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    if lottery_type not in all_params:
        return False

    params = all_params[lottery_type].get('params', {})
    if not params:
        return False

    from pattern_learner import core as core_mod
    applied = 0
    for k, v in params.items():
        if k in core_mod.SCORE_PARAMS:
            core_mod.SCORE_PARAMS[k] = v
            applied += 1

    if verbose:
        info = all_params[lottery_type]
        print(f"  📊 已应用 {lottery_type} 调参产物 "
              f"(评分={info.get('best_score', 0):.4f}, {applied} 项参数)")
    return applied > 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='P4: 自动调参 + 集成优化')
    parser.add_argument('--type', choices=['dlt', 'dlt_back', 'ssq', 'ssq_back', 'kl8'],
                        default='dlt', help='彩种类型')
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS, help='试验次数')
    parser.add_argument('--backtest', type=int, default=20, help='回测期数')
    parser.add_argument('--show-best', action='store_true', help='显示已保存的最佳参数')
    parser.add_argument('--load-best', action='store_true', help='加载最佳参数到 core.py（不调参）')
    parser.add_argument('--save', action='store_true', default=True, help='调参后保存最佳参数(默认开启)')
    parser.add_argument('--no-save', dest='save', action='store_false', help='调参后不保存最佳参数')
    args = parser.parse_args()

    if args.show_best:
        show_best()
        sys.exit(0)

    if args.load_best:
        params = load_best_params(args.type)
        if params:
            update_core_params(params)
        else:
            print(f"没有 {args.type} 的最佳参数，请先运行调参")
        sys.exit(0)

    # 自动调参
    study = tune(
        trials=args.trials,
        lottery_type=args.type,
        backtest_n=args.backtest,
    )

    if args.save:
        gen = None
        try:
            gen = study.user_attrs.get('holdout_score')
        except Exception:
            pass
        save_best_params(study, lottery_type=args.type, holdout_score=gen)
