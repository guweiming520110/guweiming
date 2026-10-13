#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
方向2（集合预测器）+ 方向3（特征校准）回归测试。
运行: pytest tests/test_ml_set.py -v
"""
import numpy as np
import pandas as pd
import pytest


# ==================== SetNumberPredictor（方向2）====================

def test_set_predictor_dlt_valid_numbers(dlt_df):
    """集合预测：输出 k 个互不重复、在 1-35 范围内的号码。"""
    from ml_models import SetNumberPredictor
    sp = SetNumberPredictor('dlt')
    sp.train(dlt_df)
    preds = sp.predict_topk()
    assert len(preds) == 5
    assert len(set(preds)) == 5
    assert all(1 <= n <= 35 for n in preds)


def test_set_predictor_kl8_valid_numbers(kl8_df):
    """集合预测：kl8 输出 20 个在 1-80 范围内不重复号码。"""
    from ml_models import SetNumberPredictor
    sp = SetNumberPredictor('kl8')
    sp.train(kl8_df)
    preds = sp.predict_topk()
    assert len(preds) == 20
    assert len(set(preds)) == 20
    assert all(1 <= n <= 80 for n in preds)


def test_set_predictor_prob_map(dlt_df):
    """predict_proba_map 返回 35 个号码的概率，和为任意(非归一化)但值在[0,1]。"""
    from ml_models import SetNumberPredictor
    sp = SetNumberPredictor('dlt')
    sp.train(dlt_df)
    probs = sp.predict_proba_map()
    assert len(probs) == 35
    assert all(0.0 <= v <= 1.0 for v in probs.values())


def test_set_predictor_save_load_roundtrip(dlt_df, tmp_path):
    """集合预测器保存/加载往返，set_latest 后仍可预测。"""
    from ml_models import SetNumberPredictor
    sp = SetNumberPredictor('dlt')
    sp.train(dlt_df)
    sp.save_models(str(tmp_path))
    sp2 = SetNumberPredictor('dlt')
    sp2.load_models(str(tmp_path))
    sp2.set_latest(dlt_df)
    preds = sp2.predict_topk()
    assert len(preds) == 5
    assert all(1 <= n <= 35 for n in preds)


# ==================== 特征校准（方向3）====================

def test_feature_count_matches_config(dlt_df):
    """M2: dlt 实际生成特征数 = 配置特征数（76），无白算 weekday/month 列。"""
    from feature_engineering import FeatureEngineer
    from feature_config import get_all_features
    fe = FeatureEngineer(dlt_df, lottery_type='dlt')
    df = fe.extract_all_features()
    configured = set(get_all_features())
    generated = set(df.columns)
    # 所有配置特征都应生成
    assert configured <= generated
    # 不应有白算的原始 weekday/month 列
    assert 'weekday' not in generated
    assert 'month' not in generated


def test_dlt_score_params_no_rarity():
    """方向3: dlt/ssq 校准后关闭稀有度（消融实验为负贡献）。"""
    from pattern_learner.core import _base_score_params
    p = _base_score_params('dlt')
    assert p['rarity_min'] == 0.0
    assert p['rarity_max'] == 0.0
    p_kl8 = _base_score_params('kl8')
    assert p_kl8['rarity_min'] > 0.0  # kl8 保留稀有度（正贡献）


def test_pattern_predict_within_range(dlt_df):
    """图案引擎预测号码在合法范围且可跑通。"""
    import io
    import contextlib
    from pattern_learner.core import PatternLearner
    pl = PatternLearner('dlt')
    with contextlib.redirect_stdout(io.StringIO()):
        pl.train(dlt_df, num_periods=50)
    preds = pl.predict(top_k=10)
    assert len(preds) == 10
    assert all(1 <= n <= 35 for n, _, _ in preds)
