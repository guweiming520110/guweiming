#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一数字彩票走势图生成器 — 入口

配置驱动，支持所有 LOTTERY_TYPES 中定义的彩种

用法:
  python generate_charts.py                           # 大乐透走势图
  python generate_charts.py --kl8                     # 快乐8走势图
  python generate_charts.py --type ssq                # 双色球走势图
  python generate_charts.py --type ssq --periods 100  # 双色球 100期
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from lottery_types import get_config, LOTTERY_TYPES
from charts.composite import generate_composite_chart


# ==================== 全局配置 ====================

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DB_PATH = ROOT_DIR / 'lottery.db'
OUTPUT_DIR = BASE_DIR / 'charts'

try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


# ==================== 数据加载 ====================

def load_data(lottery_type='dlt', db_path=None, limit=None):
    """加载指定彩种的历史数据"""
    if db_path is None:
        db_path = DB_PATH
    cfg = get_config(lottery_type)
    data_table = cfg['data_table']
    data_type = cfg['data_type']

    if not os.path.exists(str(db_path)):
        print(f"数据库不存在: {db_path}")
        for csv_name in ['lottery_data_dlt.csv', 'lottery_data.csv']:
            csv_path = BASE_DIR / csv_name
            if csv_path.exists():
                df = pd.read_csv(str(csv_path))
                if len(df) > 0:
                    print(f"从CSV加载了 {len(df)} 条记录 ({csv_name})")
                    return df
        return None

    conn = sqlite3.connect(str(db_path))
    try:
        query = f"SELECT * FROM {data_table} WHERE lottery_type='{data_type}' ORDER BY period ASC"
        df = pd.read_sql(query, conn)
        print(f"从数据库加载了 {len(df)} 条 {cfg['name']} 历史记录")
    except Exception as e:
        print(f"数据库查询失败: {e}")
        df = None
    finally:
        conn.close()

    if df is not None and limit and len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    return df


def load_kl8_data(db_path=None, limit=None):
    """加载快乐8数据（kl8_records 表，与 lottery_records 结构不同）"""
    if db_path is None:
        db_path = DB_PATH
    if not os.path.exists(str(db_path)):
        return None

    conn = sqlite3.connect(str(db_path))
    try:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='kl8_records'", conn
        )
        if len(tables) == 0:
            conn.close()
            return None

        df = pd.read_sql("SELECT * FROM kl8_records ORDER BY period ASC", conn)
        if len(df) == 0:
            conn.close()
            return None

        all_nums = []
        for nums_str in df['numbers']:
            nums = [int(x) for x in str(nums_str).split(',') if x.strip().isdigit()]
            all_nums.append(nums)
        df['num_list'] = all_nums
        conn.close()

        if limit and len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)
        print(f"从数据库加载了 {len(df)} 条快乐8历史记录")
        return df
    except Exception as e:
        print(f"加载快乐8数据失败: {e}")
        conn.close()
        return None


# ==================== 主要入口 ====================

def generate_all_charts(num_periods=50, output_dir=None, kl8=False,
                        predict_front=None, predict_back=None,
                        verify=False, lottery_type='dlt'):
    """统一入口"""
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    os.makedirs(str(output_dir), exist_ok=True)

    cfg = get_config(lottery_type)

    if kl8:
        lottery_type = 'kl8'
        cfg = get_config('kl8')
        from kl8_data import ensure_kl8_data
        df = ensure_kl8_data(num_periods=num_periods)
        if df is None or len(df) == 0:
            print("\n无法加载快乐8数据！")
            print("请先运行: python kl8_data.py")
            return

        # 验证模式：若未指定预测号码，从数据库自动加载上次预测
        target_period = None  # 预测对应的目标期号
        if verify and not predict_front:
            try:
                conn_v = sqlite3.connect(str(DB_PATH))
                cur_v = conn_v.execute(
                    """SELECT target_period, front_numbers FROM prediction_records
                       WHERE lottery_type='kl8' AND (actual_front IS NULL OR actual_front = '')
                       ORDER BY id DESC LIMIT 1""",
                )
                row = cur_v.fetchone()
                conn_v.close()
                if row:
                    target_period = row[0]
                    predict_front = [int(x) for x in row[1].split(',') if x.strip().isdigit()]
                    print(f"  📋 从数据库加载预测(期{target_period}): {predict_front}")
                else:
                    # 尝试加载已验证的最近记录（重新生成验证走势图）
                    conn_v = sqlite3.connect(str(DB_PATH))
                    cur_v = conn_v.execute(
                        """SELECT target_period, front_numbers FROM prediction_records
                           WHERE lottery_type='kl8' AND actual_front != ''
                           ORDER BY id DESC LIMIT 1""",
                    )
                    row = cur_v.fetchone()
                    conn_v.close()
                    if row:
                        target_period = row[0]
                        predict_front = [int(x) for x in row[1].split(',') if x.strip().isdigit()]
                        print(f"  📋 从数据库加载已验证预测(期{target_period}): {predict_front}")
            except Exception as e:
                print(f"  ⚠️ 加载预测记录失败: {e}")

        # 验证模式：检查预测命中，并保存验证结果到数据库
        verified_front_set = None  # None=未验证, set()=已验证无命中, {1,2}=已验证有命中
        if verify and predict_front and len(df) > 0:
            # ⚠️ 关键修复：必须用"目标期号"对应的开奖号码验证，
            # 绝不能盲取 df.iloc[-1]（数据库最新一期）——否则会出现
            # "用上一期/相邻期开奖号验证本期预测"的期号错位 bug。
            actual_row = None
            if target_period is not None:
                matched = df[df['period'].astype(str) == str(target_period)]
                if len(matched) > 0:
                    actual_row = matched.iloc[-1]

            if actual_row is None:
                # 目标期尚未开奖（数据落后）或库中无该期数据 → 跳过验证，不盲配
                print(f"  ⏳ 目标期 {target_period} 尚未开奖或数据库无该期开奖数据，跳过验证（不做盲配）")
                verified_front_set = None
            else:
                actual_front = actual_row['num_list']
                verified_front_set = set(predict_front) & set(actual_front)
                if verified_front_set:
                    print(f"  ✅ 预测正确号码: {sorted(verified_front_set)} ({len(verified_front_set)}/{len(predict_front)})")
                else:
                    print(f"  本期预测未命中")

            # 保存验证结果到数据库，使历史期号行也能显示黄色标记
            if target_period and actual_row is not None:
                try:
                    from notifier import save_verification, send_verification
                    front_str = ','.join(str(n) for n in predict_front)
                    save_verification(
                        'kl8', target_period,
                        front_str, '',
                        actual_front, [],
                        verified_front_set, set()
                    )
                    print(f"  💾 验证结果已保存到数据库(期{target_period})")
                    # 推送钉钉通知
                    try:
                        send_verification('kl8', predict_front, [],
                                          verified_front_set, set(),
                                          actual_front, [])
                        print(f"  ✅ 钉钉验证推送成功")
                    except Exception as e:
                        print(f"  ⚠️ 钉钉推送异常: {e}")
                except Exception as e:
                    print(f"  ⚠️ 保存验证结果失败: {e}")

        print("=" * 50)
        print(f"  {cfg['name']} · 综合走势图生成器")
        print("=" * 50)

        if predict_front:
            print(f"  🔮 预测号码: {predict_front}")

        # 验证模式下不传 predicted_front：已验证的预测由 history_preds
        # 在对应的历史期号行上显示（黄色=命中，红色=未命中）
        # 未来行留给新的预测（如果有的话）
        chart_pred_front = None if verify else predict_front

        generate_composite_chart(
            df, lottery_type='kl8',
            network_periods=min(20, len(df)),
            trend_periods=min(30, len(df)),
            freq_periods=min(100, len(df)),
            output_dir=output_dir,
            predicted_front=chart_pred_front,
            # 不传 verified_front：验证结果由 history_preds（数据库）在历史期号行显示
        )
    else:
        # 加载数据
        if lottery_type == 'kl8':
            df = load_kl8_data(limit=num_periods)
        else:
            df = load_data(lottery_type=lottery_type, limit=num_periods)

        if df is None or len(df) == 0:
            print(f"\n无法加载{cfg['name']}数据！")
            return

        # 验证模式：若未指定预测号码，从数据库自动加载上次预测
        target_period = None  # 预测对应的目标期号（用于期号精确校验）
        if verify and not predict_front:
            try:
                conn_v = sqlite3.connect(str(DB_PATH))
                cur_v = conn_v.execute(
                    """SELECT target_period, front_numbers, back_numbers FROM prediction_records
                       WHERE lottery_type=? AND (actual_front IS NULL OR actual_front = '')
                       ORDER BY id DESC LIMIT 1""",
                    (lottery_type,)
                )
                row = cur_v.fetchone()
                conn_v.close()
                if row:
                    target_period = row[0]
                    predict_front = [int(x) for x in row[1].split(',') if x.strip().isdigit()]
                    if row[2]:
                        predict_back = [int(x) for x in row[2].split(',') if x.strip().isdigit()]
                    print(f"  📋 从数据库加载预测(期{target_period}): 前区{predict_front} 后区{predict_back}")
                else:
                    # 尝试加载已验证的最近记录（重新生成验证走势图）
                    conn_v = sqlite3.connect(str(DB_PATH))
                    cur_v = conn_v.execute(
                        """SELECT target_period, front_numbers, back_numbers FROM prediction_records
                           WHERE lottery_type=? AND actual_front != ''
                           ORDER BY id DESC LIMIT 1""",
                        (lottery_type,)
                    )
                    row = cur_v.fetchone()
                    conn_v.close()
                    if row:
                        target_period = row[0]
                        predict_front = [int(x) for x in row[1].split(',') if x.strip().isdigit()]
                        if row[2]:
                            predict_back = [int(x) for x in row[2].split(',') if x.strip().isdigit()]
                        print(f"  📋 从数据库加载已验证预测(期{target_period}): 前区{predict_front} 后区{predict_back}")
            except Exception as e:
                print(f"  ⚠️ 加载预测记录失败: {e}")

        # 验证模式：检查预测命中
        verified_front_set, verified_back_set = None, None  # None=未验证
        if verify and predict_front:
            front_cnt = cfg['front_count']
            back_cnt = cfg['back_count']
            # ⚠️ 关键修复：用"目标期号"对应的开奖行验证，禁止盲取 df.iloc[-1]
            latest = None
            if target_period is not None:
                matched = df[df['period'].astype(str) == str(target_period)]
                if len(matched) > 0:
                    latest = matched.iloc[-1]

            if latest is None:
                print(f"  ⏳ 目标期 {target_period} 尚未开奖或数据库无该期开奖数据，跳过验证（不做盲配）")
            elif lottery_type == 'kl8':
                # 快乐8验证逻辑
                actual_front = latest['num_list']
                verified_front_set = set(predict_front) & set(actual_front)
                if verified_front_set:
                    print(f"  ✅ 预测正确号码: {sorted(verified_front_set)} ({len(verified_front_set)}/{len(predict_front)})")
                else:
                    print(f"  本期预测未命中")
            else:
                actual_front = [int(latest[f'front_{k}']) for k in range(1, front_cnt + 1)]
                actual_back = [int(latest[f'back_{k}']) for k in range(1, back_cnt + 1)]
                verified_front_set = set(predict_front) & set(actual_front)
                verified_back_set = set(predict_back or []) & set(actual_back)
                if verified_front_set or verified_back_set:
                    print(f"  ✅ 预测正确{cfg['front_name']}: {sorted(verified_front_set)}")
                    print(f"  ✅ 预测正确{cfg['back_name']}: {sorted(verified_back_set)}")
                else:
                    print(f"  本期预测未命中")

        print("=" * 50)
        print(f"  {cfg['name']} · 综合走势图生成器")
        print("=" * 50)

        if predict_front:
            name_key = cfg['front_name']
            print(f"  🔮 预测{name_key}: {predict_front}")
        if predict_back:
            name_key = cfg['back_name']
            print(f"  🔮 预测{name_key}: {predict_back}")

        # 计算预测理由
        pred_reasons = None
        if predict_front:
            from pattern_learner.core import PatternLearner
            try:
                pl = PatternLearner(lottery_type)
                pl.train(df, num_periods)
                scored = pl.score_candidates(predict_front)
                pred_reasons = {}
                for num, sc, detail in scored:
                    pred_reasons[str(num)] = {
                        'score': sc,
                        'patterns': {k: v for k, v in detail.items()}
                    }
            except Exception as e:
                print(f"  预测理由计算跳过: {e}")

        generate_composite_chart(
            df, lottery_type=lottery_type,
            network_periods=min(30, len(df)),
            trend_periods=num_periods,
            freq_periods=min(100, len(df)),
            output_dir=output_dir,
            predicted_front=predict_front,
            predicted_back=predict_back,
            verified_front=list(verified_front_set) if verify else None,
            verified_back=list(verified_back_set) if verify else None,
            predicted_reasons=pred_reasons,
        )

        # 验证模式钉钉推送
        # 仅当确实完成了验证（verified_front_set 不为 None，说明目标期已开奖并成功匹配）才推送/保存
        if (verify and predict_front and lottery_type != 'kl8'
                and verified_front_set is not None):
            try:
                from notifier import save_verification, send_verification

                # 保存验证结果到数据库（使用目标期号，禁止 df.iloc[-1] 盲配）
                front_str = ','.join(str(n) for n in predict_front)
                back_str = ','.join(str(n) for n in (predict_back or []))
                save_verification(
                    lottery_type, str(target_period),
                    front_str, back_str,
                    actual_front, actual_back,
                    verified_front_set, verified_back_set
                )

                # 发送钉钉通知
                ok = send_verification(
                    lottery_type,
                    predict_front, predict_back or [],
                    verified_front_set, verified_back_set,
                    actual_front, actual_back
                )
                print(f"\n{'✅' if ok else '❌'} 钉钉验证推送: {'成功' if ok else '失败'}")
            except Exception as e:
                print(f"\n⚠️ 钉钉推送异常: {e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='彩票综合走势图生成器')
    parser.add_argument('--periods', type=int, default=50,
                       help='期数 (默认: 50)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录 (默认: ./charts)')
    parser.add_argument('--kl8', action='store_true',
                       help='生成快乐8走势图')
    parser.add_argument('--type', choices=list(LOTTERY_TYPES.keys()), default='dlt',
                       help='彩种类型 (默认: dlt)')
    parser.add_argument('--predict-front', type=str, default=None,
                       help='预测前区号码，逗号分隔')
    parser.add_argument('--predict-back', type=str, default=None,
                       help='预测后区号码，逗号分隔')
    parser.add_argument('--verify', action='store_true',
                       help='验证模式')
    args = parser.parse_args()

    pf = [int(x.strip()) for x in args.predict_front.split(',')] if args.predict_front else None
    pb = [int(x.strip()) for x in args.predict_back.split(',')] if args.predict_back else None

    generate_all_charts(
        num_periods=args.periods, output_dir=args.output, kl8=args.kl8,
        predict_front=pf, predict_back=pb,
        verify=args.verify,
        lottery_type=args.type if not args.kl8 else 'kl8'
    )
