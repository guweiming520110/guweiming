#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键全自动运行
  - 清空旧预测记录
  - 更新三大彩种历史数据
  - 运行预测分析（含回测）
  - 生成含预测的走势图
  - 钉钉推送预测结果

用法:
  python run_all.py                          # 完整运行（含钉钉推送）
  python run_all.py --no-notify              # 运行但不推送到钉钉
  python run_all.py --skip-charts            # 跳过走势图生成
"""

import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DB_PATH = ROOT_DIR / 'lottery.db'

# 确保路径正确
os.chdir(str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(ROOT_DIR))


def step_header(num, total, icon, title):
    print(f"\n{'█' * 55}")
    print(f"  Step {num}/{total}: {icon} {title}")
    print(f"{'█' * 55}")


def main():
    import argparse
    import sqlite3
    from datetime import datetime

    parser = argparse.ArgumentParser(description='一键全自动运行')
    parser.add_argument('--no-notify', action='store_true', help='不推送钉钉通知')
    parser.add_argument('--skip-charts', action='store_true', help='跳过走势图生成')
    args = parser.parse_args()

    no_notify = args.no_notify

    print(f"{'█' * 55}")
    print(f"  🚀 数字彩票AI分析系统 · 一键全自动运行")
    print(f"  📂 {BASE_DIR}")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'█' * 55}")

    TOTAL_STEPS = 6

    # ──────────────────────────────
    # Step 1: 清理未验证的旧预测记录（保留已验证的历史）
    # ──────────────────────────────
    step_header(1, TOTAL_STEPS, '🧹', '清理未验证的旧预测记录')
    conn = sqlite3.connect(str(DB_PATH))
    # 只删除尚未验证的记录（actual_front为空），保留已验证的历史记录
    cur = conn.execute(
        "DELETE FROM prediction_records WHERE actual_front IS NULL OR actual_front = ''"
    )
    deleted = cur.rowcount
    # 统计保留的已验证记录
    cur = conn.execute(
        "SELECT COUNT(*) FROM prediction_records WHERE actual_front IS NOT NULL AND actual_front != ''"
    )
    kept = cur.fetchone()[0]
    conn.commit()
    conn.close()
    print(f"  已清理 {deleted} 条未验证记录，保留 {kept} 条已验证历史 ✅")

    # ──────────────────────────────
    # Step 2: 更新大乐透数据
    # ──────────────────────────────
    step_header(2, TOTAL_STEPS, '📥', '更新大乐透数据')
    try:
        import fill_dlt_data
        fill_dlt_data.fill_missing_data()
    except Exception as e:
        print(f"  ⚠️ 大乐透数据更新异常: {e}")

    # ──────────────────────────────
    # Step 3: 更新双色球数据
    # ──────────────────────────────
    step_header(3, TOTAL_STEPS, '📥', '更新双色球数据')
    try:
        import fill_ssq_data
        fill_ssq_data.fill_ssq_data()
    except Exception as e:
        print(f"  ⚠️ 双色球数据更新异常: {e}")

    # ──────────────────────────────
    # Step 4: 更新快乐8数据
    # ──────────────────────────────
    step_header(4, TOTAL_STEPS, '📥', '更新快乐8数据')
    try:
        from kl8_data import ensure_kl8_data
        df = ensure_kl8_data(num_periods=100, force_refresh=True)
        print(f"  快乐8: {len(df) if df is not None else 0} 期 ✅")
    except Exception as e:
        print(f"  ⚠️ 快乐8数据更新异常: {e}")

    # ──────────────────────────────
    # Step 5: 运行三大彩种预测
    # ──────────────────────────────
    step_header(5, TOTAL_STEPS, '🎯', '运行预测分析')
    from generate_charts import load_data
    import predict as pred_mod

    # 存储预测结果用于走势图
    chart_configs = []

    # --- 5a: 大乐透 ---
    print(f"\n{'=' * 40}")
    print(f"  超级大乐透预测")
    print(f"{'=' * 40}")
    try:
        df = load_data()
        if df is not None:
            result = pred_mod.predict_dlt(df, num_periods=50, backtest_n=5, use_ml=True)
            if result:
                front_preds, back_preds, front_report, back_report = result
                front_top5 = [str(n) for n, s, d in front_preds[:5]]
                back_top2 = [str(n) for n, s, d in back_preds[:2]]
                chart_configs.append({
                    'type': 'dlt',
                    'front': ','.join(front_top5),
                    'back': ','.join(back_top2),
                })
                # 钉钉推送
                if not no_notify:
                    from notifier import save_prediction, send_prediction as send_pred
                    latest_period = str(int(df.iloc[-1]['period']) + 1)
                    front_nums = [n for n, s, d in front_preds[:5]]
                    back_nums = [n for n, s, d in back_preds[:2]]
                    save_prediction('dlt', latest_period, front_nums, back_nums)
                    ok = send_pred('dlt', front_preds, back_preds, front_report, back_report, df)
                    print(f"  {'✅' if ok else '❌'} 钉钉推送: {'成功' if ok else '失败'}")
    except Exception as e:
        import traceback
        print(f"  ⚠️ 大乐透预测异常: {e}")
        traceback.print_exc()

    # --- 5b: 双色球 ---
    print(f"\n{'=' * 40}")
    print(f"  双色球预测")
    print(f"{'=' * 40}")
    try:
        df_ssq = load_data(lottery_type='ssq', limit=50)
        if df_ssq is not None:
            result = pred_mod.predict_ssq(df_ssq, num_periods=50, backtest_n=5)
            if result:
                front_preds, back_preds, front_report, back_report = result
                front_top6 = [str(n) for n, s, d in front_preds[:6]]
                back_top1 = [str(n) for n, s, d in back_preds[:1]]
                chart_configs.append({
                    'type': 'ssq',
                    'front': ','.join(front_top6),
                    'back': ','.join(back_top1),
                })
                if not no_notify:
                    from notifier import save_prediction, send_prediction as send_pred
                    latest_period = str(int(df_ssq.iloc[-1]['period']) + 1)
                    front_nums = [n for n, s, d in front_preds[:6]]
                    back_nums = [n for n, s, d in back_preds[:1]]
                    save_prediction('ssq', latest_period, front_nums, back_nums)
                    ok = send_pred('ssq', front_preds, back_preds, front_report, back_report, df_ssq)
                    print(f"  {'✅' if ok else '❌'} 钉钉推送: {'成功' if ok else '失败'}")
    except Exception as e:
        import traceback
        print(f"  ⚠️ 双色球预测异常: {e}")
        traceback.print_exc()

    # --- 5c: 快乐8 ---
    print(f"\n{'=' * 40}")
    print(f"  快乐8预测")
    print(f"{'=' * 40}")
    try:
        from kl8_data import ensure_kl8_data
        df_kl8 = ensure_kl8_data(num_periods=30)
        if df_kl8 is not None:
            result = pred_mod.predict_kl8(df_kl8, num_periods=30)
            if result:
                preds, report = result
                pred_top20 = [str(n) for n, s, d in preds[:20]]
                chart_configs.append({
                    'type': 'kl8',
                    'front': ','.join(pred_top20),
                    'back': '',
                })
                if not no_notify:
                    from notifier import save_prediction, send_prediction as send_pred
                    latest_period = str(int(df_kl8.iloc[-1]['period']) + 1)
                    save_prediction('kl8', latest_period, [n for n, s, d in preds[:20]])
                    ok = send_pred('kl8', preds, None, report, '', df_kl8)
                    print(f"  {'✅' if ok else '❌'} 钉钉推送: {'成功' if ok else '失败'}")
    except Exception as e:
        import traceback
        print(f"  ⚠️ 快乐8预测异常: {e}")
        traceback.print_exc()

    # ──────────────────────────────
    # Step 6: 生成走势图
    # ──────────────────────────────
    if args.skip_charts:
        step_header(6, TOTAL_STEPS, '⏭️', '走势图生成已跳过')
    else:
        step_header(6, TOTAL_STEPS, '📊', '生成含预测走势图')
        from charts.composite import generate_composite_chart

        for cfg in chart_configs:
            lt = cfg['type']
            pf = [int(x) for x in cfg['front'].split(',') if x.strip()]
            pb = [int(x) for x in cfg['back'].split(',') if x.strip()] if cfg['back'] else None
            names = {'dlt': '大乐透', 'ssq': '双色球', 'kl8': '快乐8'}

            print(f"\n  正在生成 {names.get(lt, lt)}走势图...")
            try:
                if lt == 'kl8':
                    from kl8_data import ensure_kl8_data
                    df_c = ensure_kl8_data(num_periods=30)
                else:
                    df_c = load_data(lottery_type=lt, limit=50)

                if df_c is not None:
                    generate_composite_chart(
                        df_c, lottery_type=lt,
                        network_periods=min(30, len(df_c)),
                        trend_periods=min(50, len(df_c)),
                        freq_periods=min(100, len(df_c)),
                        predicted_front=pf,
                        predicted_back=pb,
                    )
                    print(f"  ✅ {names.get(lt, lt)}走势图生成完成")
            except Exception as e:
                import traceback
                print(f"  ⚠️ {names.get(lt, lt)}走势图异常: {e}")
                traceback.print_exc()

    # ──────────────────────────────
    # 完成
    # ──────────────────────────────
    print(f"\n{'█' * 55}")
    print(f"  ✅ 一键全自动运行完成!")
    print(f"  📂 走势图: {BASE_DIR / 'charts'}")
    print(f"{'█' * 55}")

    # 统计
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.execute('SELECT lottery_type, COUNT(*) FROM prediction_records GROUP BY lottery_type')
    rows = cur.fetchall()
    if rows:
        print(f"\n📋 本次预测记录:")
        type_names = {'dlt': '大乐透', 'ssq': '双色球', 'kl8': '快乐8'}
        for lt, cnt in rows:
            print(f"  {type_names.get(lt, lt)}: {cnt} 条")
    conn.close()
    print()


if __name__ == '__main__':
    main()
