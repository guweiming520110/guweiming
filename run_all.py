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

from lottery_types import next_period

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
    # Step 1: 自动验证未验证的旧预测记录（保留所有历史，已开奖的自动标记）
    # ──────────────────────────────
    step_header(1, TOTAL_STEPS, '🔄', '自动验证未验证的预测记录')
    conn = sqlite3.connect(str(DB_PATH))

    # 查询所有未验证的记录
    cur = conn.execute(
        "SELECT id, lottery_type, target_period, front_numbers, back_numbers "
        "FROM prediction_records "
        "WHERE actual_front IS NULL OR actual_front = ''"
    )
    pending = list(cur.fetchall())

    if pending:
        auto_verified = 0
        for row in pending:
            pid, lt, period, front_str, back_str = row
            # 兼容 front/back_numbers 为 NULL 的历史记录
            front_pred = [int(x) for x in (front_str or '').split(',') if x.strip().isdigit()]
            back_pred = [int(x) for x in (back_str or '').split(',') if x.strip().isdigit()]

            actual_found = False
            actual_front_list = []
            actual_back_list = []

            if lt == 'kl8':
                cur = conn.execute(
                    "SELECT numbers FROM kl8_records WHERE period=?", (period,)
                )
                r = cur.fetchone()
                if r:
                    actual_nums = [int(x) for x in r[0].split(',') if x.strip().isdigit()]
                    actual_front_list = list(actual_nums)
                    actual_back_list = []
                    actual_found = True
            else:
                front_cnt = 5 if lt == 'dlt' else 6
                back_cnt = 2 if lt == 'dlt' else 1
                front_cols = ','.join([f'front_{k}' for k in range(1, front_cnt+1)])
                back_cols = ','.join([f'back_{k}' for k in range(1, back_cnt+1)])
                cur = conn.execute(
                    f"SELECT {front_cols},{back_cols} FROM lottery_records "
                    f"WHERE lottery_type=? AND period=?", (lt, period)
                )
                r = cur.fetchone()
                if r:
                    # 过滤 NULL 字段，避免后续 sorted/拼接出错
                    actual_front_list = [n for n in r[:front_cnt] if n is not None]
                    actual_back_list = [n for n in r[front_cnt:] if n is not None]
                    actual_found = True

            if actual_found:
                front_match = set(front_pred) & set(actual_front_list)
                back_match = set(back_pred) & set(actual_back_list)
                kl8_combos = None
                # 计算奖级
                from prize_calculator import get_prize_level_string, get_prize, _fmt_nums
                match_level = get_prize_level_string(
                    lt, len(front_match), len(back_match)
                )
                conn.execute(
                    """UPDATE prediction_records SET
                       actual_front=?, actual_back=?,
                       is_matched=?, match_level=?,
                       front_match_count=?, back_match_count=?
                       WHERE id=?""",
                    (','.join(str(n) for n in actual_front_list),
                     ','.join(str(n) for n in actual_back_list),
                     1 if front_match or back_match else 0,
                     match_level,
                     len(front_match), len(back_match), pid)
                )
                auto_verified += 1
                # 终端显示命中详情 + 奖级
                front_name_map = {'dlt': '前区', 'ssq': '红球', 'kl8': '号码'}
                back_name_map = {'dlt': '后区', 'ssq': '蓝球'}
                f_name = front_name_map.get(lt, '前区')
                b_name = back_name_map.get(lt, '后区')

                if lt == 'kl8':
                    hit_str = f"{f_name}{len(front_match)}/20"
                    # 重建组合用于验证显示
                    from prize_calculator import generate_kl8_combinations, verify_kl8_combinations, get_kl8_sh10_prize
                    kl8_combos = generate_kl8_combinations(front_pred)
                    kl8_verified = verify_kl8_combinations(kl8_combos, actual_front_list)
                    # 汇总
                    any_win = any(v['prize'] and v['prize']['is_winning']
                                  for key in ['single', 'complex_11', 'complex_12']
                                  for v in kl8_verified[key])
                    if any_win:
                        best = max((v['prize'] for key in ['single', 'complex_11', 'complex_12']
                                    for v in kl8_verified[key]
                                    if v['prize'] and v['prize']['is_winning']),
                                   key=lambda x: x['match_count'])
                        level_str = f"🎉 {best['level']} {best['bonus']}"
                    else:
                        level_str = "未中奖"
                    print(f"   ✅ 期{period}: {hit_str} → {level_str}")
                    # 打印各组合命中
                    for key, label in [('single', '单式'), ('complex_11', '11码'), ('complex_12', '12码')]:
                        for i, v in enumerate(kl8_verified[key]):
                            tag = f"{'AB'[i]}" if key == 'single' else f"注{i+1}"
                            print(f"      {label} {tag}: 中{v['hit_count']}/10 {'🎉' if v['prize'] and v['prize']['is_winning'] else ''}")
                else:
                    hit_str = f"{f_name}{len(front_match)}/{len(front_pred)}"
                    if back_pred:
                        hit_str += f" {b_name}{len(back_match)}/{len(back_pred)}"
                    prize = get_prize(lt, len(front_match), len(back_match))
                    level_str = prize['level'] or '未中奖'
                    print(f"   ✅ 期{period}: {hit_str} → {level_str}")

                # 钉钉推送验证结果（含命中详情 + 中奖详情）
                if not no_notify:
                    try:
                        from notifier import send_verification, send_kl8_verification
                        if lt == 'kl8':
                            # kl8_combos 只有在 kl8 且 actual_found 时才会生成
                            ok = send_kl8_verification(kl8_combos, actual_front_list, period)
                        else:
                            ok = send_verification(
                                lt, front_pred, back_pred,
                                front_match, back_match,
                                actual_front_list, actual_back_list
                            )
                        if not ok:
                            print(f"     ⚠️ 期{period}: 钉钉验证推送失败（返回错误码）")
                        else:
                            print(f"     ✅ 期{period}: 钉钉验证推送成功")
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"     ⚠️ 期{period}: 钉钉验证推送异常: {e}")

        conn.commit()

        # 统计
        cur = conn.execute("SELECT COUNT(*) FROM prediction_records")
        total = cur.fetchone()[0]
        cur = conn.execute(
            "SELECT COUNT(*) FROM prediction_records "
            "WHERE actual_front IS NOT NULL AND actual_front != ''"
        )
        verified = cur.fetchone()[0]
        unverified = total - verified
        print(f"  📊 总计 {total} 条 | 已验证 {verified} 条 | 待验证 {unverified} 条")
    else:
        cur = conn.execute("SELECT COUNT(*) FROM prediction_records")
        total = cur.fetchone()[0]
        print(f"  ✅ 没有待验证的记录（共 {total} 条历史预测）")

    # ──────────────────────────────
    # 自动推送每个彩种最近一期验证结果（确保每次都能看到）
    # ──────────────────────────────
    if not no_notify:
        print(f"\n  📤 推送各彩种最近一期验证结果...")
        for lt in ['dlt', 'ssq', 'kl8']:
            # 查询最近一期已验证记录
            cur = conn.execute(
                """SELECT * FROM prediction_records
                   WHERE lottery_type=? AND actual_front IS NOT NULL AND actual_front != ''
                   ORDER BY target_period DESC LIMIT 1""",
                (lt,)
            )
            row = cur.fetchone()
            if row:
                # 列顺序: id, lottery_type, predict_time, target_period, play_type, front_numbers, back_numbers, ...
                pid = row[0]
                lt = row[1]
                target_period = row[3]
                front_str = row[5]
                back_str = row[6]
                front_pred = [int(x) for x in front_str.split(',') if x.strip().isdigit()]
                back_pred = [int(x) for x in back_str.split(',') if x.strip().isdigit()]
                # 重新查询完整数据获取 actual 字段
                cur2 = conn.execute(
                    "SELECT actual_front, actual_back FROM prediction_records WHERE id=?", (pid,)
                )
                actual_row = cur2.fetchone()
                actual_front_str = actual_row[0]
                actual_back_str = actual_row[1]
                actual_front_list = [int(x) for x in actual_front_str.split(',') if x.strip().isdigit()]
                actual_back_list = [int(x) for x in actual_back_str.split(',') if x.strip().isdigit()]
                front_match = set(front_pred) & set(actual_front_list)
                back_match = set(back_pred) & set(actual_back_list)

                from notifier import send_verification, send_kl8_verification
                type_names = {'dlt': '大乐透', 'ssq': '双色球', 'kl8': '快乐8'}
                try:
                    if lt == 'kl8':
                        from prize_calculator import generate_kl8_combinations
                        kl8_combos = generate_kl8_combinations(front_pred)
                        ok = send_kl8_verification(kl8_combos, actual_front_list, target_period)
                    else:
                        ok = send_verification(lt, front_pred, back_pred,
                                              front_match, back_match,
                                              actual_front_list, actual_back_list)
                    print(f"    {'✅' if ok else '❌'} {type_names[lt]} 期{target_period}: {'推送成功' if ok else '推送失败'}")
                    import time
                    time.sleep(1)  # 间隔1秒避免限流
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"    ⚠️ {type_names[lt]} 推送异常: {e}")
            else:
                type_names = {'dlt': '大乐透', 'ssq': '双色球', 'kl8': '快乐8'}
                print(f"    ⚠️ {type_names[lt]}: 没有已验证记录")

    conn.commit()
    conn.close()

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
                    latest_period = next_period('dlt', df.iloc[-1]['period'])
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
                    latest_period = next_period('ssq', df_ssq.iloc[-1]['period'])
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
    print(f"  快乐8预测（选十组合）")
    print(f"{'=' * 40}")
    try:
        from kl8_data import ensure_kl8_data
        df_kl8 = ensure_kl8_data(num_periods=30)
        if df_kl8 is not None:
            result = pred_mod.predict_kl8(df_kl8, num_periods=30)
            if result:
                preds, report, kl8_combos = result
                pred_top20 = [str(n) for n in kl8_combos['top_20']]
                chart_configs.append({
                    'type': 'kl8',
                    'front': ','.join(pred_top20),
                    'back': '',
                })
                if not no_notify:
                    from notifier import save_prediction, send_prediction as send_pred
                    latest_period = next_period('kl8', df_kl8.iloc[-1]['period'])
                    save_prediction('kl8', latest_period, kl8_combos['top_20'])
                    ok = send_pred('kl8', preds, None, report, '', df_kl8, kl8_combos)
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
