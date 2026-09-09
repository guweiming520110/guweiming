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

退出码:
  0 = 全部步骤成功
  1 = 至少一个步骤失败（供 cron / CI 判断，避免"失败也报成功"）
"""

import sys
import os
import time
import traceback
import argparse
from pathlib import Path
from datetime import datetime

# 从项目根目录加载 .env 环境变量（覆盖系统已有变量）；未装 python-dotenv 时静默降级
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / '.env'
    if _env_path.exists():
        load_dotenv(dotenv_path=str(_env_path), override=True)
except ImportError:
    pass

from lottery_types import next_period, get_config
from notifier import (
    send_verification, send_kl8_verification,
    save_prediction, send_prediction
)
from prize_calculator import (
    get_prize_level_string, get_prize, generate_kl8_combinations,
    verify_kl8_combinations
)
from generate_charts import load_data
from kl8_data import ensure_kl8_data
import predict as pred_mod
import fill_dlt_data
import fill_ssq_data

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# 确保本项目模块可导入（必须在导入本项目模块之前设置）
# 注意：os.chdir 已移入 main()，避免 import 本模块就产生切换工作目录的副作用
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(ROOT_DIR))


def step_header(num, total, icon, title):
    print(f"\n{'█' * 55}")
    print(f"  Step {num}/{total}: {icon} {title}")
    print(f"{'█' * 55}")


TOTAL_STEPS = 5


def step_init_db():
    """初始化数据库表结构（首次运行/空库场景必需）"""
    step_header(1, TOTAL_STEPS, '🗄️', '初始化数据库')
    from db_utils import get_connection
    from schema import init_db
    with get_connection() as _init_conn:
        init_db(_init_conn)


def step_update_data():
    """更新三大彩种开奖数据

    返回:
        dict: {'dlt': bool, 'ssq': bool, 'kl8': bool} 各彩种数据是否可用。
        数据不可用时后续步骤必须跳过该彩种，否则会用过期期号预测，
        进而覆盖掉已验证记录的后验结果。
    """
    step_header(2, TOTAL_STEPS, '📥', '更新三大彩种开奖数据')
    ok = {'dlt': True, 'ssq': True, 'kl8': True}
    try:
        _st = fill_dlt_data.fill_missing_data()
        ok['dlt'] = bool(_st and _st.get('count'))
    except Exception as e:
        ok['dlt'] = False
        print(f"  ⚠️ 大乐透数据更新异常: {e}")
        traceback.print_exc()
    try:
        _st = fill_ssq_data.fill_ssq_data()
        ok['ssq'] = bool(_st and _st.get('count'))
    except Exception as e:
        ok['ssq'] = False
        print(f"  ⚠️ 双色球数据更新异常: {e}")
        traceback.print_exc()
    try:
        _df_kl8 = ensure_kl8_data(num_periods=100, force_refresh=True)
        _n = len(_df_kl8) if _df_kl8 is not None else 0
        print(f"  快乐8: {_n} 期 ✅")
        ok['kl8'] = _n > 0
    except Exception as e:
        ok['kl8'] = False
        print(f"  ⚠️ 快乐8数据更新异常: {e}")
        traceback.print_exc()
    return ok


def step_verify_predictions(no_notify):
    """Step 2: 自动验证未验证的旧预测记录"""
    step_header(3, TOTAL_STEPS, '🔄', '自动验证未验证的预测记录')
    from db_utils import get_connection
    verified_pushed = set()   # 本轮已推送过验证结果的彩种，避免同一期推送两次
    with get_connection() as conn:

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
                    cfg = get_config(lt)
                    front_cnt = int(cfg['front_count'])
                    back_cnt = int(cfg['back_count'])
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
                    kl8_combos: dict = {}
                    # 计算奖级
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
                            if lt == 'kl8':
                                # kl8_combos 只有在 kl8 且 actual_found 时才会生成
                                combos = kl8_combos or generate_kl8_combinations(front_pred)
                                ok = send_kl8_verification(combos, actual_front_list, period)
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
                                verified_pushed.add(lt)
                        except Exception as e:
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
                if lt in verified_pushed:
                    # 刚才已验证并推送过这一期，跳过以免重复打扰
                    continue
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
                    row_lt = row[1]  # 使用独立变量，不覆盖循环变量 lt
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
                    if actual_row is None:
                        # 两次查询之间记录可能被删除，解引用 None 会中断整个流程
                        continue
                    actual_front_str = actual_row[0] or ''
                    actual_back_str = actual_row[1] or ''
                    actual_front_list = [int(x) for x in actual_front_str.split(',') if x.strip().isdigit()]
                    actual_back_list = [int(x) for x in actual_back_str.split(',') if x.strip().isdigit()]
                    front_match = set(front_pred) & set(actual_front_list)
                    back_match = set(back_pred) & set(actual_back_list)

                    type_names = {'dlt': '大乐透', 'ssq': '双色球', 'kl8': '快乐8'}
                    try:
                        if row_lt == 'kl8':
                            kl8_combos = generate_kl8_combinations(front_pred)
                            ok = send_kl8_verification(kl8_combos, actual_front_list, target_period)
                        else:
                            ok = send_verification(row_lt, front_pred, back_pred,
                                                  front_match, back_match,
                                                  actual_front_list, actual_back_list)
                        print(f"    {'✅' if ok else '❌'} {type_names[lt]} 期{target_period}: {'推送成功' if ok else '推送失败'}")
                        time.sleep(1)  # 间隔1秒避免限流
                    except Exception as e:
                        traceback.print_exc()
                        print(f"    ⚠️ {type_names[lt]} 推送异常: {e}")
                else:
                    type_names = {'dlt': '大乐透', 'ssq': '双色球', 'kl8': '快乐8'}
                    print(f"    ⚠️ {type_names[lt]}: 没有已验证记录")



def step_run_predictions(no_notify, data_ok=None, failures=None):
    """Step 4: 运行三大彩种预测，返回 chart_configs 供走势图使用

    参数:
        data_ok: step_update_data() 的返回值；数据不可用的彩种会跳过预测，
                 避免用过期期号生成预测并覆盖已验证记录的后验结果
        failures: 失败步骤清单；单个彩种预测失败/无产出时 append 描述，
                 由 main 统一决定退出码（避免"三大彩种全失败但仍退出 0"）
    """
    step_header(4, TOTAL_STEPS, '🎯', '运行预测分析')
    data_ok = data_ok or {'dlt': True, 'ssq': True, 'kl8': True}
    failures = failures if failures is not None else []
    cfg_dlt = get_config('dlt')
    cfg_ssq = get_config('ssq')

    # 存储预测结果用于走势图
    chart_configs = []

    # --- 3a: 大乐透 ---
    print(f"\n{'=' * 40}")
    print(f"  超级大乐透预测")
    print(f"{'=' * 40}")
    try:
        if data_ok.get('dlt'):
            df = load_data()
        else:
            print("  ⚠️ 跳过：大乐透数据不可用（上一步更新失败）")
            df = None
            failures.append('预测: 大乐透数据不可用，已跳过')
        if df is not None:
            result = pred_mod.predict_dlt(df, num_periods=50, backtest_n=5, use_ml=True)
            if result:
                front_preds, back_preds, front_report, back_report = result
                front_top5 = [str(n) for n, s, d in front_preds[:cfg_dlt['front_count']]]
                back_top2 = [str(n) for n, s, d in back_preds[:cfg_dlt['back_count']]]
                chart_configs.append({
                    'type': 'dlt',
                    'front': ','.join(front_top5),
                    'back': ','.join(back_top2),
                })
                # 钉钉推送
                if not no_notify:
                    latest_period = next_period('dlt', df.iloc[-1]['period'])
                    front_nums = [n for n, s, d in front_preds[:5]]
                    back_nums = [n for n, s, d in back_preds[:2]]
                    save_prediction('dlt', latest_period, front_nums, back_nums)
                    ok = send_prediction('dlt', front_preds, back_preds, front_report, back_report, df, target_period=latest_period)
                    print(f"  {'✅' if ok else '❌'} 钉钉推送: {'成功' if ok else '失败'}")
            else:
                failures.append('预测: 大乐透未产出任何预测结果')
    except Exception as e:
        print(f"  ⚠️ 大乐透预测异常: {e}")
        traceback.print_exc()
        failures.append(f"预测: 大乐透异常: {e}")

    # --- 3b: 双色球 ---
    print(f"\n{'=' * 40}")
    print(f"  双色球预测")
    print(f"{'=' * 40}")
    try:
        if data_ok.get('ssq'):
            df_ssq = load_data(lottery_type='ssq', limit=50)
        else:
            print("  ⚠️ 跳过：双色球数据不可用（上一步更新失败）")
            df_ssq = None
            failures.append('预测: 双色球数据不可用，已跳过')
        if df_ssq is not None:
            result = pred_mod.predict_ssq(df_ssq, num_periods=50, backtest_n=5)
            if result:
                front_preds, back_preds, front_report, back_report = result
                front_top6 = [str(n) for n, s, d in front_preds[:cfg_ssq['front_count']]]
                back_top1 = [str(n) for n, s, d in back_preds[:cfg_ssq['back_count']]]
                chart_configs.append({
                    'type': 'ssq',
                    'front': ','.join(front_top6),
                    'back': ','.join(back_top1),
                })
                if not no_notify:
                    latest_period = next_period('ssq', df_ssq.iloc[-1]['period'])
                    front_nums = [n for n, s, d in front_preds[:6]]
                    back_nums = [n for n, s, d in back_preds[:1]]
                    save_prediction('ssq', latest_period, front_nums, back_nums)
                    ok = send_prediction('ssq', front_preds, back_preds, front_report, back_report, df_ssq, target_period=latest_period)
                    print(f"  {'✅' if ok else '❌'} 钉钉推送: {'成功' if ok else '失败'}")
            else:
                failures.append('预测: 双色球未产出任何预测结果')
    except Exception as e:
        print(f"  ⚠️ 双色球预测异常: {e}")
        traceback.print_exc()
        failures.append(f"预测: 双色球异常: {e}")

    # --- 3c: 快乐8 ---
    print(f"\n{'=' * 40}")
    print(f"  快乐8预测（选十组合）")
    print(f"{'=' * 40}")
    try:
        if data_ok.get('kl8'):
            df_kl8 = ensure_kl8_data(num_periods=30)
        else:
            print("  ⚠️ 跳过：快乐8数据不可用（上一步更新失败）")
            df_kl8 = None
            failures.append('预测: 快乐8数据不可用，已跳过')
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
                    latest_period = next_period('kl8', df_kl8.iloc[-1]['period'])
                    save_prediction('kl8', latest_period, kl8_combos['top_20'])
                    ok = send_prediction('kl8', preds, None, report, '', df_kl8, kl8_combos, target_period=latest_period)
                    print(f"  {'✅' if ok else '❌'} 钉钉推送: {'成功' if ok else '失败'}")
            else:
                failures.append('预测: 快乐8未产出任何预测结果')
    except Exception as e:
        print(f"  ⚠️ 快乐8预测异常: {e}")
        traceback.print_exc()
        failures.append(f"预测: 快乐8异常: {e}")

    return chart_configs


def step_generate_charts(chart_configs, skip_charts=False):
    """Step 5: 生成含预测走势图"""
    if skip_charts:
        step_header(5, TOTAL_STEPS, '⏭️', '走势图生成已跳过')
        return

    step_header(5, TOTAL_STEPS, '📊', '生成含预测走势图')
    from charts.composite import generate_composite_chart

    for cfg in chart_configs:
        lt = cfg['type']
        pf = [int(x) for x in cfg['front'].split(',') if x.strip()]
        pb = [int(x) for x in cfg['back'].split(',') if x.strip()] if cfg['back'] else None
        names = {'dlt': '大乐透', 'ssq': '双色球', 'kl8': '快乐8'}

        print(f"\n  正在生成 {names.get(lt, lt)}走势图...")
        try:
            if lt == 'kl8':
                df_c = ensure_kl8_data(num_periods=30)
            else:
                df_c = load_data(lottery_type=lt, limit=50)

            if df_c is not None:
                out = generate_composite_chart(
                    df_c, lottery_type=lt,
                    network_periods=min(30, len(df_c)),
                    trend_periods=min(50, len(df_c)),
                    freq_periods=min(100, len(df_c)),
                    predicted_front=pf,
                    predicted_back=pb,
                )
                if out:
                    print(f"  ✅ {names.get(lt, lt)}走势图生成完成")
                else:
                    print(f"  ❌ {names.get(lt, lt)}走势图生成失败（详见上方错误输出）")
        except Exception as e:
            print(f"  ⚠️ {names.get(lt, lt)}走势图异常: {e}")
            traceback.print_exc()


def main():
    # os.chdir 只在这里执行：模块被 import 时不应改变调用方的工作目录
    os.chdir(str(BASE_DIR))

    parser = argparse.ArgumentParser(description='一键全自动运行')
    
    parser.add_argument('--no-notify', action='store_true', help='不推送钉钉通知')
    parser.add_argument('--skip-charts', action='store_true', help='跳过走势图生成')
    
    args = parser.parse_args()

    no_notify = args.no_notify
    failures = []          # 失败步骤清单，决定进程退出码
    
    print(f"{'█' * 55}")
    print(f"  🚀 数字彩票AI分析系统 · 一键全自动运行")
    print(f"  📂 {BASE_DIR}")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'█' * 55}")
    
    # Step 1: 初始化数据库
    step_init_db()
    
    # Step 2: 更新数据（各彩种可用性会传给预测步骤，不可用则跳过）
    data_ok = step_update_data()
    if not any(data_ok.values()):
        failures.append('数据更新: 三大彩种均无可用数据')

    # Step 3: 自动验证未验证的旧预测记录（此时数据库已是最新开奖，避免落后一轮）
    step_verify_predictions(no_notify)

    # Step 4: 运行三大彩种预测
    chart_configs = step_run_predictions(no_notify, data_ok=data_ok, failures=failures)
    if not chart_configs:
        failures.append('预测: 未产出任何预测结果')

    # Step 5: 生成走势图
    step_generate_charts(chart_configs, skip_charts=args.skip_charts)

    # ──────────────────────────────
    # 完成
    # ──────────────────────────────
    print(f"\n{'█' * 55}")
    print(f"  ✅ 一键全自动运行完成!")
    print(f"  📂 走势图: {BASE_DIR / 'charts'}")
    print(f"{'█' * 55}")

    # 统计
    from db_utils import get_connection
    with get_connection(commit=False) as conn:
        cur = conn.execute(
            'SELECT lottery_type, COUNT(*) FROM prediction_records GROUP BY lottery_type'
        )
        rows = cur.fetchall()
    if rows:
        print(f"\n📋 本次预测记录:")
        type_names = {'dlt': '大乐透', 'ssq': '双色球', 'kl8': '快乐8'}
        for lt, cnt in rows:
            print(f"  {type_names.get(lt, lt)}: {cnt} 条")
    print()

    if failures:
        print("⚠️ 本次运行存在失败步骤:")
        for f in failures:
            print(f"   - {f}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
