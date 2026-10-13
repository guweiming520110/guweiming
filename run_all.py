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
    generate_kl8_combinations,
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


def _push_state_file():
    """验证推送去重状态文件路径（置于子模块根，.gitignore 已忽略）"""
    return Path(BASE_DIR) / '.push_state.json'


def _load_push_state():
    """读取上次各彩种推送过的验证期号，用于跨天去重"""
    import json
    try:
        f = _push_state_file()
        if f.exists():
            return json.loads(f.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {}


def _save_push_state(state):
    """持久化推送去重状态；写失败不影响主流程"""
    import json
    try:
        _push_state_file().write_text(json.dumps(state), encoding='utf-8')
    except Exception:
        pass


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
    # 改进4: 数据可用性判定——"更新到最新"才可用。
    # 旧实现用 bool(count>0) 判断，爬取失败但库有旧数据时误判可用，
    # 导致用过期期号预测并覆盖已验证记录后验。
    from db_utils import get_connection

    def _is_fresh(table, lt=None, max_stale_days=3):
        """检查该彩种最新开奖日期是否在容差内（新鲜=可用）。"""
        try:
            with get_connection(commit=False) as conn:
                if table == 'kl8_records':
                    row = conn.execute("SELECT MAX(date) FROM kl8_records").fetchone()
                else:
                    row = conn.execute(
                        "SELECT MAX(date) FROM lottery_records WHERE lottery_type=?", (lt,)).fetchone()
                if not row or not row[0]:
                    return False
                from datetime import datetime
                latest = datetime.strptime(row[0], '%Y-%m-%d').date()
                return (datetime.now().date() - latest).days <= max_stale_days
        except Exception:
            return False

    try:
        _st = fill_dlt_data.fill_missing_data()
        ok['dlt'] = _is_fresh('lottery_records', 'dlt')
        if not ok['dlt']:
            print(f"  ⚠️ 大乐透数据未更新到最新(可能爬取失败/停更)，标记不可用")
    except Exception as e:
        ok['dlt'] = False
        print(f"  ⚠️ 大乐透数据更新异常: {e}")
        traceback.print_exc()
    try:
        _st = fill_ssq_data.fill_ssq_data()
        ok['ssq'] = _is_fresh('lottery_records', 'ssq')
        if not ok['ssq']:
            print(f"  ⚠️ 双色球数据未更新到最新(可能爬取失败/停更)，标记不可用")
    except Exception as e:
        ok['ssq'] = False
        print(f"  ⚠️ 双色球数据更新异常: {e}")
        traceback.print_exc()
    try:
        _df_kl8 = ensure_kl8_data(num_periods=100, force_refresh=True)
        _n = len(_df_kl8) if _df_kl8 is not None else 0
        print(f"  快乐8: {_n} 期 ✅")
        ok['kl8'] = _is_fresh('kl8_records')
        if not ok['kl8']:
            print(f"  ⚠️ 快乐8数据未更新到最新，标记不可用")
    except Exception as e:
        ok['kl8'] = False
        print(f"  ⚠️ 快乐8数据更新异常: {e}")
        traceback.print_exc()
    return ok


def step_verify_predictions(no_notify):
    """Step 3: 自动验证未验证的旧预测记录

    去重修复: 验证/落库/推送核心逻辑复用 verification.auto_verify_predictions，
    本函数只保留 run_all 独有的"跨天去重推送各彩种最近一期验证结果"。
    """
    step_header(3, TOTAL_STEPS, '🔄', '自动验证未验证的预测记录')
    from verification import auto_verify_predictions
    from db_utils import get_connection

    # 1) 核心验证：查找待验证记录、对比开奖、更新数据库、终端显示、钉钉推送
    auto_verify_predictions(no_notify=no_notify)

    # 2) 自动推送每个彩种最近一期验证结果（确保每次都能看到）
    if no_notify:
        return
    from notifier import send_verification, send_kl8_verification
    from prize_calculator import generate_kl8_combinations

    # 跨天去重：同一彩种同一期号的验证结果只推一次，避免每轮运行重复骚扰/逼近限速
    push_state = _load_push_state()
    print("\n  📤 推送各彩种最近一期验证结果...")
    with get_connection() as conn:
        for lt in ['dlt', 'ssq', 'kl8']:
            type_names = {'dlt': '大乐透', 'ssq': '双色球', 'kl8': '快乐8'}
            # 查询最近一期已验证记录
            cur = conn.execute(
                """SELECT * FROM prediction_records
                   WHERE lottery_type=? AND actual_front IS NOT NULL AND actual_front != ''
                   ORDER BY target_period DESC LIMIT 1""",
                (lt,)
            )
            row = cur.fetchone()
            if not row:
                print(f"    ⚠️ {type_names[lt]}: 没有已验证记录")
                continue
            # 列顺序: id, lottery_type, predict_time, target_period, play_type, front_numbers, back_numbers, ...
            pid, row_lt = row[0], row[1]
            target_period = row[3]
            # 跨天去重：该彩种此期验证结果已推送过 → 跳过，不重复打扰
            if push_state.get(f'verif_{lt}') == str(target_period):
                print(f"    ⏭️ {type_names[lt]} 期{target_period} 验证结果已推送过，跳过")
                continue
            front_str, back_str = row[5], row[6]
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
            actual_front_list = [int(x) for x in (actual_row[0] or '').split(',') if x.strip().isdigit()]
            actual_back_list = [int(x) for x in (actual_row[1] or '').split(',') if x.strip().isdigit()]
            front_match = set(front_pred) & set(actual_front_list)
            back_match = set(back_pred) & set(actual_back_list)

            try:
                if row_lt == 'kl8':
                    kl8_combos = generate_kl8_combinations(front_pred)
                    ok = send_kl8_verification(kl8_combos, actual_front_list, target_period)
                else:
                    ok = send_verification(row_lt, front_pred, back_pred,
                                          front_match, back_match,
                                          actual_front_list, actual_back_list)
                if ok:
                    push_state[f'verif_{lt}'] = str(target_period)
                print(f"    {'✅' if ok else '❌'} {type_names[lt]} 期{target_period}: {'推送成功' if ok else '推送失败'}")
                time.sleep(1)  # 间隔1秒避免限流
            except Exception as e:
                traceback.print_exc()
                print(f"    ⚠️ {type_names[lt]} 推送异常: {e}")
    _save_push_state(push_state)


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
                # M1 修复: save_prediction 不受 --no-notify 影响——预测记录始终落库，
                # 否则无通知模式下验证机制失效（无记录可对）。钉钉推送才受 no_notify 控制。
                latest_period = next_period('dlt', df.iloc[-1]['period'])
                front_nums = [n for n, s, d in front_preds[:5]]
                back_nums = [n for n, s, d in back_preds[:2]]
                save_prediction('dlt', latest_period, front_nums, back_nums)
                if not no_notify:
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
            result = pred_mod.predict_ssq(df_ssq, num_periods=50, backtest_n=5, use_ml=True)
            if result:
                front_preds, back_preds, front_report, back_report = result
                front_top6 = [str(n) for n, s, d in front_preds[:cfg_ssq['front_count']]]
                back_top1 = [str(n) for n, s, d in back_preds[:cfg_ssq['back_count']]]
                chart_configs.append({
                    'type': 'ssq',
                    'front': ','.join(front_top6),
                    'back': ','.join(back_top1),
                })
                # M1 修复: 同 dlt，预测记录始终落库，钉钉推送受 no_notify 控制
                latest_period = next_period('ssq', df_ssq.iloc[-1]['period'])
                front_nums = [n for n, s, d in front_preds[:6]]
                back_nums = [n for n, s, d in back_preds[:1]]
                save_prediction('ssq', latest_period, front_nums, back_nums)
                if not no_notify:
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
                # M1 修复: 同 dlt/ssq，预测记录始终落库
                latest_period = next_period('kl8', df_kl8.iloc[-1]['period'])
                save_prediction('kl8', latest_period, kl8_combos['top_20'])
                if not no_notify:
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
    print(f"  📂 走势图输出目录: {ROOT_DIR / 'output'}")
    print(f"    (config.OUTPUT_DIR，与 .py 源码分离)")
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
