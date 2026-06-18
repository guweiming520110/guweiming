#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预测器 — 统一入口，支持大乐透/快乐8/双色球
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pattern_learner.core import PatternLearner


def predict_dlt(df, top_k_front=10, top_k_back=5, num_periods=50, backtest_n=0):
    """预测大乐透"""
    print("\n" + "█" * 55)
    print("  超级大乐透 · 几何图案预测")
    print("█" * 55)

    learner_front = PatternLearner('dlt')
    learner_front.train(df, num_periods=num_periods)
    front_preds, front_report = learner_front.predict_with_report(top_k=top_k_front)
    print(front_report)

    learner_back = PatternLearner('dlt_back')
    learner_back.train(df, num_periods=num_periods)
    back_preds, back_report = learner_back.predict_with_report(top_k=top_k_back)
    print(back_report)

    backtest_result = None
    if backtest_n > 0:
        print("\n正在回测前区...")
        bf = PatternLearner('dlt')
        backtest_result = bf.backtest(df, test_periods=backtest_n)
        print("\n正在回测后区...")
        bb = PatternLearner('dlt_back')
        bb.backtest(df, test_periods=backtest_n)

    front_nums = [str(n) for n, s, d in front_preds[:5]]
    back_nums = [str(n) for n, s, d in back_preds[:2]]
    _print_cli_hint('dlt', front_nums, back_nums)

    return front_preds, back_preds, front_report, back_report


def predict_ssq(df, top_k_front=12, top_k_back=3, num_periods=50, backtest_n=0):
    """预测双色球"""
    print("\n" + "█" * 55)
    print("  双色球 · 几何图案预测")
    print("█" * 55)

    learner_front = PatternLearner('ssq')
    learner_front.train(df, num_periods=num_periods)
    front_preds, front_report = learner_front.predict_with_report(top_k=top_k_front)
    print(front_report)

    learner_back = PatternLearner('ssq_back')
    learner_back.train(df, num_periods=num_periods)
    back_preds, back_report = learner_back.predict_with_report(top_k=top_k_back)
    print(back_report)

    if backtest_n > 0:
        print("\n正在回测红球...")
        PatternLearner('ssq').backtest(df, test_periods=backtest_n)
        print("\n正在回测蓝球...")
        PatternLearner('ssq_back').backtest(df, test_periods=backtest_n)

    front_nums = [str(n) for n, s, d in front_preds[:6]]
    back_nums = [str(n) for n, s, d in back_preds[:1]]
    _print_cli_hint('ssq', front_nums, back_nums)

    return front_preds, back_preds, front_report, back_report


def predict_kl8(df, top_k=30, num_periods=30, backtest_n=0):
    """预测快乐8"""
    print("\n" + "█" * 55)
    print("  快乐8 · 几何图案预测")
    print("█" * 55)

    learner = PatternLearner('kl8')
    learner.train(df, num_periods=num_periods)
    preds, report = learner.predict_with_report(top_k=top_k)
    print(report)

    if backtest_n > 0:
        print("\n正在回测...")
        learner.backtest(df, test_periods=backtest_n)

    # 输出可视化命令
    pred_nums = [str(n) for n, s, d in preds[:20]]
    _print_kl8_hint(pred_nums)

    return preds, report


def _print_cli_hint(lottery_type, front_nums, back_nums):
    """打印走势图查看命令"""
    if lottery_type == 'dlt':
        cmd = f"python generate_charts.py --predict-front {','.join(front_nums)} --predict-back {','.join(back_nums)}"
    elif lottery_type == 'ssq':
        cmd = f"python generate_charts.py --type ssq --predict-front {','.join(front_nums)} --predict-back {','.join(back_nums)}"
    else:
        return
    print(f"\n💡 在走势图上查看预测号码:")
    print(f"   {cmd}")
    print(f"   (预测号码以深色圆球+白字显示在第0期位置)")
    print(f"\n💡 开奖后验证预测:")
    print(f"   {cmd} --verify")
    print(f"   (预测正确→黄底圆球 ✅ | 预测错误→红底圆球 ❌)")


def _print_kl8_hint(front_nums):
    """打印快乐8走势图命令"""
    cmd = f"python generate_charts.py --kl8 --predict-front {','.join(front_nums[:20])}"
    print(f"\n💡 在走势图上查看预测号码:")
    print(f"   {cmd}")
    print(f"   (预测号码以黑色圆球+白字显示在第0期位置)")


def predict(lottery_type='dlt', df=None, periods=50):
    """统一预测入口"""
    if lottery_type == 'dlt':
        return predict_dlt(df, num_periods=periods)
    elif lottery_type == 'kl8':
        return predict_kl8(df, num_periods=periods)
    elif lottery_type == 'ssq':
        return predict_ssq(df, num_periods=periods)
    else:
        raise ValueError(f"不支持的彩种: {lottery_type}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='几何图案预测系统')
    parser.add_argument('--kl8', action='store_true', help='预测快乐8')
    parser.add_argument('--type', choices=['dlt', 'kl8', 'ssq'], default='dlt',
                       help='彩种类型 (默认: dlt)')
    parser.add_argument('--periods', type=int, default=50, help='训练期数')
    parser.add_argument('--backtest', type=int, default=0, help='回测期数')
    parser.add_argument('--force', action='store_true', help='强制预测（跳过待验证检查）')
    parser.add_argument('--no-notify', action='store_true', help='不推送钉钉通知')
    parser.add_argument('--report', action='store_true', help='发送累计统计报告并退出')
    parser.add_argument('--pending', action='store_true', help='查看待验证的预测记录')
    args = parser.parse_args()

    # 统计报告模式
    if args.report:
        from notifier import send_stats_report, get_accuracy_stats
        stats = get_accuracy_stats()
        total_preds = sum(s['total'] for s in stats.values())
        total_verified = sum(s['verified'] for s in stats.values())
        print(f"📊 累计预测: {total_preds} 次 | 已验证: {total_verified} 次")
        ok = send_stats_report()
        print(f"{'✅' if ok else '❌'} 统计报告推送: {'成功' if ok else '失败'}")
        sys.exit(0)

    # 待验证查询模式
    if args.pending:
        from notifier import get_pending_verifications, send_text
        pending = get_pending_verifications()
        if pending:
            print(f"📋 发现 {len(pending)} 条待验证记录:")
            for r in pending[:10]:
                print(f"  [{r['lottery_type']}] 期{r['target_period']} 预测: {r['front_numbers']} | {r['back_numbers']} ({r['predict_time']})")
            # 发送到钉钉
            msg_lines = [f"📋 待验证预测 ({len(pending)} 条)"]
            for r in pending[:10]:
                msg_lines.append(f"- [{r['lottery_type']}] 期{r['target_period']}: {r['front_numbers']} | {r['back_numbers']}")
            if len(pending) > 10:
                msg_lines.append(f"... 还有 {len(pending)-10} 条")
            send_text('\n'.join(msg_lines))
            print("✅ 已推送到钉钉")
        else:
            print("✅ 没有待验证的记录")
        sys.exit(0)

    # 自动验证上期预测：有新开奖数据时自动验证
    lottery_type = 'kl8' if args.kl8 else args.type
    if not args.pending and not args.report:
        try:
            from notifier import get_pending_verifications, save_verification, send_verification
            import sqlite3
            from pathlib import Path

            pending = get_pending_verifications()
            my_pending = [p for p in pending if p['lottery_type'] == lottery_type]

            if my_pending:
                # 尝试自动验证
                db = Path(__file__).resolve().parent.parent / 'lottery.db'
                conn = sqlite3.connect(str(db))
                auto_verified = 0
                still_pending = []

                for p in my_pending:
                    target_period = p['target_period']
                    front_pred = [int(x) for x in p['front_numbers'].split(',') if x.strip().isdigit()]
                    back_pred = [int(x) for x in p['back_numbers'].split(',') if x.strip().isdigit()]

                    # 查找实际开奖数据
                    actual_found = False
                    if lottery_type == 'kl8':
                        cur = conn.execute(
                            "SELECT numbers FROM kl8_records WHERE period=?",
                            (target_period,)
                        )
                        row = cur.fetchone()
                        if row:
                            actual_nums = [int(x) for x in row[0].split(',') if x.strip().isdigit()]
                            front_match = set(front_pred) & set(actual_nums)
                            save_verification(lottery_type, target_period,
                                               p['front_numbers'], p['back_numbers'],
                                               actual_nums, [], front_match, set())
                            actual_found = True
                    else:
                        front_cnt = 5 if lottery_type == 'dlt' else 6
                        back_cnt = 2 if lottery_type == 'dlt' else 1
                        cur = conn.execute(
                            f"SELECT {','.join([f'front_{k}' for k in range(1, front_cnt+1)])},"
                            f"{','.join([f'back_{k}' for k in range(1, back_cnt+1)])} "
                            f"FROM lottery_records WHERE lottery_type=? AND period=?",
                            (lottery_type, target_period)
                        )
                        row = cur.fetchone()
                        if row:
                            actual_front = list(row[:front_cnt])
                            actual_back = list(row[front_cnt:])
                            front_match = set(front_pred) & set(actual_front)
                            back_match = set(back_pred) & set(actual_back)
                            save_verification(lottery_type, target_period,
                                               p['front_numbers'], p['back_numbers'],
                                               actual_front, actual_back,
                                               front_match, back_match)
                            actual_found = True

                    if actual_found:
                        auto_verified += 1
                        # 推送验证结果到钉钉
                        try:
                            if lottery_type == 'kl8':
                                if not args.no_notify:
                                    send_verification(lottery_type, front_pred, [],
                                                      front_match, set(), actual_nums, [])
                            else:
                                if not args.no_notify:
                                    send_verification(lottery_type, front_pred, back_pred,
                                                      front_match, back_match,
                                                      actual_front, actual_back)
                        except Exception:
                            pass
                        print(f"  ✅ 自动验证: 期{target_period} 命中前区{len(front_match)}/{len(front_pred)} 后区{len(back_match)}/{len(back_pred)}")
                    else:
                        still_pending.append(p)

                conn.close()

                if auto_verified > 0:
                    print(f"\n📌 自动验证了 {auto_verified} 条待验证预测")

                if still_pending:
                    print(f"\n⚠️  还有 {len(still_pending)} 条预测尚未到开奖时间，无法验证:")
                    for sp in still_pending[:3]:
                        print(f"    [{sp['lottery_type']}] 期{sp['target_period']}: {sp['front_numbers']} | {sp['back_numbers']}")
                    print(f"\n   使用 `python predict.py --pending` 查看全部待验证记录")
                    print(f"   使用 `python predict.py --type {lottery_type} --force` 强制进行新预测")
                    if not args.force:
                        print(f"\n   已阻止新预测，请先验证上期结果!")
                        sys.exit(1)
        except Exception as e:
            print(f"\n⚠️ 自动验证检查异常: {e}")
    from lottery_types import get_config
    cfg = get_config(lottery_type)

    df = None
    front_preds = back_preds = None
    front_report = back_report = None

    if lottery_type == 'kl8':
        from kl8_data import ensure_kl8_data
        df = ensure_kl8_data(args.periods)
        if df is not None:
            result = predict_kl8(df, num_periods=args.periods, backtest_n=args.backtest)
            if result:
                front_preds, front_report = result
    elif lottery_type == 'dlt':
        from generate_charts import load_data
        df = load_data()
        if df is not None:
            result = predict_dlt(df, num_periods=args.periods, backtest_n=args.backtest)
            if result:
                front_preds, back_preds, front_report, back_report = result
    elif lottery_type == 'ssq':
        from generate_charts import load_data
        df = load_data(lottery_type='ssq', limit=args.periods)
        if df is not None:
            result = predict_ssq(df, num_periods=args.periods, backtest_n=args.backtest)
            if result:
                front_preds, back_preds, front_report, back_report = result

    # ============ 保存预测记录 + 钉钉推送 ============
    if df is not None and front_preds:
        try:
            from notifier import save_prediction, send_prediction

            # 保存预测记录到数据库（始终保存）
            # 重要：预测的期号 = 最新开奖期号 + 1（下一期）
            latest_period = str(int(df.iloc[-1]['period']) + 1)
            if lottery_type == 'kl8':
                front_nums = [n for n, s, d in front_preds[:20]]
                save_prediction(lottery_type, latest_period, front_nums)
            else:
                front_cnt = 5 if lottery_type == 'dlt' else 6
                back_cnt = 2 if lottery_type == 'dlt' else 1
                front_nums = [n for n, s, d in front_preds[:front_cnt]]
                back_nums = [n for n, s, d in (back_preds or [])[:back_cnt]] if back_preds else []
                save_prediction(lottery_type, latest_period, front_nums, back_nums)

            # 发送钉钉通知（受 --no-notify 控制）
            if not args.no_notify:
                ok = send_prediction(
                    lottery_type, front_preds, back_preds,
                    front_report, back_report, df
                )
                print(f"\n{'✅' if ok else '❌'} 钉钉推送: {'成功' if ok else '失败'}")
            else:
                print(f"\n📝 预测已保存到数据库 (钉钉推送已跳过)")
        except Exception as e:
            print(f"\n⚠️ 保存/推送异常: {e}")
