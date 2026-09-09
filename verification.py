#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一预测验证模块
将 run_all.py 和 predict.py 中重复的验证逻辑抽取为独立模块
"""

import sqlite3
from db_utils import get_connection
from lottery_types import get_config


def auto_verify_predictions(lottery_type=None, no_notify=False):
    """
    统一自动验证入口：查找所有未验证的预测记录，与实际开奖对比并更新数据库
    
    参数:
        lottery_type: 指定彩种(None=全部)
        no_notify: 是否跳过钉钉推送
    
    返回:
        int: 本次验证的记录数
    """
    with get_connection(sqlite3.Row) as conn:
        # 查询所有未验证的记录
        if lottery_type:
            cur = conn.execute(
                "SELECT id, lottery_type, target_period, front_numbers, back_numbers "
                "FROM prediction_records "
                "WHERE (actual_front IS NULL OR actual_front = '') AND lottery_type=?",
                (lottery_type,)
            )
        else:
            cur = conn.execute(
                "SELECT id, lottery_type, target_period, front_numbers, back_numbers "
                "FROM prediction_records "
                "WHERE actual_front IS NULL OR actual_front = ''"
            )
        pending = list(cur.fetchall())

        if not pending:
            print("  没有待验证的预测记录")
            return 0

        auto_verified = 0
        for row in pending:
            pid = row['id']
            lt = row['lottery_type']
            period = row['target_period']
            front_str = row['front_numbers'] or ''
            back_str = row['back_numbers'] or ''

            front_pred = [int(x) for x in front_str.split(',') if x.strip().isdigit()]
            back_pred = [int(x) for x in back_str.split(',') if x.strip().isdigit()]

            actual_front_list, actual_back_list, actual_found = _get_actual_result(conn, lt, period)

            if actual_found:
                front_match = set(front_pred) & set(actual_front_list)
                back_match = set(back_pred) & set(actual_back_list)

                # 计算奖级
                from prize_calculator import get_prize_level_string
                match_level = get_prize_level_string(lt, len(front_match), len(back_match))

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

                # 终端显示
                _print_verification_result(lt, period, front_pred, back_pred,
                                           front_match, back_match,
                                           actual_front_list, actual_back_list)

                # 钉钉推送
                if not no_notify:
                    _push_verification(lt, period, front_pred, back_pred,
                                       front_match, back_match,
                                       actual_front_list, actual_back_list)

        # 统计
        cur = conn.execute("SELECT COUNT(*) FROM prediction_records")
        total = cur.fetchone()[0]
        cur = conn.execute(
            "SELECT COUNT(*) FROM prediction_records "
            "WHERE actual_front IS NOT NULL AND actual_front != ''"
        )
        verified = cur.fetchone()[0]
        unverified = total - verified
        print(f"  总计 {total} 条 | 已验证 {verified} 条 | 待验证 {unverified} 条")

    return auto_verified


def _get_actual_result(conn, lottery_type, period):
    """
    从数据库查询某期的实际开奖结果
    
    返回:
        (actual_front_list, actual_back_list, found: bool)
    """
    if lottery_type == 'kl8':
        cur = conn.execute(
            "SELECT numbers FROM kl8_records WHERE period=?", (period,)
        )
        r = cur.fetchone()
        if r:
            actual_nums = [int(x) for x in r[0].split(',') if x.strip().isdigit()]
            return actual_nums, [], True
        return [], [], False
    else:
        cfg = get_config(lottery_type)
        front_cnt = cfg['front_count']
        back_cnt = cfg['back_count']
        front_cols = ','.join([f'front_{k}' for k in range(1, front_cnt + 1)])
        back_cols = ','.join([f'back_{k}' for k in range(1, back_cnt + 1)])
        cur = conn.execute(
            f"SELECT {front_cols},{back_cols} FROM lottery_records "
            f"WHERE lottery_type=? AND period=?", (lottery_type, period)
        )
        r = cur.fetchone()
        if r:
            actual_front_list = [n for n in r[:front_cnt] if n is not None]
            actual_back_list = [n for n in r[front_cnt:] if n is not None]
            return actual_front_list, actual_back_list, True
        return [], [], False


def _print_verification_result(lt, period, front_pred, back_pred,
                               front_match, back_match,
                               actual_front_list, actual_back_list):
    """终端打印验证结果"""
    from prize_calculator import get_prize
    front_name_map = {'dlt': '前区', 'ssq': '红球', 'kl8': '号码'}
    back_name_map = {'dlt': '后区', 'ssq': '蓝球'}
    f_name = front_name_map.get(lt, '前区')
    b_name = back_name_map.get(lt, '后区')

    if lt == 'kl8':
        hit_str = f"{f_name}{len(front_match)}/20"
        print(f"   期{period}: {hit_str}")
    else:
        hit_str = f"{f_name}{len(front_match)}/{len(front_pred)}"
        if back_pred:
            hit_str += f" {b_name}{len(back_match)}/{len(back_pred)}"
        prize = get_prize(lt, len(front_match), len(back_match))
        level_str = prize['level'] or '未中奖'
        print(f"   期{period}: {hit_str} -> {level_str}")


def _push_verification(lt, period, front_pred, back_pred,
                       front_match, back_match,
                       actual_front_list, actual_back_list):
    """推送验证结果到钉钉"""
    try:
        from notifier import send_verification, send_kl8_verification
        if lt == 'kl8':
            from prize_calculator import generate_kl8_combinations
            kl8_combos = generate_kl8_combinations(front_pred)
            ok = send_kl8_verification(kl8_combos, actual_front_list, period)
        else:
            ok = send_verification(
                lt, front_pred, back_pred,
                front_match, back_match,
                actual_front_list, actual_back_list
            )
        if ok:
            print(f"     期{period}: 钉钉验证推送成功")
        else:
            print(f"     期{period}: 钉钉验证推送失败")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"     期{period}: 钉钉验证推送异常: {e}")
