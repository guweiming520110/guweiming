#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
奖级计算模块
根据预测命中数计算三大彩种的中奖奖级和奖金
"""

import itertools


# ==================== 大乐透奖级规则 ====================
# (前区命中数, 后区命中数, 奖级, 匹配描述, 奖金说明)
DLT_PRIZE_RULES = [
    (5, 2, '一等奖', '5+2', '浮动奖金（约1000万）'),
    (5, 1, '二等奖', '5+1', '浮动奖金（约20万）'),
    (5, 0, '三等奖', '5+0', '10000元'),
    (4, 2, '四等奖', '4+2', '3000元'),
    (4, 1, '五等奖', '4+1', '300元'),
    (3, 2, '六等奖', '3+2', '200元'),
    (4, 0, '七等奖', '4+0', '100元'),
    (3, 1, '八等奖', '3+1', '15元'),
    (2, 2, '八等奖', '2+2', '15元'),
    (3, 0, '九等奖', '3+0', '5元'),
    (1, 2, '九等奖', '1+2', '5元'),
    (2, 1, '九等奖', '2+1', '5元'),
    (0, 2, '九等奖', '0+2', '5元'),
]


# ==================== 双色球奖级规则 ====================
# (红球命中数, 蓝球命中数, 奖级, 匹配描述, 奖金说明)
SSQ_PRIZE_RULES = [
    (6, 1, '一等奖', '6+1', '浮动奖金（约500-1000万）'),
    (6, 0, '二等奖', '6+0', '浮动奖金（约10-30万）'),
    (5, 1, '三等奖', '5+1', '3000元'),
    (5, 0, '四等奖', '5+0', '200元'),
    (4, 1, '四等奖', '4+1', '200元'),
    (4, 0, '五等奖', '4+0', '10元'),
    (3, 1, '五等奖', '3+1', '10元'),
    (2, 1, '六等奖', '2+1', '5元'),
    (1, 1, '六等奖', '1+1', '5元'),
    (0, 1, '六等奖', '0+1', '5元'),
]


# ==================== 快乐8 选十玩法 官方奖级表 ====================
# (命中数, 奖级, 匹配描述, 奖金说明)
KL8_SHI10_PRIZE_RULES = [
    (10, '一等奖', '中10', '浮动奖金（最高500万）'),
    (9,  '二等奖', '中9', '浮动奖金（约8000元）'),
    (8,  '三等奖', '中8', '800元'),
    (7,  '四等奖', '中7', '80元'),
    (6,  '五等奖', '中6', '5元'),
    (5,  '六等奖', '中5', '3元'),
    (0,  '六等奖(全不中)', '中0', '2元（全不中）'),
]

# 选十各奖级的单注固定奖金（元）；一/二等奖为浮动奖金，无法精确累加，用 None 标记
# 复式拆单式后按此表累计总奖金，浮动档单独按注数统计
KL8_SHI10_BONUS_AMOUNT = {
    10: None,   # 一等奖：浮动（最高500万）
    9:  None,   # 二等奖：浮动（约8000元）
    8:  800,
    7:  80,
    6:  5,
    5:  3,
    0:  2,
}

# ==================== 快乐8 奖级规则 ====================
# 快乐8 预测20个号码，实际开奖20个，按命中数量分级
# (命中数, 命中率阈值, 等级名, 说明)
KL8_HIT_LEVELS = [
    (16, '⭐⭐⭐⭐⭐', '极佳', '命中率≥80%，建议重点关注'),
    (12, '⭐⭐⭐⭐', '优秀', '命中率≥60%，选号方向正确'),
    (10, '⭐⭐⭐', '良好', '命中率≥50%'),
    (8, '⭐⭐', '一般', '命中率≥40%'),
    (5, '⭐', '偏低', '命中率≥25%'),
    (0, '-', '未达标', '命中率<25%，建议调整策略'),
]

# 快乐8「选十」玩法奖级映射（从预测20码中任选10码，按期望命中估算）
# 20码命中N个 → 从中选10码的期望中奖数 ≈ N/2
# 20码命中数 → 选十对应奖级参考（按阈值降序排列）
KL8_SHI_PRIZE_MAP = [
    (16, '选十中八~中十', '有望冲击大奖'),
    (13, '选十中六~中七', '80~8000元'),
    (10, '选十中五~中六', '3~5元'),
    (0,  '选十中五以下', '未中奖'),
]


def _get_kl8_shi_reference(match_count):
    """根据命中数获取选十奖级参考"""
    for threshold, level_name, bonus_hint in KL8_SHI_PRIZE_MAP:
        if match_count >= threshold:
            return f'{level_name}（{bonus_hint}）'
    return '未中奖'


# ==================== 奖级计算函数 ====================

def get_dlt_prize(front_match_count: int, back_match_count: int) -> dict:
    """
    计算大乐透中奖奖级

    参数:
        front_match_count: 前区命中数 (0-5)
        back_match_count: 后区命中数 (0-2)

    返回:
        {
            'level': '一等奖' 或 None (未中奖),
            'match_desc': '5+2',
            'bonus': '浮动奖金（约1000万）',
            'is_winning': True/False,
            'front_match': 3,
            'back_match': 1,
        }
    """
    for fm, bm, level, desc, bonus in DLT_PRIZE_RULES:
        if front_match_count == fm and back_match_count == bm:
            return {
                'level': level,
                'match_desc': desc,
                'bonus': bonus,
                'is_winning': True,
                'front_match': front_match_count,
                'back_match': back_match_count,
            }
    return {
        'level': None,
        'match_desc': f'{front_match_count}+{back_match_count}',
        'bonus': '未中奖',
        'is_winning': False,
        'front_match': front_match_count,
        'back_match': back_match_count,
    }


def get_ssq_prize(front_match_count: int, back_match_count: int) -> dict:
    """
    计算双色球中奖奖级

    参数:
        front_match_count: 红球命中数 (0-6)
        back_match_count: 蓝球命中数 (0-1)

    返回:
        同 get_dlt_prize 结构
    """
    for fm, bm, level, desc, bonus in SSQ_PRIZE_RULES:
        if front_match_count == fm and back_match_count == bm:
            return {
                'level': level,
                'match_desc': desc,
                'bonus': bonus,
                'is_winning': True,
                'front_match': front_match_count,
                'back_match': back_match_count,
            }
    return {
        'level': None,
        'match_desc': f'{front_match_count}+{back_match_count}',
        'bonus': '未中奖',
        'is_winning': False,
        'front_match': front_match_count,
        'back_match': back_match_count,
    }


def get_kl8_sh10_prize(match_count: int) -> dict:
    """
    计算快乐8选十玩法的实际奖级（基于官方奖级表）

    参数:
        match_count: 10码中命中数 (0-10)

    返回:
        {'level': '一等奖', 'match_desc': '中10', 'bonus': '...', 'is_winning': True/False}
    """
    # 选十规则为精确命中数对应奖级：中10/9/8/7/6/5 中奖，中0(全不中)中奖，
    # 中1/2/3/4 不中奖。必须用等值匹配，避免 >= 把 中1-4 误判为 中0 档。
    for fm, level, desc, bonus in KL8_SHI10_PRIZE_RULES:
        if match_count == fm:
            return {
                'level': level,
                'match_desc': desc,
                'bonus': bonus,
                'is_winning': True,
                'match_count': match_count,
            }
    return {
        'level': '未中奖',
        'match_desc': f'中{match_count}',
        'bonus': '0元',
        'is_winning': False,
        'match_count': match_count,
    }


def generate_kl8_combinations(predictions: list) -> dict:
    """
    从 Top-24 预测号码生成选十投注组合

    参数:
        predictions: [(num, score, detail), ...] 按评分降序，或直接 [num, ...] 列表

    策略（改进2: 错位窗口，最大化号码覆盖）：
      旧实现 4 组复式全部基于 Top-10 + 个别补位，总覆盖仅 Top-14 且大量重复，
      6 组注之间重叠严重。改进后各组从不同评分段取号：
      - 单式A组: Top-1~10
      - 单式B组: Top-11~20
      - 复式11码A: Top-1~11
      - 复式11码B: Top-6~16（错位，覆盖中段）
      - 复式12码A: Top-1~12
      - 复式12码B: Top-9~20（错位，覆盖低段）
      → 总覆盖从 Top-14 提升到 Top-20，分散风险（号码分布广）且每注质量仍高。

    返回:
        {
            'single': [[10 nums], [10 nums]],     # 单式2组
            'complex_11': [[11 nums], [11 nums]],  # 复式10+1 2注
            'complex_12': [[12 nums], [12 nums]],  # 复式10+2 2注
            'top_20': [num, ...],                   # Top-20
        }
    """
    # 兼容两种输入：[(num, score, detail), ...] 或 [num, ...]
    if predictions and isinstance(predictions[0], (int, float)):
        all_nums = list(predictions)
    else:
        all_nums = [n for n, s, d in predictions]

    if len(all_nums) < 24:
        all_nums = all_nums[:min(len(all_nums), 24)]

    # 单式A: Top-1~10
    single_a = sorted(all_nums[:10])
    # 单式B: Top-11~20
    single_b = sorted(all_nums[10:20]) if len(all_nums) >= 20 else sorted(all_nums[10:])

    # 复式11码A: Top-1~11
    c11_a = sorted(all_nums[:11]) if len(all_nums) >= 11 else sorted(all_nums)
    # 复式11码B: Top-6~16（错位窗口，覆盖中段，与A重叠最小）
    c11_b = sorted(all_nums[5:16]) if len(all_nums) >= 16 else sorted(all_nums[:10])

    # 复式12码A: Top-1~12
    c12_a = sorted(all_nums[:12]) if len(all_nums) >= 12 else sorted(all_nums)
    # 复式12码B: Top-9~20（错位窗口，覆盖低段）
    c12_b = sorted(all_nums[8:20]) if len(all_nums) >= 20 else sorted(all_nums[:10])

    return {
        'single': [single_a, single_b],
        'complex_11': [c11_a, c11_b],
        'complex_12': [c12_a, c12_b],
        'top_20': all_nums[:20],
    }


def verify_kl8_combinations(combinations: dict, actual_nums: list) -> dict:
    """
    验证所有投注组合的命中情况

    参数:
        combinations: generate_kl8_combinations() 的返回值
        actual_nums: 实际开奖号码列表

    返回:
        {
            'single': [{'nums': [...], 'hit': [...], 'hit_count': N, 'prize': {...}}, ...],
            'complex_11': [...],
            'complex_12': [...],
            'top_20_hit': [...],  # 全部20码命中
            'top_20_hit_count': N,
        }
    """
    actual_set = set(actual_nums)

    def _verify_one(nums):
        nums = list(nums)
        hit = sorted(set(nums) & actual_set)
        hit_count = len(hit)

        # 单式（恰好10码）：直接按命中数定级
        if len(nums) == 10:
            prize = get_kl8_sh10_prize(hit_count)
            return {
                'nums': sorted(nums),
                'hit': hit,
                'hit_count': hit_count,
                'prize': prize,
            }

        # 号码不足 10 码：不是一注合法的选十投注，不能套用选十奖级表
        if len(nums) < 10:
            return {
                'nums': sorted(nums),
                'hit': hit,
                'hit_count': hit_count,
                'prize': {
                    'is_winning': False,
                    'level': '',
                    'bonus': '',
                    'match_desc': f'号码不足10码（{len(nums)}个），非有效选十投注',
                    'match_count': hit_count,
                },
            }

        # 复式（11码=10+1 / 12码=10+2）：本质是 C(n,10) 张 10码单式的组合，
        # 例如 11码=C(11,10)=11 注、12码=C(12,10)=66 注。
        # 必须对每一张 10码子集分别定级再汇总，不能把复式整体套单式奖级表，
        # 否则命中数>10 会落入兜底误判为“未中奖”，或整体套表算错奖金。
        hit_set = set(hit)
        total_bonus = 0.0        # 可精确累计的固定奖金合计（元），浮动大奖不计入此值
        has_floating = False     # 是否含浮动奖金档（一/二等奖，金额浮动无法精确累加）
        winning_tickets = 0      # 中奖注数
        ticket_count = 0         # 复式拆出的总注数
        # 各奖级注数明细：level -> {'count', 'match_count', 'match_desc', 'bonus'}
        level_breakdown = {}
        best_prize = None        # 最高奖级单注（供旧读取方按奖级展示/排序）

        for subset in itertools.combinations(sorted(nums), 10):
            ticket_count += 1
            sub_hit = len(hit_set & set(subset))
            pz = get_kl8_sh10_prize(sub_hit)
            if not pz['is_winning']:
                continue
            winning_tickets += 1
            amount = KL8_SHI10_BONUS_AMOUNT.get(sub_hit)
            if amount is None:
                has_floating = True
            else:
                total_bonus += amount
            # 用 (level_name, match_count) 复合键避免同名奖级(如中5/中0)相互覆盖
            lb_key = (pz['level'], sub_hit)
            lb = level_breakdown.setdefault(lb_key, {
                'count': 0,
                'match_count': sub_hit,
                'match_desc': pz['match_desc'],
                'bonus': pz['bonus'],
            })
            lb['count'] += 1
            if best_prize is None or sub_hit > best_prize['match_count']:
                best_prize = pz

        # 组装与单式兼容的 prize 字段（含总奖金与命中明细）
        if winning_tickets > 0:
            # winning_tickets>0 必有至少一中奖注，故 best_prize 一定已被赋值
            assert best_prize is not None
            # 各奖级注数明细文字，按命中数从高到低
            parts = [
                f"{v['match_desc']}×{v['count']}注"
                for _, v in sorted(level_breakdown.items(),
                                   key=lambda kv: kv[1]['match_count'], reverse=True)
            ]
            if has_floating:
                bonus_str = f"共{winning_tickets}注中奖（含浮动大奖），固定奖金合计{total_bonus:.0f}元"
            else:
                bonus_str = f"共{winning_tickets}注中奖，合计{total_bonus:.0f}元"
            prize = {
                'level': best_prize['level'],          # 最高奖级
                'match_desc': '，'.join(parts),         # 各奖级注数明细
                'bonus': bonus_str,                     # 总奖金描述
                'is_winning': True,
                'match_count': best_prize['match_count'],  # 最高命中数（供排序）
                'is_complex': True,
                'ticket_count': ticket_count,
                'winning_tickets': winning_tickets,
                'total_bonus': total_bonus,
                'has_floating': has_floating,
                'level_breakdown': level_breakdown,
            }
        else:
            prize = {
                'level': '未中奖',
                'match_desc': f'最高中{min(hit_count, 10)}',
                'bonus': '0元',
                'is_winning': False,
                'match_count': min(hit_count, 10),
                'is_complex': True,
                'ticket_count': ticket_count,
                'winning_tickets': 0,
                'total_bonus': 0.0,
                'has_floating': False,
                'level_breakdown': {},
            }

        return {
            'nums': sorted(nums),
            'hit': hit,
            'hit_count': hit_count,
            'prize': prize,
        }

    result = {}
    for key in ['single', 'complex_11', 'complex_12']:
        result[key] = [_verify_one(combos) for combos in combinations.get(key, [])]

    # Top-20 整体命中
    top_20 = combinations.get('top_20', [])
    top_20_hit = sorted(set(top_20) & actual_set)
    result['top_20_hit'] = top_20_hit
    result['top_20_hit_count'] = len(top_20_hit)

    return result


def get_kl8_hit_level(match_count: int) -> dict:
    """
    计算快乐8命中等级

    参数:
        match_count: 20码中命中数 (0-20)

    返回:
        {
            'stars': '⭐⭐⭐',
            'grade': '良好',
            'description': '命中率≥50%',
            'match_count': 10,
            'match_rate': 50.0,
        }
    """
    rate = match_count / 20 * 100
    for threshold, stars, grade, desc in KL8_HIT_LEVELS:
        if match_count >= threshold:
            # 选十参考
            shi_ref = _get_kl8_shi_reference(match_count)
            return {
                'stars': stars,
                'grade': grade,
                'description': desc,
                'match_count': match_count,
                'match_rate': round(rate, 1),
                'is_winning': match_count >= 10,
                'shi_reference': shi_ref,
            }
    # fallback (should never reach here)
    return {
        'stars': '-',
        'grade': '未达标',
        'description': '命中率<25%',
        'match_count': match_count,
        'match_rate': round(rate, 1),
        'is_winning': False,
        'shi_reference': '未中奖',
    }


def get_prize(lottery_type: str, front_match_count: int, back_match_count: int = 0) -> dict:
    """
    统一奖级计算入口

    参数:
        lottery_type: 'dlt' | 'ssq' | 'kl8'
        front_match_count: 前区/红球/号码命中数
        back_match_count: 后区/蓝球命中数 (kl8忽略)

    返回:
        dict 包含奖级信息
    """
    if lottery_type == 'dlt':
        return get_dlt_prize(front_match_count, back_match_count)
    elif lottery_type == 'ssq':
        return get_ssq_prize(front_match_count, back_match_count)
    elif lottery_type == 'kl8':
        return get_kl8_prize(front_match_count)
    else:
        raise ValueError(f"不支持的彩种: {lottery_type}")


def get_kl8_prize(match_count):
    """
    快乐8 统一奖级入口

    重要：系统预测的是 20 个号码，"20 码命中 N 个"并不等于某一注选十中奖
    （官方选十：单注 10 码中 10/9/8/7/6/5/0 才算中奖，中 1~4 不中奖）。
    因此这里只给出"命中评价 + 选十参考"，is_winning 恒为 False；
    真实中奖判定请用 verify_kl8_combinations() / send_kl8_verification()。

    返回:
        dict，含 level / bonus / is_winning / match_rate 等字段
    """
    info = get_kl8_hit_level(match_count)
    return {
        'level': f"{info.get('stars', '')} {info.get('grade', '')}".strip(),
        'bonus': info.get('shi_reference', ''),
        # 20 码命中数无法判定选十中奖，置为 False 以免推送"假中奖"
        'is_winning': False,
        'match_count': info.get('match_count', match_count),
        'match_rate': info.get('match_rate', 0),
        'stars': info.get('stars', ''),
        'grade': info.get('grade', ''),
        'description': info.get('description', ''),
    }


def get_prize_level_string(lottery_type: str, front_match_count: int,
                           back_match_count: int = 0) -> str:
    """
    获取奖级字符串（用于保存到数据库 match_level 字段）

    返回:
        '一等奖' | '八等奖' | '未中奖' | '⭐⭐⭐ 良好' (kl8)
    """
    result = get_prize(lottery_type, front_match_count, back_match_count)
    if lottery_type == 'kl8':
        return f"{result['stars']} {result['grade']}"
    return result['level'] or '未中奖'


def format_hit_detail(lottery_type: str, predicted_front: list, predicted_back: list,
                      actual_front: list, actual_back: list) -> str:
    """
    格式化命中详情文本（用于终端输出和钉钉消息）

    返回:
        多行字符串，显示命中/未命中的具体号码
    """
    front_name_map = {'dlt': '前区', 'ssq': '红球', 'kl8': '号码'}
    back_name_map = {'dlt': '后区', 'ssq': '蓝球'}

    front_name = front_name_map.get(lottery_type, '前区')
    back_name = back_name_map.get(lottery_type, '后区')

    front_set = set(predicted_front)
    actual_f_set = set(actual_front)
    hit_front = front_set & actual_f_set
    miss_front = front_set - actual_f_set

    lines = []
    lines.append(f"🎯 **命中{front_name}**: `{_fmt_nums(sorted(hit_front))}` ({len(hit_front)}个)")
    if miss_front:
        lines.append(f"❌ **未命中{front_name}**: `{_fmt_nums(sorted(miss_front))}` ({len(miss_front)}个)")

    if lottery_type != 'kl8' and predicted_back and actual_back:
        back_set = set(predicted_back)
        actual_b_set = set(actual_back)
        hit_back = back_set & actual_b_set
        miss_back = back_set - actual_b_set
        lines.append(f"🎯 **命中{back_name}**: `{_fmt_nums(sorted(hit_back))}` ({len(hit_back)}个)")
        if miss_back:
            lines.append(f"❌ **未命中{back_name}**: `{_fmt_nums(sorted(miss_back))}` ({len(miss_back)}个)")

    return '\n'.join(lines)


def format_prize_detail(lottery_type: str, front_match_count: int,
                        back_match_count: int = 0) -> str:
    """
    格式化中奖详情文本

    返回:
        多行 markdown 字符串
    """
    result = get_prize(lottery_type, front_match_count, back_match_count)

    lines = []
    if lottery_type == 'kl8':
        lines.append(f"**命中等级**: {result['stars']} {result['grade']}")
        lines.append(f"**命中数量**: {result['match_count']}/20 ({result['match_rate']}%)")
        lines.append(f"**{result['description']}**")
        if result.get('shi_reference'):
            lines.append(f"**选十参考**: {result['shi_reference']}")
    else:
        if result['is_winning']:
            lines.append(f"**奖级**: {result['level']}")
            lines.append(f"**匹配**: {result['match_desc']}")
            lines.append(f"**奖金**: {result['bonus']}")
        else:
            lines.append(f"**奖级**: 未中奖")
            lines.append(f"**匹配**: {result['match_desc']}（不满足任何奖级条件）")

    return '\n'.join(lines)


def _fmt_nums(nums: list) -> str:
    """格式化号码列表为两位数字符串"""
    return ', '.join(f'{n:02d}' if isinstance(n, int) else str(n) for n in nums)


# ==================== 测试 ====================

if __name__ == '__main__':
    # 测试大乐透
    print("=" * 50)
    print("大乐透奖级测试")
    print("=" * 50)
    for fm, bm in [(5, 2), (5, 1), (4, 2), (4, 1), (3, 2), (3, 1), (2, 2), (3, 0), (1, 2), (2, 1), (0, 2), (2, 0), (1, 0)]:
        r = get_dlt_prize(fm, bm)
        status = '✅' if r['is_winning'] else '❌'
        print(f"  前区{fm}+后区{bm} → {status} {r['level'] or '未中奖'} ({r['bonus']})")

    print()
    print("=" * 50)
    print("双色球奖级测试")
    print("=" * 50)
    for fm, bm in [(6, 1), (6, 0), (5, 1), (5, 0), (4, 1), (4, 0), (3, 1), (2, 1), (1, 1), (0, 1), (3, 0)]:
        r = get_ssq_prize(fm, bm)
        status = '✅' if r['is_winning'] else '❌'
        print(f"  红球{fm}+蓝球{bm} → {status} {r['level'] or '未中奖'} ({r['bonus']})")

    print()
    print("=" * 50)
    print("快乐8命中等级测试")
    print("=" * 50)
    for mc in [18, 15, 12, 10, 8, 5, 3, 0]:
        r = get_kl8_hit_level(mc)
        print(f"  命中{mc}/20 ({r['match_rate']}%) → {r['stars']} {r['grade']} | 选十: {r['shi_reference']}")

    print()
    print("=" * 50)
    print("命中详情格式化测试")
    print("=" * 50)
    detail = format_hit_detail(
        'dlt',
        predicted_front=[5, 12, 18, 25, 33],
        predicted_back=[4, 9],
        actual_front=[3, 12, 18, 25, 30],
        actual_back=[4, 11],
    )
    print(detail)
    print()
    print(format_prize_detail('dlt', 3, 1))
