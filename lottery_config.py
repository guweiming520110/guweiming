"""
彩票配置模块
定义不同彩票品种的规则、号码范围、中奖等级等
"""

LOTTERY_TYPES = {
    'dlt': {
        'name': '大乐透',
        'front_name': '前区',
        'back_name': '后区',
        'front_count': 5,
        'back_count': 2,
        'front_range': (1, 35),
        'back_range': (1, 12),
        'front_select_count': 5,
        'back_select_count': 2,
        'draw_days': ['一', '三', '六'],
        'prize_levels': {
            '一等奖': {'front': 5, 'back': 2},
            '二等奖': {'front': 5, 'back': 1},
            '三等奖': {'front': 5, 'back': 0},
            '四等奖': {'front': 4, 'back': 2},
            '五等奖': {'front': 4, 'back': 1},
            '六等奖': {'front': 3, 'back': 2, 'or': [(4, 0)]},
            '七等奖': {'front': 2, 'back': 2, 'or': [(3, 1), (None, 2)]},
            '八等奖': {'front': 3, 'back': 0, 'or': [(2, 1), (1, 2), (None, 1)]}
        },
        'data_sources': [
            'https://www.500.com/dlt/',
            'https://sports.sina.com.cn/l/dlt/'
        ]
    },
    'ssq': {
        'name': '双色球',
        'front_name': '红球',
        'back_name': '蓝球',
        'front_count': 6,
        'back_count': 1,
        'front_range': (1, 33),
        'back_range': (1, 16),
        'front_select_count': 6,
        'back_select_count': 1,
        'draw_days': ['二', '四', '日'],
        'prize_levels': {
            '一等奖': {'front': 6, 'back': 1},
            '二等奖': {'front': 6, 'back': 0},
            '三等奖': {'front': 5, 'back': 1},
            '四等奖': {'front': 5, 'back': 0, 'or': [(4, 1)]},
            '五等奖': {'front': 4, 'back': 0, 'or': [(3, 1)]},
            '六等奖': {'front': 0, 'back': 1}
        },
        'data_sources': [
            'https://www.500.com/ssq/',
            'https://sports.sina.com.cn/l/ssq/'
        ]
    },
    'fc3d': {
        'name': '福彩3D',
        'front_name': '号码',
        'back_name': None,
        'front_count': 3,
        'back_count': 0,
        'front_range': (0, 9),
        'back_range': None,
        'front_select_count': 3,
        'back_select_count': 0,
        'draw_days': ['每日'],
        'prize_levels': {
            '直选': {'exact_match': True},
            '组选三': {'two_same': True},
            '组选六': {'all_different': True}
        },
        'data_sources': [
            'https://www.500.com/3d/',
            'https://sports.sina.com.cn/l/3D/'
        ]
    }
}

def get_lottery_config(lottery_type):
    """获取指定彩票类型的配置"""
    if lottery_type not in LOTTERY_TYPES:
        raise ValueError(f"不支持的彩票类型: {lottery_type}")
    return LOTTERY_TYPES[lottery_type]

def validate_lottery_type(lottery_type):
    """验证彩票类型是否有效"""
    return lottery_type in LOTTERY_TYPES

def get_all_lottery_types():
    """获取所有支持的彩票类型"""
    return list(LOTTERY_TYPES.keys())

def get_lottery_display_name(lottery_type):
    """获取彩票的显示名称"""
    config = get_lottery_config(lottery_type)
    return config['name']

def validate_numbers(lottery_type, front_numbers, back_numbers):
    """验证号码是否符合规则"""
    config = get_lottery_config(lottery_type)
    
    if len(front_numbers) != config['front_select_count']:
        return False, f"{config['front_name']}必须选择{config['front_select_count']}个号码"
    
    if config['back_count'] > 0:
        if len(back_numbers) != config['back_select_count']:
            return False, f"{config['back_name']}必须选择{config['back_select_count']}个号码"
    
    front_min, front_max = config['front_range']
    for num in front_numbers:
        if num < front_min or num > front_max:
            return False, f"{config['front_name']}号码必须在{front_min}-{front_max}之间"
    
    if config['back_count'] > 0 and config['back_range']:
        back_min, back_max = config['back_range']
        for num in back_numbers:
            if num < back_min or num > back_max:
                return False, f"{config['back_name']}号码必须在{back_min}-{back_max}之间"
    
    return True, "验证通过"

def calculate_prize_level(lottery_type, front_match, back_match):
    """计算中奖等级"""
    config = get_lottery_config(lottery_type)
    
    if lottery_type == 'dlt':
        if front_match == 5 and back_match == 2:
            return "一等奖"
        elif front_match == 5 and back_match == 1:
            return "二等奖"
        elif front_match == 5 and back_match == 0:
            return "三等奖"
        elif front_match == 4 and back_match == 2:
            return "四等奖"
        elif front_match == 4 and back_match == 1:
            return "五等奖"
        elif front_match == 3 and back_match == 2:
            return "六等奖"
        elif front_match == 4 and back_match == 0:
            return "七等奖"
        elif (front_match == 3 and back_match == 1) or (front_match == 2 and back_match == 2):
            return "八等奖"
        elif (front_match == 3 and back_match == 0) or (front_match == 2 and back_match == 1) or (front_match == 1 and back_match == 2) or (front_match == 0 and back_match == 2):
            return "九等奖"
        else:
            return "未中奖"
    
    elif lottery_type == 'ssq':
        if front_match == 6 and back_match == 1:
            return "一等奖"
        elif front_match == 6 and back_match == 0:
            return "二等奖"
        elif front_match == 5 and back_match == 1:
            return "三等奖"
        elif front_match == 5 and back_match == 0 or (front_match == 4 and back_match == 1):
            return "四等奖"
        elif front_match == 4 and back_match == 0 or (front_match == 3 and back_match == 1):
            return "五等奖"
        elif back_match == 1:
            return "六等奖"
        else:
            return "未中奖"
    
    elif lottery_type == 'fc3d':
        return "待实现"
    
    return "未中奖"

def format_numbers_display(lottery_type, front_numbers, back_numbers=None):
    """格式化号码显示"""
    config = get_lottery_config(lottery_type)
    
    if lottery_type == 'fc3d':
        return ''.join([f"{num:01d}" for num in front_numbers])
    else:
        front_str = ','.join([f"{num:02d}" for num in front_numbers])
        if back_numbers and config['back_count'] > 0:
            back_str = ','.join([f"{num:02d}" for num in back_numbers])
            return f"{front_str} + {back_str}"
        return front_str
