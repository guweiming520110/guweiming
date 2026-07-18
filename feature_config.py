"""特征配置管理 - 统一管理所有特征列表"""

# 四段特征清单提取为模块级常量，供 get_all_features 与 print_feature_summary 共用，
# 避免双份硬编码导致维护时不一致（增删特征只需改这里一处）。
BASE_FEATURES = [
    'odd_count', 'even_count', 'large_count', 'small_count',
    'front_sum', 'ac_value', 'span',
    'max_front_omission', 'avg_front_omission', 'min_front_omission',
    'max_back_omission', 'avg_back_omission'
]

PHASE1_ADVANCED_FEATURES = [
    'max_pair_freq', 'max_triplet_freq', 'pmi_score',
    'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos',
    'consecutive_count', 'max_consecutive',
    'zone1_count', 'zone2_count', 'zone3_count',
    'rolling_std', 'rolling_entropy', 'similarity_score'
]

PHASE4_TEMPORAL_FEATURES = [
    'front_momentum_5', 'back_momentum_5',
    'front_momentum_10', 'back_momentum_10',
    'front_momentum_20', 'back_momentum_20',
    'odd_gap', 'even_gap', 'large_gap', 'small_gap',
    'prev_front_overlap', 'prev_back_overlap', 'transition_score'
]

# Phase 5 新增高阶特征
PHASE5_ADVANCED_FEATURES = [
    # 尾数特征
    'tail_0', 'tail_1', 'tail_2', 'tail_3', 'tail_4',
    'tail_5', 'tail_6', 'tail_7', 'tail_8', 'tail_9',
    'max_same_tail',
    'tail_mod3_0', 'tail_mod3_1', 'tail_mod3_2',
    # 和值趋势
    'sum_ma_5', 'sum_dev_5', 'sum_ma_10', 'sum_dev_10',
    # AC/跨度趋势
    'ac_ma_10', 'span_ma_10',
    # 边码/重号
    'edge_code_count', 'repeat_from_last',
    # Phase 6 共现/邻号/区间增强
    'pair_coverage_top5', 'pair_coverage_top10', 'pair_score_sum',
    'triplet_coverage_top5', 'triplet_coverage_top10', 'triplet_score_sum',
    'neighbor_hit_count', 'neighbor_hit_ratio',
    'zone4_1_count', 'zone4_2_count', 'zone4_3_count', 'zone4_4_count',
    'zone4_max_omission', 'zone4_hot_zone_id'
]

def get_all_features():
    """
    获取所有特征列表（Phase 1 + Phase 4 + Phase 5 新增）

    返回:
        list: 所有特征名称
    """
    all_features = (BASE_FEATURES + PHASE1_ADVANCED_FEATURES
                    + PHASE4_TEMPORAL_FEATURES + PHASE5_ADVANCED_FEATURES)

    return all_features

def get_available_features(df, feature_list=None):
    """
    获取数据框中实际存在的特征
    
    参数:
        df: DataFrame
        feature_list: 特征列表（默认使用全部特征）
    
    返回:
        list: 存在的特征列表
    """
    if feature_list is None:
        feature_list = get_all_features()
    
    available = [col for col in feature_list if col in df.columns]
    return available

def print_feature_summary():
    """打印特征总结"""
    all_features = get_all_features()

    # 直接引用模块级常量计数，与 get_all_features 共用同一份清单
    base_count = len(BASE_FEATURES)
    phase1_adv_count = len(PHASE1_ADVANCED_FEATURES)
    phase4_count = len(PHASE4_TEMPORAL_FEATURES)
    phase5_count = len(PHASE5_ADVANCED_FEATURES)

    print("📊 特征配置总结:")
    print(f"  Phase 1 基础特征: {base_count}个")
    print(f"  Phase 1 高级特征: {phase1_adv_count}个")
    print(f"  Phase 1 小计: {base_count + phase1_adv_count}个")
    print(f"  Phase 4 时间动态: {phase4_count}个")
    print(f"  Phase 5 新增高阶: {phase5_count}个")
    print(f"  ================")
    print(f"  总计: {len(all_features)}个")

    return len(all_features)

if __name__ == "__main__":
    features = get_all_features()
    print("所有特征列表:")
    for i, feat in enumerate(features, 1):
        print(f"  {i}. {feat}")
    print()
    print_feature_summary()
