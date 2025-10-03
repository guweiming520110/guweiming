"""特征配置管理 - 统一管理所有特征列表"""

def get_all_features():
    """
    获取所有特征列表（Phase 1 + Phase 4）
    
    返回:
        list: 所有特征名称
    """
    base_features = [
        'odd_count', 'even_count', 'large_count', 'small_count', 
        'front_sum', 'ac_value', 'span', 
        'max_front_omission', 'avg_front_omission', 
        'max_back_omission', 'avg_back_omission'
    ]
    
    phase1_advanced_features = [
        'max_pair_freq', 'max_triplet_freq', 'pmi_score',
        'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos',
        'consecutive_count', 'max_consecutive',
        'zone1_count', 'zone2_count', 'zone3_count',
        'rolling_std', 'rolling_entropy', 'similarity_score'
    ]
    
    phase4_temporal_features = [
        'front_momentum_5', 'back_momentum_5',
        'front_momentum_10', 'back_momentum_10',
        'front_momentum_20', 'back_momentum_20',
        'odd_gap', 'even_gap', 'large_gap', 'small_gap',
        'prev_front_overlap', 'prev_back_overlap', 'transition_score'
    ]
    
    all_features = base_features + phase1_advanced_features + phase4_temporal_features
    
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
    
    print("📊 特征配置总结:")
    print(f"  Phase 1 基础特征: 11个")
    print(f"  Phase 1 高级特征: 11个")
    print(f"  Phase 1 小计: 22个")
    print(f"  Phase 4 时间动态: 13个")
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
