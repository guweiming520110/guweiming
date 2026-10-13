#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
矩阵构建模块
将彩票历史数据构建为 期数×号码 的布尔矩阵
"""

import numpy as np


class LotteryMatrix:
    """彩票矩阵：行=期数(0=最新)，列=号码"""

    def __init__(self, lottery_type='dlt'):
        self.lottery_type = lottery_type
        if lottery_type == 'dlt':
            self.min_num, self.max_num = 1, 35
            self.count = 5
            self.cols = ['front']  # 前区
            self.range_info = {'front': (1, 35, 5)}
        elif lottery_type == 'dlt_back':
            self.min_num, self.max_num = 1, 12
            self.count = 2
            self.range_info = {'back': (1, 12, 2)}
        elif lottery_type == 'ssq':
            self.min_num, self.max_num = 1, 33
            self.count = 6
            self.cols = ['front']
            self.range_info = {'front': (1, 33, 6)}
        elif lottery_type == 'ssq_back':
            self.min_num, self.max_num = 1, 16
            self.count = 1
            self.range_info = {'back': (1, 16, 1)}
        elif lottery_type == 'kl8':
            self.min_num, self.max_num = 1, 80
            self.count = 20
            self.range_info = {'kl8': (1, 80, 20)}
        else:
            raise ValueError(f"Unknown type: {lottery_type}")

        self.num_range = self.max_num - self.min_num + 1
        # 用空矩阵占位（而非 None），避免下标/属性访问被类型检查器判定为对 None 操作；
        # build_* 方法会填充真实数据，未构建时按空矩阵安全处理
        self.matrix = np.zeros((0, self.max_num + 1), dtype=bool)
        self.periods = np.array([], dtype=object)
        self.n_periods = 0

    def build_from_dlt(self, df, num_periods=None):
        """从大乐透DataFrame构建矩阵"""
        data = df.copy()
        if num_periods:
            data = data.tail(num_periods)
        # 数据源为 period ASC（最旧在前），翻转为最新在前，保证 matrix[0]=最新一期
        data = data.iloc[::-1].reset_index(drop=True)

        self.n_periods = len(data)
        self.matrix = np.zeros((self.n_periods, self.max_num + 1), dtype=bool)
        self.periods = data['period'].astype(str).values

        for i in range(self.n_periods):
            # H4 修复: 用 self.count 驱动列数，替代硬编码 range(1,6)，与配置解耦
            for col in [f'front_{k}' for k in range(1, self.count + 1)]:
                if col not in data.columns:
                    continue
                n = int(data.iloc[i][col])
                if 1 <= n <= self.max_num:
                    self.matrix[i][n] = True

        return self

    def build_from_dlt_back(self, df, num_periods=None):
        """从大乐透后区构建矩阵"""
        data = df.copy()
        if num_periods:
            data = data.tail(num_periods)
        # 数据源为 period ASC（最旧在前），翻转为最新在前，保证 matrix[0]=最新一期
        data = data.iloc[::-1].reset_index(drop=True)

        self.n_periods = len(data)
        self.matrix = np.zeros((self.n_periods, self.max_num + 1), dtype=bool)
        self.periods = data['period'].astype(str).values

        for i in range(self.n_periods):
            # H4 修复: 用 self.count 驱动列数，替代硬编码 range(1,3)
            for col in [f'back_{k}' for k in range(1, self.count + 1)]:
                if col not in data.columns:
                    continue
                n = int(data.iloc[i][col])
                if 1 <= n <= self.max_num:
                    self.matrix[i][n] = True
        return self

    def build_from_kl8(self, df, num_periods=None):
        """从快乐8DataFrame构建矩阵（需含num_list列）"""
        data = df.copy()
        if num_periods:
            data = data.tail(num_periods)
        # 数据源为 period ASC（最旧在前），翻转为最新在前，保证 matrix[0]=最新一期
        data = data.iloc[::-1].reset_index(drop=True)

        self.n_periods = len(data)
        self.matrix = np.zeros((self.n_periods, self.max_num + 1), dtype=bool)
        self.periods = data['period'].astype(str).values

        for i in range(self.n_periods):
            if 'num_list' in data.columns:
                nums = data.iloc[i]['num_list']
            else:
                nums_str = data.iloc[i]['numbers']
                nums = [int(x) for x in str(nums_str).split(',')]
            for n in nums:
                if 1 <= n <= self.max_num:
                    self.matrix[i][n] = True
        return self

    def build_from_ssq(self, df, num_periods=None):
        """从双色球DataFrame构建前区(红球)矩阵"""
        data = df.copy()
        if num_periods:
            data = data.tail(num_periods)
        # 数据源为 period ASC（最旧在前），翻转为最新在前，保证 matrix[0]=最新一期
        data = data.iloc[::-1].reset_index(drop=True)

        self.n_periods = len(data)
        self.matrix = np.zeros((self.n_periods, self.max_num + 1), dtype=bool)
        self.periods = data['period'].astype(str).values

        for i in range(self.n_periods):
            # H4 修复: 用 self.count 驱动列数，替代硬编码 range(1,7)
            for col in [f'front_{k}' for k in range(1, self.count + 1)]:
                if col not in data.columns:
                    continue
                n = int(data.iloc[i][col])
                if 1 <= n <= self.max_num:
                    self.matrix[i][n] = True
        return self

    def build_from_ssq_back(self, df, num_periods=None):
        """从双色球DataFrame构建后区(蓝球)矩阵"""
        data = df.copy()
        if num_periods:
            data = data.tail(num_periods)
        # 数据源为 period ASC（最旧在前），翻转为最新在前，保证 matrix[0]=最新一期
        data = data.iloc[::-1].reset_index(drop=True)

        self.n_periods = len(data)
        self.matrix = np.zeros((self.n_periods, self.max_num + 1), dtype=bool)
        self.periods = data['period'].astype(str).values

        for i in range(self.n_periods):
            # H4 修复: 用 self.count 驱动列数，替代只取 back_1 的硬编码
            for col in [f'back_{k}' for k in range(1, self.count + 1)]:
                if col not in data.columns:
                    continue
                n = int(data.iloc[i][col])
                if 1 <= n <= self.max_num:
                    self.matrix[i][n] = True
        return self

    def get_latest_row(self):
        """获取最新一期（第0行）"""
        return self.matrix[0].copy()

    def has_number(self, period_idx, number):
        """检查指定期数是否开出了某号码"""
        if period_idx < 0 or period_idx >= self.n_periods:
            return False
        if number < 1 or number > self.max_num:
            return False
        return self.matrix[period_idx][number]

    def count_in_period(self, period_idx):
        """统计某期开出了多少个号码"""
        if period_idx < 0 or period_idx >= self.n_periods:
            return 0
        return int(np.sum(self.matrix[period_idx]))

    def add_candidate(self, candidate_num):
        """
        在最新一期添加一个候选号码，返回新的矩阵副本
        不修改原始矩阵
        """
        if not (1 <= candidate_num <= self.max_num):
            return None
        # C1 修复: 空矩阵(未训练)时退化为不可加，避免 matrix[0] 越界崩溃
        if self.n_periods == 0:
            return None
        new_matrix = self.matrix.copy()
        new_matrix[0][candidate_num] = True
        return new_matrix

    def add_candidate_inplace(self, candidate_num):
        """
        原地添加候选号码到第0行，返回是否成功
        性能优化：避免每个候选号码都复制整个矩阵
        """
        if not (1 <= candidate_num <= self.max_num):
            return False
        # C1 修复: 空矩阵(未训练/无数据)时无法在第0行添加，直接返回失败。
        # 此前会 IndexError: index 0 out of bounds for axis 0 with size 0，
        # 导致空库场景下 pattern_learner.predict() 直接崩溃。
        if self.n_periods == 0:
            return False
        if self.matrix[0][candidate_num]:
            return False  # 已存在
        self.matrix[0][candidate_num] = True
        return True

    def remove_candidate_inplace(self, candidate_num):
        """移除第0行的候选号码（与 add_candidate_inplace 配对使用）"""
        if 1 <= candidate_num <= self.max_num and self.n_periods > 0:
            self.matrix[0][candidate_num] = False

    def get_context(self, num_periods=None):
        """获取上下文矩阵片段"""
        if num_periods is None:
            return self.matrix
        n = min(num_periods, self.n_periods)
        return self.matrix[:n]

    def __str__(self):
        return (f"LotteryMatrix({self.lottery_type}): "
                f"{self.n_periods} periods, "
                f"numbers 1-{self.max_num}")
