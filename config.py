#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一配置管理模块
集中管理路径、凭据、爬虫参数等全局配置

所有模块都应从这里读取配置，禁止在业务文件中重复定义 DB_PATH / HEADERS 等常量。
"""

import os
from pathlib import Path

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DB_PATH = ROOT_DIR / 'lottery.db'
MODELS_DIR = BASE_DIR / 'models'
CHARTS_DIR = BASE_DIR / 'charts'
# 图表产物独立目录，避免与 charts/ 源码包混放（Windows 下图片被占用会导致写入失败）
OUTPUT_DIR = ROOT_DIR / 'output'
LOGS_DIR = BASE_DIR / 'logs'

# ==================== 环境变量加载 ====================
# 同时探测项目目录与上级目录的 .env，解决 .env.example 与代码查找路径不一致的问题
try:
    from dotenv import load_dotenv
    for _p in (BASE_DIR / '.env', ROOT_DIR / '.env'):
        if _p.exists():
            load_dotenv(dotenv_path=str(_p), override=True)
            break
except ImportError:  # 未安装 python-dotenv 时静默降级为纯环境变量
    pass

# ==================== 钉钉配置（纯环境变量） ====================
DINGTALK_TOKEN = os.environ.get('DINGTALK_TOKEN', '')
DINGTALK_SECRET = os.environ.get('DINGTALK_SECRET', '')

# ==================== 爬虫配置 ====================
HTTP_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Connection': 'keep-alive',
}
HTTP_TIMEOUT = 20
HTTP_MAX_RETRIES = 3
# 500.com 实际使用 GBK；gb2312 是其真子集，遇到生僻字会解码失败
HTTP_ENCODING = 'gbk'
# 连续请求间隔(秒)，避免高频打站被封 IP
HTTP_REQUEST_INTERVAL = 1.0

# ==================== 数据库配置 ====================
DB_TIMEOUT = 30

# ==================== 钉钉推送配置 ====================
DINGTALK_TIMEOUT = 10
DINGTALK_MAX_RETRIES = 3
# 钉钉机器人限流：同一机器人每分钟最多 20 条
DINGTALK_RATE_LIMIT = 20
DINGTALK_RATE_WINDOW = 60
