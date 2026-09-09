#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公共HTTP请求工具模块
提供带指数退避重试的 GET 请求，消除各爬虫文件中的重复代码
"""

import time

import requests

from config import HTTP_HEADERS, HTTP_TIMEOUT, HTTP_MAX_RETRIES, HTTP_ENCODING, HTTP_REQUEST_INTERVAL
from logger import log


def get_with_retry(url, max_retries=HTTP_MAX_RETRIES, timeout=HTTP_TIMEOUT,
                   headers=None, encoding=HTTP_ENCODING, interval=HTTP_REQUEST_INTERVAL):
    """
    带指数退避重试的 GET 请求

    参数:
        url: 请求URL
        max_retries: 最大重试次数
        timeout: 超时时间(秒)
        headers: 自定义请求头(默认使用全局配置)
        encoding: 响应编码，None 表示不覆盖(由 requests 自动推断)
        interval: 成功后到下一次请求的建议间隔(秒)，用于调用方限流

    返回:
        requests.Response 或 None(全部失败时)
    """
    req_headers = headers or HTTP_HEADERS
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=req_headers, timeout=timeout)
            resp.raise_for_status()
            if encoding:
                resp.encoding = encoding
            return resp
        except Exception as e:
            wait = 2 ** attempt
            log.warning("请求失败(第%d/%d次): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(wait)
    return None


def sleep_interval(seconds=HTTP_REQUEST_INTERVAL):
    """连续请求之间的节流，避免高频打站被封 IP"""
    if seconds and seconds > 0:
        time.sleep(seconds)
