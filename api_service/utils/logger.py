# API 访问日志记录

# -*- coding: utf-8 -*-
import logging
import sys

# 创建一个格式化器
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 创建控制台处理器
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

# 设置 logger
logger = logging.getLogger("MindJourney-API")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# 测试日志
# logger.info("Logger 模块初始化成功")