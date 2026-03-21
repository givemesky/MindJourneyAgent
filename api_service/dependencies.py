# 依赖管理：全局共享模型实例 (SharedModelProvider) 的注入

# -*- coding: utf-8 -*-
import sys
import os
from typing import Generator

# 确保能找到项目根目录下的 bridge
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from integrated_bridge import DPI_SynPT_Bridge
from api_service.manager.session_manager import session_manager

# --- 全局单例模型实例 ---
_global_bridge = None

def get_bridge() -> DPI_SynPT_Bridge:
    """
    依赖注入函数：获取 DPI 推理桥接器实例。
    第一次调用时会初始化模型（耗时较长）。
    """
    global _global_bridge
    if _global_bridge is None:
        print("🚀 [Dependencies] 正在初始化全局 DPI 桥接器 (Qwen-7B)...")
        _global_bridge = DPI_SynPT_Bridge()
        print("✅ [Dependencies] DPI 桥接器已就绪。")
    return _global_bridge

def get_session_manager():
    """
    依赖注入函数：获取会话管理器。
    """
    return session_manager

# --- 预加载函数 ---
def preload_models():
    """
    在 FastAPI 启动时手动触发，避免第一个用户请求时等待太久。
    """
    get_bridge()