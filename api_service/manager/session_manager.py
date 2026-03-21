# 会话管理器：在内存中维护每个用户的 Persona、约束和对话历史

# -*- coding: utf-8 -*-
from typing import Dict, Any, Optional, List
import threading
from pydantic import BaseModel

# --- 定义 Session 内部存储的数据结构 ---
class SessionData(BaseModel):
    session_id: str
    turn: int = 1
    # 存储 DPI 推理出的最新画像
    current_persona: Dict[str, Any] = {}
    # 存储识别到的硬约束（如：苏州、3天、2500元）
    constraints: List[str] = []
    # 存储 RAG 检索出的候选列表（用于阶段 C 让用户勾选）
    rag_candidates: Optional[List[Dict[str, Any]]] = None
    # 存储对话历史，用于大模型上下文（可选）
    chat_history: List[Dict[str, str]] = []

class SessionManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SessionManager, cls).__new__(cls)
                cls._instance.sessions: Dict[str, SessionData] = {}
        return cls._instance

    def get_session(self, session_id: str) -> SessionData:
        """获取 Session，如果没有则创建一个新的"""
        with self._lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionData(session_id=session_id)
            return self.sessions[session_id]

    def update_session(self, session_id: str, **kwargs):
        """更新 Session 中的字段"""
        with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                for key, value in kwargs.items():
                    if hasattr(session, key):
                        setattr(session, key, value)

    def clear_session(self, session_id: str):
        """删除或重置 Session"""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]

# 全局单例对象
session_manager = SessionManager()