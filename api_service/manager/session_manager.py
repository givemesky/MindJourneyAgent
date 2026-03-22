# 会话管理器：在内存中维护每个用户的 Persona、约束和对话历史

# -*- coding: utf-8 -*-
from typing import Dict, Any, Optional, List
import threading
from pydantic import BaseModel, Field

# --- 定义 Session 内部存储的数据结构 ---
class SessionData(BaseModel):
    session_id: str
    turn: int = 1
    # 存储最新的画像
    current_persona: Dict[str, Any] = Field(default_factory=dict)
    # 存储识别到的硬约束
    constraints: List[str] = Field(default_factory=list)
    # 存储 RAG 检索出的候选列表 (List[Dict])
    # 阶段 B 写入，阶段 C 读取
    rag_candidates: Optional[List[Dict[str, Any]]] = None
    # 存储用户最终勾选的名称列表
    # 阶段 C 写入，阶段 D 读取
    user_selections: List[str] = Field(default_factory=list)
    # 聊天历史
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

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
        """获取 Session，线程安全"""
        with self._lock:
            if session_id not in self.sessions:
                print(f"🆕 [Session] 为用户 {session_id} 创建新会话")
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
                        # print(f"📝 [Session] 更新 {session_id} 的 {key}")

    def clear_session(self, session_id: str):
        """清除会话"""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                print(f"🗑️ [Session] 会话 {session_id} 已销毁")

# 全局单例对象
session_manager = SessionManager()