# 定义来自 Android 的请求格式 (如 UserMessage, SelectedIDs)

# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    """阶段 A/B: 动态对话请求"""
    session_id: str = Field(..., description="用户唯一会话ID")
    user_input: str = Field(..., description="用户输入的聊天文本")

class SelectionRequest(BaseModel):
    """阶段 C: 用户选择景点/餐厅请求"""
    session_id: str = Field(..., description="用户唯一会话ID")
    selected_attractions: List[str] = Field(..., description="用户选中的景点名称列表")
    selected_restaurants: List[str] = Field(..., description="用户选中的餐厅名称列表")

class PlanGenerationRequest(BaseModel):
    """阶段 D: 请求生成最终行程规划"""
    session_id: str = Field(..., description="用户唯一会话ID")