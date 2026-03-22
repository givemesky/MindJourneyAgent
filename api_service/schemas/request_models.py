# 定义来自 Android 的请求格式 (如 UserMessage, SelectedIDs)

# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

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
    

# --- 画像维度详情 ---
class DimensionDetail(BaseModel):
    label: str
    confidence: float
    is_stable: bool

# --- RAG 推荐项详情 ---
class RecommendationItem(BaseModel):
    name: str
    type: str
    price: float
    description: Optional[str] = None
    recommend_food: Optional[str] = None

# --- 核心对话响应 ---
class ChatResponse(BaseModel):
    status: str
    reply: str
    turn: int
    persona: Dict[str, DimensionDetail]
    recommendations: Optional[List[RecommendationItem]] = None

# --- 阶段 D: 最终行程响应 ---
class PlanResponse(BaseModel):
    status: str = "SUCCESS"
    total_price: float
    rendered_report: str = Field(..., description="Qwen 渲染的最终 Markdown 文本报告")
    # 可选：结构化行程数据，供高级 UI 使用
    itinerary_data: Optional[List[Dict[str, Any]]] = None