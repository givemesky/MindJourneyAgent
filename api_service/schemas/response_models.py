# 定义返回给 Android 的数据格式 (如 ChatResponse, ItineraryPlan)

# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- 画像维度详情 ---
class DimensionDetail(BaseModel):
    label: str = Field(..., description="当前推断的标签，如 Budget/Culture/Fast")
    confidence: float = Field(..., description="置信度 (0.0-1.0)")
    is_stable: bool = Field(..., description="该维度是否已收敛")

# --- RAG 推荐项详情 ---
class RecommendationItem(BaseModel):
    name: str = Field(..., description="景点或餐厅名称")
    type: str = Field(..., description="分类，如 人文景观/江浙菜")
    price: float = Field(0.0, description="价格或人均消费")
    description: Optional[str] = Field(None, description="特色介绍")
    recommend_food: Optional[str] = Field(None, description="如果是餐厅，其招牌菜")

# --- 核心对话响应 (用于 /v1/plan) ---
class ChatResponse(BaseModel):
    status: str = Field(..., description="当前动作状态: ASK(继续提问) 或 TERMINATE(已收敛)")
    reply: str = Field(..., description="助手的回复文本")
    turn: int = Field(..., description="当前对话轮次")
    
    # 用户画像监控面板数据
    persona: Dict[str, DimensionDetail] = Field(..., description="用户画像三个维度的实时状态")
    
    # 只有当 status 为 TERMINATE 时，该字段才会有数据
    recommendations: Optional[List[RecommendationItem]] = Field(None, description="RAG 检索出的推荐列表")

# --- 最终行程响应 ---
class ItineraryDay(BaseModel):
    day: int
    morning: Optional[Dict[str, Any]]
    lunch: Optional[Dict[str, Any]]
    afternoon: Optional[Dict[str, Any]]
    dinner: Optional[Dict[str, Any]]
    accommodation: Optional[Dict[str, Any]]

class PlanResponse(BaseModel):
    # 增加 status 字段（因为你路由里返回了这个）
    status: str = Field("SUCCESS", description="状态")
    
    total_price: float = Field(0.0, description="预估总价")
    
    rendered_report: str = Field(..., description="Qwen 渲染的最终 Markdown 文本报告")

    # 【关键修改】：改为 Optional 并设置默认值为 None
    strategy_name: Optional[str] = Field(None, description="规划策略名称")
    
    # 【关键修改】：改为 Optional 并设置默认值为 None
    itinerary: Optional[List[ItineraryDay]] = Field(None, description="每日行程详情")