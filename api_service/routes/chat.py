# 阶段 A/B：处理对话、推理偏好、触发 RAG 推荐

# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, HTTPException
from api_service.schemas.request_models import ChatRequest
from api_service.schemas.response_models import ChatResponse
from api_service.manager.mj_coordinator import MJCoordinator
from api_service.dependencies import get_bridge

# 创建路由，前缀在 main.py 中统一挂载
router = APIRouter()

@router.post("/plan", response_model=ChatResponse)
async def chat_interaction(
    request: ChatRequest, 
    bridge=Depends(get_bridge)
):
    """
    阶段 A/B: DPI 动态对话接口。
    Android 端每发送一句话，调用一次此接口。
    URL: http://IP:8000/v1/plan
    """
    try:
        # 1. 初始化协调器
        coordinator = MJCoordinator(bridge)
        
        # 2. 处理业务逻辑
        result = await coordinator.handle_chat_turn(
            session_id=request.session_id,
            user_input=request.user_input
        )
        
        # 3. 返回响应 (FastAPI 自动根据 ChatResponse 模型进行序列化)
        return ChatResponse(
            status=result["status"],
            reply=result["reply"],
            turn=result["turn"],
            persona=result["persona"],
            recommendations=result["recommendations"]
        )
        
    except Exception as e:
        # 详细记录错误日志
        print(f"❌ [Chat Route Error] {str(e)}")
        raise HTTPException(status_code=500, detail=f"服务器内部推理错误: {str(e)}")