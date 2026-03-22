# 阶段 C：接收用户勾选的景点/餐厅列表

# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, HTTPException
from api_service.schemas.request_models import SelectionRequest
from api_service.manager.mj_coordinator import MJCoordinator
from api_service.dependencies import get_bridge

router = APIRouter()

@router.post("/select")
async def handle_selection(
    request: SelectionRequest, 
    bridge=Depends(get_bridge)
):
    """
    阶段 C: 接收用户在前端勾选的卡片
    URL: http://IP:8000/v1/select
    """
    try:
        coordinator = MJCoordinator(bridge)
        # 将勾选数据存入 session
        await coordinator.handle_selection(
            session_id=request.session_id,
            attractions=request.selected_attractions,
            restaurants=request.selected_restaurants
        )
        return {"status": "SUCCESS", "message": "已记录您的选择"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))