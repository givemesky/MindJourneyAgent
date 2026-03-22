# 阶段 D：调用 Planner 生成最终行程报表

# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends, HTTPException
from api_service.schemas.request_models import PlanGenerationRequest
from api_service.schemas.response_models import PlanResponse
from api_service.manager.mj_coordinator import MJCoordinator
from api_service.dependencies import get_bridge
import time

router = APIRouter()

@router.post("/generate_plan", response_model=PlanResponse)
async def generate_final_plan(
    request: PlanGenerationRequest, 
    bridge=Depends(get_bridge)
):
    """
    阶段 D: 根据已识别画像和用户勾选，生成最终三套行程方案
    URL: http://IP:8000/v1/generate_plan
    """
    start_time = time.time()
    try:
        coordinator = MJCoordinator(bridge)
        
        # 核心逻辑：缝合数据 -> 调用 Planner 算法 -> 调用 Qwen 渲染
        report_text = await coordinator.handle_planning(request.session_id)
        
        duration = round(time.time() - start_time, 2)
        print(f"✅ [Route] 行程规划生成完毕，耗时: {duration}s")
        
        return PlanResponse(
            status="SUCCESS",
            total_price=0.0,  # 总价已包含在 report 文本中，此处可设为 0
            rendered_report=report_text
        )
        
    except FileNotFoundError as e:
        # 通常是对话还没收敛就点生成，找不到日志
        raise HTTPException(status_code=400, detail=f"规划失败：{str(e)}")
    except Exception as e:
        print(f"❌ [Route Error] 规划逻辑崩溃: {str(e)}")
        raise HTTPException(status_code=500, detail=f"内部规划引擎错误: {str(e)}")