# 业务协调器：核心逻辑封装，负责串联 DPI -> RAG -> Planner

# -*- coding: utf-8 -*-
from typing import Dict, Any, List
from integrated_bridge import DPI_SynPT_Bridge
from RAG.rag_system.rag_interface import run_rag_recommendation
from api_service.manager.session_manager import session_manager, SessionData
from api_service.schemas.response_models import DimensionDetail, RecommendationItem

class MJCoordinator:
    def __init__(self, bridge: DPI_SynPT_Bridge):
        self.bridge = bridge

    async def handle_chat_turn(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """
        处理一轮对话的核心流程：DPI 推理 -> 状态更新 -> (可选) RAG 检索
        """
        # 1. 获取当前会话状态
        session: SessionData = session_manager.get_session(session_id)
        
        # 2. 调用 DPI-SynPT 引擎进行推理和回复生成
        # result 包含: response, persona, decision, metrics, duration
        result = self.bridge.chat_step(user_input, session.turn)
        
        # 3. 提取并格式化画像数据 (DimensionDetail)
        formatted_persona = {}
        for dim, info in result['persona'].items():
            formatted_persona[dim] = DimensionDetail(
                label=info['label'],
                confidence=info['confidence'],
                is_stable=info['is_stable']
            )
        
        # 4. 更新 Session 中的画像状态
        session_manager.update_session(
            session_id, 
            current_persona=result['persona'],
            turn=session.turn + 1
        )

        # 5. 检查是否收敛 (TERMINATE)
        recommendations = None
        action = result['decision']['action']
        
        if action == "TERMINATE":
            # 触发 RAG 检索
            # 注意：这里假设 run_rag_recommendation 返回的是结构化数据或可解析的列表
            # 为了演示，我们模拟将其转化为 List[RecommendationItem]
            raw_rag_data = run_rag_recommendation() 
            recommendations = self._parse_rag_results(raw_rag_data)
            
            # 将候选结果存入 Session，供阶段 C 使用
            session_manager.update_session(session_id, rag_candidates=recommendations)

        # 6. 返回给 Route 层的数据
        return {
            "status": action,
            "reply": result['response'],
            "turn": session.turn, # 返回当前的轮次
            "persona": formatted_persona,
            "recommendations": recommendations
        }

    def _parse_rag_results(self, raw_data: Any) -> List[RecommendationItem]:
        """
        [逻辑占位] 将 RAG 的原始输出转化为 Android 需要的 RecommendationItem 列表
        实际开发中，你需要根据 run_rag_recommendation() 的返回格式进行解析
        """
        # 这里仅为示例：假设 RAG 返回了一个列表或我们需要构造它
        # 如果 RAG 目前只返回大文本，建议修改 RAG 接口返回结构化 List
        items = []
        # 示例模拟数据
        if isinstance(raw_data, str):
            # 如果是字符串，暂时封装进一个 Item，或者在这里做正则解析
            items.append(RecommendationItem(
                name="推荐方案已生成",
                type="Report",
                price=0.0,
                description=raw_data
            ))
        return items