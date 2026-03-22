# 业务协调器：核心逻辑封装，负责串联 DPI -> RAG -> Planner

# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional
from integrated_bridge import DPI_SynPT_Bridge
from RAG.rag_system.rag_interface import run_rag_recommendation
from api_service.manager.session_manager import session_manager, SessionData
from api_service.schemas.response_models import DimensionDetail, RecommendationItem

class MJCoordinator:
    def __init__(self, bridge: DPI_SynPT_Bridge):
        self.bridge = bridge

    async def handle_chat_turn(self, session_id: str, user_input: str) -> Dict[str, Any]:
        # 1. 获取 Session
        session: SessionData = session_manager.get_session(session_id)
        recommendations = None 

        # 2. 调用推理引擎
        # 此时最新的画像和约束已经存在于 self.bridge 的内存中了
        result = self.bridge.chat_step(user_input, session.turn)

        # 3. 提取状态和原始回复
        action = result['decision']['action']
        assistant_reply = result['response'] 

        # ---------------------------------------------------------
        # 【核心修改点】：处理收敛状态
        # ---------------------------------------------------------
        if action == "TERMINATE":
            print(f"🎯 [Coordinator] 会话 {session_id} 已收敛。准备生成方案...")

            # 【第一步：先落盘！】
            # 必须在 RAG 运行前调用 finalize，将内存中最新的画像轨迹和约束写入磁盘文件
            if hasattr(self.bridge, 'finalize'):
                print("💾 [Log] 正在将最新推断结果写入磁盘以供 RAG 引擎读取...")
                self.bridge.finalize()

            # 【第二步：替换引导语】
            assistant_reply = (
                "好的，我已经完全掌握了您的旅行偏好！✨\n\n"
                "根据您的需求，我从知识图谱中为您挑选了一些最匹配的景点和餐厅。"
                "请在下方**点击勾选**您心仪的选项，选好后告诉我，我将为您生成精准的深度行程规划。"
            )

            # 【第三步：执行 RAG 检索】
            # 此时 run_rag_recommendation 去读文件，拿到的就是刚存进去的热乎数据
            print("🚀 [RAG] 正在检索匹配的景点与餐厅...")
            raw_rag_list = run_rag_recommendation() 
            recommendations = self._parse_rag_results(raw_rag_list)
            
            # 将候选结果同步到 Session
            session_manager.update_session(session_id, rag_candidates=raw_rag_list)
        # ---------------------------------------------------------

        # 4. 格式化画像 (保持不变)
        formatted_persona = {}
        for dim, info in result['persona'].items():
            formatted_persona[dim] = DimensionDetail(
                label=info['label'],
                confidence=info['confidence'],
                is_stable=info['is_stable']
            )

        # 5. 更新 Session 轮次
        session_manager.update_session(
            session_id, 
            current_persona=result['persona'],
            turn=session.turn + 1
        )

        return {
            "status": action,
            "reply": assistant_reply,
            "turn": session.turn,
            "persona": formatted_persona,
            "recommendations": recommendations
        }
        
    def _parse_rag_results(self, raw_data: List[Dict[str, Any]]) -> List[RecommendationItem]:
        """将 RAG 返回的字典列表转换为 Pydantic 对象列表"""
        items = []
        if not raw_data:
            return items
            
        for data in raw_data:
            try:
                item = RecommendationItem(
                    name=data.get("name", "未知地点"),
                    type=data.get("type", "其他"),
                    price=data.get("price", 0.0),
                    description=data.get("description", ""),
                    recommend_food=data.get("recommend_food")
                )
                items.append(item)
            except Exception as e:
                print(f"⚠️ 解析 RAG 数据项失败: {e}")
                
        return items