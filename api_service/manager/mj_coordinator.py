# -*- coding: utf-8 -*-
import os
import json
import re
from typing import Dict, Any, List, Optional
from integrated_bridge import DPI_SynPT_Bridge
from RAG.rag_system.rag_interface import run_rag_recommendation
from RAG.rag_system.planner import TravelPlanner 
from api_service.manager.session_manager import session_manager, SessionData
from api_service.schemas.response_models import DimensionDetail, RecommendationItem
# 【关键导入】导入模型提供者
from model_provider import SharedModelProvider

class MJCoordinator:
    def __init__(self, bridge: DPI_SynPT_Bridge):
        self.bridge = bridge
        self.root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    async def handle_chat_turn(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """阶段 A/B: 对话推理"""
        session: SessionData = session_manager.get_session(session_id)
        recommendations = None 
        result = self.bridge.chat_step(user_input, session.turn)
        action = result['decision']['action']
        assistant_reply = result['response'] 

        if action == "TERMINATE":
            if hasattr(self.bridge, 'finalize'):
                self.bridge.finalize() 

            assistant_reply = (
                "好的，我已经完全掌握了您的旅行偏好！✨\n\n"
                "根据您的需求，我从知识图谱中为您挑选了一些最匹配的景点和餐厅。"
                "请在下方**点击勾选**您心仪的选项，选好后告诉我，我将为您生成精准的深度行程规划。"
            )
            raw_rag_list = run_rag_recommendation() 
            recommendations = self._parse_rag_results(raw_rag_list)
            session_manager.update_session(session_id, rag_candidates=raw_rag_list)

        formatted_persona = {}
        for dim, info in result['persona'].items():
            formatted_persona[dim] = DimensionDetail(
                label=info['label'],
                confidence=info['confidence'],
                is_stable=info['is_stable']
            )

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

    async def handle_selection(self, session_id: str, attractions: List[str], restaurants: List[str]):
        """阶段 C: 处理勾选"""
        session_manager.update_session(
            session_id,
            user_selections_attractions=attractions,
            user_selections_restaurants=restaurants
        )

    async def handle_planning(self, session_id: str) -> str:
        """阶段 D: 最终规划生成"""
        session: SessionData = session_manager.get_session(session_id)
        
        # 1. 加载约束日志
        constraints_path = os.path.join(self.root_path, "dpi_me/logs/UserProfile/constraints_log.json")
        all_constraint_strings = []
        if os.path.exists(constraints_path):
            try:
                with open(constraints_path, 'r', encoding='utf-8') as f:
                    turn_data_list = json.load(f)
                    # 遍历每一个轮次，把所有的 constraints 字符串提取到一个大列表中
                    for turn in turn_data_list:
                        if isinstance(turn, dict) and "constraints" in turn:
                            all_constraint_strings.extend(turn["constraints"])
            except Exception as e:
                print(f"⚠️ 读取日志文件失败: {e}")

        # 2. 提取基础约束 (从后往前找，以最后的修正为准)
        # 我们倒序遍历，因为后面的轮次通常包含更准确的信息
        target_city = self._smart_extract(all_constraint_strings, "Target City")
        day_count = self._smart_extract(all_constraint_strings, "Day Count")
        people_count = self._smart_extract(all_constraint_strings, "People Count")

        # 3. 构造 Planner 数据包
        user_data_for_planner = {
            "basic_info": {
                "target_city": target_city or "成都", # 默认成都
                "people_count": int(people_count) if people_count and people_count.isdigit() else 1,
                "day_count": int(day_count) if day_count and day_count.isdigit() else 3
            },
            "hard_constraints": {
                "must_visit_attractions": getattr(session, 'user_selections_attractions', []),
                "must_eat_restaurants": getattr(session, 'user_selections_restaurants', []),
                "min_activities_count": 5
            },
            "soft_constraints_persona": {
                # 这里的 key 必须对应 planner.py 里的解析逻辑
                "consumption_level": session.current_persona.get('Consumption', {}).get('label', 'Budget'),
                "interest_preference": session.current_persona.get('Interest', {}).get('label', 'Culture'),
                "travel_pace": session.current_persona.get('Pace', {}).get('label', 'Relaxed')
            }
        }

        print(f"🚀 [Coordinator] 最终规划数据包: {user_data_for_planner}")

        # 4. 运行规划器
        shared_res = SharedModelProvider.get_instance()
        planner = TravelPlanner(
            user_data=user_data_for_planner,
            model=shared_res.model, 
            tokenizer=shared_res.tokenizer
        )

        return planner.run()

    def _smart_extract(self, strings: List[str], key: str) -> Optional[str]:
        """
        专门针对 "1. Key == \"Value\"" 这种格式的提取器
        从列表末尾向前搜索，返回第一个匹配到的值
        """
        # 倒序遍历，获取最新的约束
        for s in reversed(strings):
            if key.lower() in s.lower():
                # 使用正则匹配 == 之后的内容
                # 适配情况: "1. Target City == \"成都\"" 或 "Day Count == 6"
                match = re.search(r'==\s*"?\'?([^"\'\s;]+)"?\'?', s)
                if match:
                    val = match.group(1).strip()
                    # 特殊处理：如果是数字约束如 ">= 2"，我们只想要那个数字
                    digit_match = re.search(r'(\d+)', val)
                    if key in ["Day Count", "People Count"] and digit_match:
                        return digit_match.group(1)
                    return val
        return None

    def _parse_rag_results(self, raw_data: List[Dict[str, Any]]) -> List[RecommendationItem]:
        items = []
        if not raw_data: return items
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
            except: continue
        return items