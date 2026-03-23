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
        session: SessionData = session_manager.get_session(session_id)
        
        # 1. 获取基础日志数据
        constraints_path = os.path.join(self.root_path, "dpi_me/logs/UserProfile/constraints_log.json")
        all_constraint_strings = []
        if os.path.exists(constraints_path):
            with open(constraints_path, 'r', encoding='utf-8') as f:
                turn_data_list = json.load(f)
                for turn in turn_data_list:
                    all_constraint_strings.extend(turn.get("constraints", []))

        # 2. 提取城市/天数/人数
        target_city = self._smart_extract(all_constraint_strings, "Target City")
        day_count = self._smart_extract(all_constraint_strings, "Day Count")
        people_count = self._smart_extract(all_constraint_strings, "People Count")

        # 3. 【核心修复】：合并对话识别出的景点 和 用户卡片勾选的景点
        # A. 从日志字符串中提取对话里提到的景点 (基于正则)
        chat_identified_attractions = []
        for s in all_constraint_strings:
            # 匹配 Position == "xxx"
            attr_match = re.search(r'Position\s*==\s*"?([^"\'\s;]+)"?', s)
            if attr_match:
                chat_identified_attractions.append(attr_match.group(1))

        # B. 从 Session 获取用户勾选的卡片
        user_clicked_attractions = session.user_selections_attractions
        user_clicked_restaurants = session.user_selections_restaurants

        # C. 取并集，并去重
        final_attractions = list(set(chat_identified_attractions + user_clicked_attractions))
        # 餐厅同理，如果日志里有就合并，没有就只用选中的
        final_restaurants = list(set(user_clicked_restaurants))

        # 4. 重新构造数据包
        user_data_for_planner = {
            "basic_info": {
                "target_city": target_city or "成都",
                "people_count": int(people_count) if people_count else 1,
                "day_count": int(day_count) if day_count else 3
            },
            "hard_constraints": {
                "must_visit_attractions": final_attractions, # 使用合并后的列表
                "must_eat_restaurants": final_restaurants,
                "min_activities_count": 5
            },
            "soft_constraints_persona": {
                "consumption_level": session.current_persona.get('Consumption', {}).get('label', 'Economy'),
                "interest_preference": session.current_persona.get('Interest', {}).get('label', 'Culture'),
                "travel_pace": session.current_persona.get('Pace', {}).get('label', 'Relaxed')
            }
        }

        print(f"🚀 [Coordinator] 规划器输入数据确认(含手动勾选): {user_data_for_planner}")

        # 5. 调用规划器 (保持不变)
        shared_res = SharedModelProvider.get_instance()
        planner = TravelPlanner(user_data=user_data_for_planner, model=shared_res.model, tokenizer=shared_res.tokenizer)
        return planner.run()
    
    '''
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
        
        print(f"🔍 [Debug] 提取结果: City={target_city}, Days={day_count}, People={people_count}")

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
        
        user_data_for_planner = {
            "basic_info": {
                "target_city": target_city or "南京",
                "people_count": int(people_count) if people_count else 1,
                "day_count": int(day_count) if day_count else 3
            },
            "hard_constraints": {
                "must_visit_attractions": getattr(session, 'user_selections_attractions', []),
                "must_eat_restaurants": getattr(session, 'user_selections_restaurants', []),
                "min_activities_count": 5
            },
            "soft_constraints_persona": {
                "consumption_level": session.current_persona.get('Consumption', {}).get('label', 'Economy'),
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
    ''' 
    
    def _smart_extract(self, strings: List[str], key: str) -> Optional[str]:
        """
        定制化提取器：
        1. Target City: 正向搜索 (strings)，优先保留第一印象。
        2. Days/People: 逆向搜索 (reversed)，以最后一次修正为准。
        3. 强制过滤占位符。
        """
        # 定义需要过滤的无效占位符
        placeholders = ["目标城市", "目的地城市", "当前城市", "目标", "当前", "city", "target", "destination", "目的地"]

        # --- 核心逻辑：决定搜索方向 ---
        # 如果找城市，就正向找；如果是数字约束，就逆向找最新修正
        search_queue = strings if key == "Target City" else reversed(strings)
        direction = "正向" if key == "Target City" else "逆向"

        for s in search_queue:
            parts = s.split('==')
            if len(parts) < 2:
                continue
            
            left_side = parts[0].lower()
            right_side = parts[1].strip()
            
            if key.lower() in left_side:
                if key in ["Day Count", "People Count"]:
                    # 针对数字类：提取第一个连续数字
                    num_match = re.search(r'(\d+)', right_side)
                    if num_match:
                        return num_match.group(1)
                else:
                    # 针对城市名类：提取引号内的具体地名
                    # 匹配双引号或单引号内的非空字符
                    val_match = re.search(r'["\']?([^"\'\s;]+)["\']?', right_side)
                    if val_match:
                        val = val_match.group(1)
                        
                        # 检查是否是无意义的占位符
                        is_placeholder = any(p in val for p in placeholders)
                        if is_placeholder:
                            # 如果是城市名且是占位符，跳过，看下一条
                            continue
                        
                        # 找到了一个具体的、非占位符的名称
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