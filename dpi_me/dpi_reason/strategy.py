# dpi_me/dpi_reason/strategy.py [决策大脑]：
# 决策逻辑层：基于信息熵指标，决定“追问”或“结束”
# 功能：判断系统现在该干什么（追问某个维度？还是结束对话？）。

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Optimized Decision Strategy Module (差异化熵值决策逻辑)
"""

# -*- coding: utf-8 -*-
import numpy as np

class DialogueStrategy:
    def __init__(self, max_turns=6):
        # 1. 差异化阈值 (基于最大概率值 Confidence)
        # 只要最大标签的概率超过这个数，就认为该维度“收敛”了
        self.thresholds = {
            "Consumption": 0.40,  # 3分类：1/3=33%，45%已有明显偏向
            "Interest": 0.3,     # 5分类：1/5=20%，38%已经是绝对优势
            "Pace": 0.62          # 2分类：1/2=50%，要求62%以上才稳定
        }
        
        self.max_turns = max_turns
        # 优先级：兴趣通常最难确定，建议先攻克
        self.dim_priority = ["Interest", "Consumption", "Pace"]
        self.asked_history = []

    def get_decision(self, metrics, turn_id):
        """
        做出下一步动作决策
        """
        # 1. 强制终止逻辑
        if turn_id >= self.max_turns:
            return {"action": "TERMINATE", "target_dim": None, "reason": "达到最大对话轮数"}

        # 2. 筛选未达标的维度
        candidate_dims = []
        for dim in self.dim_priority:
            # 【注意】这里必须确保 evaluator.py 传进来的指标是最大概率
            conf = metrics.get(f"{dim}_confidence", 0.0)
            threshold = self.thresholds.get(dim, 0.6) # 获取该维度的专属阈值
            
            if conf < threshold:
                gap = threshold - conf
                candidate_dims.append({
                    "dim": dim,
                    "gap": gap,
                    "conf": conf
                })

        # 3. 如果所有维度都达标了，结束对话
        if not candidate_dims:
            return {"action": "TERMINATE", "target_dim": None, "reason": "所有维度已收敛"}

        # 4. 排序逻辑：优先选缺口最大的
        candidate_dims.sort(key=lambda x: x['gap'], reverse=True)
        
        # 5. 【核心改进】防复读：如果该维度上一轮刚刚问过，尝试切换到另一个未达标维度
        target = candidate_dims[0]
        if len(self.asked_history) > 0 and self.asked_history[-1] == target['dim']:
            if len(candidate_dims) > 1:
                target = candidate_dims[1] # 切换到第二紧迫的维度
                reason = f"换个话题（{target['dim']}），避免连续追问"
            else:
                reason = f"虽然问过，但没有其他未收敛维度，继续深化 {target['dim']}"
        else:
            reason = f"{target['dim']} 缺口最大 ({target['gap']:.2f})"

        self.asked_history.append(target['dim'])
        
        return {
            "action": "ASK",
            "target_dim": target['dim'],
            "reason": reason
        }

    def get_prompt_hint(self, decision, persona):
        """
        根据决策生成给小助手的“贴条”提示
        """
        if decision["action"] == "TERMINATE":
            return "【系统指令】用户的偏好已经非常清晰了。请不要再提问，直接结合目前已知的预算、兴趣和节奏，给出一份具体的行程建议。"

        target = decision["target_dim"]
        
        # 获取当前最可能的标签（即使还没达到阈值）
        current_guess = persona.get_top_label(target)
        current_conf = persona.get_confidence(target)

        # 灵活引导：根据当前的置信度，告诉 LLM 是“盲问”还是“确认”
        if current_conf > 0.3:
            # 已经有倾向了，让助手去“确认”而不是重新问
            base_hint = f"目前系统初步推测用户偏好是【{current_guess}】，请在回复中通过‘确认’或‘顺着话头说’的方式来验证，不要机械提问。"
        else:
            base_hint = f"目前对用户在【{target}】方面的意图完全不清楚，请主动发起提问。"

        hints = {
            "Consumption": f"{base_hint} 重点关注预算（经济/豪华/具体金额）。",
            "Interest": f"{base_hint} 重点关注景点偏好（自然/人文/购物等）。",
            "Pace": f"{base_hint} 重点关注旅行节奏（紧凑打卡/悠闲慢游）。"
        }
        
        # 增加防复读强指令
        anti_repeat = " 注意：请使用自然的、口语化的表达，千万不要复读上一次的问题！"
        
        return f"【系统指令】下一步目标：获取用户的 {target} 偏好。{hints.get(target, '')}{anti_repeat}"




# import numpy as np

# class DialogueStrategy:
#     def __init__(self, max_turns=6):
#         # 1. 差异化阈值：根据标签数量设定不同的“收敛标准”
#         # 标签越多，达到高置信度的数学难度越大，因此阈值应降低
#         self.thresholds = {
#             "Consumption": 0.40,  # 3分类，>0.50 意味着它是绝对多数
#             "Interest": 0.35,     # 5分类，>0.40 已经是第二名的两倍以上，足以判定
#             "Pace": 0.55          # 2分类，要求较高（>0.65）才算稳定
#         }
        
#         self.max_turns = max_turns
#         self.dim_priority = ["Interest", "Consumption", "Pace"]
        
#         # 记录已询问过的维度，防止连续复读
#         self.asked_history = []

#     def get_decision(self, metrics, turn_id):
#         """
#         更加灵活的决策逻辑
#         """
#         # 1. 强制终止检查
#         if turn_id >= self.max_turns:
#             return {"action": "TERMINATE", "target_dim": None, "reason": "Max turns reached"}

#         # 2. 评估各维度的收敛状态
#         candidate_dims = []

#         # for dim in self.dim_priority:
#         #     conf = metrics.get(f"{dim}_confidence", 0.0)
#         #     threshold = self.thresholds.get(dim, 0.6)
            
#         #     # 如果置信度未达标，列为待询问对象
#         #     if conf < threshold:
#         #         # 计算“不确定缺口”：缺口越大，越值得问
#         #         gap = threshold - conf
#         #         candidate_dims.append({
#         #             "dim": dim,
#         #             "gap": gap,
#         #             "conf": conf
#         #         })
#         # # 3. 灵活决策逻辑
#         # if not candidate_dims:
#         #     return {"action": "TERMINATE", "target_dim": None, "reason": "All dimensions satisfied"}

#         for dim in self.dim_priority:
#             # 【重要】确保 key 名与 evaluator.py 生成的一致 (通常是 dim_confidence)
#             conf = metrics.get(f"{dim}_confidence", 0.0)
#             threshold = self.thresholds.get(dim, 0.6)
            
#             # 如果置信度还没达标
#             if conf < threshold:
#                 gap = threshold - conf
#                 candidate_dims.append({"dim": dim, "gap": gap})

#         if not candidate_dims:
#             return {"action": "TERMINATE", "target_dim": None, "reason": "All clear"}

#         # 4. 优化排序：优先选择“缺口最大”且“上一轮没问过”的维度
#         # 这样可以交替询问，避免盯着一个维度猛薅
#         candidate_dims.sort(key=lambda x: x['gap'], reverse=True)
        
#         # 尝试换个话题，不要连续问同一个维度超过2次
#         target = candidate_dims[0]
#         if len(self.asked_history) >= 1 and self.asked_history[-1] == target['dim']:
#             if len(candidate_dims) > 1:
#                 target = candidate_dims[1] # 换第二紧迫的话题问

#         self.asked_history.append(target['dim'])
        
#         return {
#             "action": "ASK",
#             "target_dim": target['dim'],
#             "reason": f"{target['dim']} gap is {target['gap']:.2f}"
#         }

#     def get_prompt_hint(self, decision, persona):
#         """
#         优化提示词：增加“已知信息确认”，让对话更自然
#         """
#         if decision["action"] == "TERMINATE":
#             return "【系统指令】用户偏好已明确。请结合已知信息（预算、兴趣、节奏）直接输出详细的行程方案。"

#         target = decision["target_dim"]
        
#         # 从 persona 获取当前该维度概率最高的标签，用于提问时的“推测性确认”
#         current_guess = persona.get_top_label(target)
#         current_conf = persona.get_confidence(target)

#         # 动态提示词：如果已经有一定倾向，让 LLM 去“确认”而不是“盲问”
#         if current_conf > 0.3:
#             base_hint = f"目前系统推测用户可能倾向于【{current_guess}】，请在对话中委婉地确认这一点。"
#         else:
#             base_hint = "目前该信息完全缺失，请主动发起询问。"

#         hints = {
#             "Consumption": f"{base_hint} 引导用户聊聊预算（如：经济型、豪华型）。",
#             "Interest": f"{base_hint} 引导用户聊聊景点偏好（如：自然风光、古迹、购物）。",
#             "Pace": f"{base_hint} 引导用户明确节奏（如：想多打卡几个地方还是悠闲慢游）。"
#         }
        
#         # 增加防复读指令
#         anti_repeat = " 注意：请使用自然的口吻，不要机械化重复问题。如果用户已经部分提到，请通过确认的方式引导。"
        
#         return f"【系统指令】目标维度：{target}。{hints.get(target, '')}{anti_repeat}"
