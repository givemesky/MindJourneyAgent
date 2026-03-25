import json
import openai
import random

# 直接在文件中定义 Prompt
USER_SIMULATOR_PROMPT_TEMPLATE = """你现在扮演一名正在计划北京旅游的真实游客。
你的行为必须严格受限于下方的【核心人设清单】。

### 1. 【核心人设清单】 (这是你的最高指令，绝对不可逾越):
- 目的地: {destination}
- 消费预算限制: {consumption} 
- 核心兴趣偏好: {interest}
- 行程节奏倾向: {pace}
- 必须去的景点（重点参考）: {attractions}
- 必须尝的美食（重点参考）: {dishes}
- 其他硬性约束: 
{original_goals_str}

### 2. 你的行为准则:
1. **立场坚定 (Persona Loyalty)**：即使助手引导你尝试更贵、更快或不同类型的方案，你也必须忠于上述清单。禁止为了“礼貌”而配合助手更改你的消费档次或偏好。
2. **被动释放信息**：不要一次性交代所有需求。助手问什么，你才答什么。每一轮对话只回答助手提出的**一个**核心问题。如果助手问得太宽泛，你只给出一个大方向。
3. **细节驱动回答**：提到兴趣时，优先说具体的景点/美食名称（如“故宫”），而非抽象词（如“文化”）。
4. **口语化**：使用自然、口语化的中文，像在微信上聊天一样，可以带一点点情绪。
5. **隐性约束（Original Goal）**：清单中的“其他具体要求”是你的底线。如果助手推荐的方案不符（如评分不够、价格太高），你要表现出不满意并提出这些要求。

### 3. 对话背景:
你刚开始和文旅助手对话。助手的首句通常是欢迎语。
请开始你的第一轮动作：表达你想去北京玩的初步意图，不要透露任何具体细节。
"""


class UserSimulator:
    def __init__(self, persona_data, api_key, base_url):
        self.persona = persona_data
        self.ground_truth = persona_data['ground_truth']
        self.attractions = persona_data.get('specific_attractions', [])
        self.dishes = persona_data.get('specific_dishes', [])
        self.original_goal = persona_data.get('original_goal', [])
        
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = "gemma-3-27b-it"

    # def _format_original_goals(self):
    #     """
    #     优化格式：将复杂的 CrossWOZ 列表转化为对 User 更有指引性的文字
    #     """
    #     lines = []
    #     for g in self.original_goal:
    #         # 格式: [领域, 属性, 值]
    #         domain, attr, val = g[1], g[2], g[3]
    #         if not val or val == "": continue
            
    #         if attr == "评分":
    #             lines.append(f"- 你非常看重{domain}的口碑，要求评分在{val}。")
    #         elif attr == "人均消费" or attr == "价格":
    #             lines.append(f"- 你对{domain}的预算比较敏感，期望在{val}。")
    #         elif attr == "推荐菜":
    #             lines.append(f"- 你听说那里的{val}很出名，想尝试。")
    #         else:
    #             lines.append(f"- 关于{domain}，你希望{attr}是：{val}。")
        
    #     return "\n".join(lines) if lines else "无特殊硬性要求。"

    def _format_original_goals(self):
        lines = []
        for g in self.original_goal:
            domain, attr, val = g[1], g[2], g[3]
            if not val or val == "": continue
            
            # 使用更强硬的语气
            if attr == "评分":
                lines.append(f"- 【硬性标准】你绝对不接受{domain}评分低于{val}，这是你的底线。")
            elif attr == "人均消费" or attr == "价格":
                lines.append(f"- 【预算警告】你对{domain}的开销非常敏感，必须控制在{val}左右。")
            elif attr == "推荐菜":
                lines.append(f"- 你听说那里的{val}很出名，想尝试。")
            else:
                lines.append(f"- 【具体细节】关于{domain}，你的要求是：{val}。")
        
        return "\n".join(lines) if lines else "无特殊硬性要求。"

    def _get_strategy(self, turn_count):
        """
        优化策略：平衡“保守”与“角色真实感”
        """
        if turn_count == 0:
            if random.random() < 0.8:
                return "【指令：首轮锁定】只表达想去北京的意图。如果助手问‘想去哪’，只说北京。绝对不提预算和具体的景点名。"
            else:
                return "【指令：首轮微露】说出想去北京，并顺带提一个你清单里【一定要去的景点】的名字，看看助手的反应。但不要提预算。"
        
        r = random.random()
        if r < 0.3:
            return "【指令：细节控】如果助手问到兴趣，请从你的【景点】或【美食】清单里选一个具体的告诉它，并询问‘那边环境怎么样？’或‘贵不贵？’。"
        elif r < 0.7:
            return "【指令：保守回复】针对助手的问题给出简洁回答。如果问预算，只给一个模糊区间，不要直接报清单上的原始数据。"
        else:
            return "【当前动作：反向询问】问一下助手关于清单中某个‘硬性标准’（如评分或价格）的具体情况。"

    
    def get_response(self, conversation_history):
        turn_count = len(conversation_history) // 2
        
        # 获取当前轮次的概率策略
        strategy = self._get_strategy(turn_count)
        
        # 准备指令
        prompt = USER_SIMULATOR_PROMPT_TEMPLATE.format(
            strategy_instruction=strategy,
            destination=self.ground_truth['Destination'],
            consumption=self.ground_truth['Consumption'],
            interest=self.ground_truth['Interest'],
            pace=self.ground_truth['Pace'],
            attractions=", ".join(self.attractions) if self.attractions else "未确定",
            dishes=", ".join(self.dishes) if self.dishes else "未确定",
            original_goals_str=self._format_original_goals()
        )

        messages = [{"role": "user", "content": f"【任务指令】\n{prompt}"}]
        
        # 转换历史角色
        for msg in conversation_history:
            role = "user" if msg['role'] == 'assistant' else "assistant"
            messages.append({"role": role, "content": msg['content']})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6, # 保持一定的随机性
                max_tokens=512
            )
            content = response.choices[0].message.content
            # 如果模型在回复里带了策略前缀，可以清理掉（虽然通常不会）
            return content
        except Exception as e:
            return f"API调用失败: {str(e)}"

    # def get_response(self, conversation_history):
    #     """
    #     conversation_history: 之前的对话记录
    #     """
    #     # 准备系统提示词
    #     prompt = USER_SIMULATOR_PROMPT_TEMPLATE.format(
    #         destination=self.ground_truth['Destination'],
    #         consumption=self.ground_truth['Consumption'],
    #         interest=self.ground_truth['Interest'],
    #         pace=self.ground_truth['Pace'],
    #         attractions=", ".join(self.attractions) if self.attractions else "未确定",
    #         dishes=", ".join(self.dishes) if self.dishes else "未确定",
    #         original_goals_str=self._format_original_goals()
    #     )

    #     # Gemma-3 兼容性处理 (System 转 User)
    #     # 将系统提示词作为第一条消息发送
    #     messages = [{"role": "user", "content": f"【任务指令】\n{prompt}"}]
        
    #     # 转换历史记录角色：
    #     # 在模拟器看来，原本 Assistant 的回复是它的 "user" 输入，它的回复是 "assistant" 输出
    #     for msg in conversation_history:
    #         role = "user" if msg['role'] == 'assistant' else "assistant"
    #         messages.append({"role": role, "content": msg['content']})

    #     try:
    #         response = self.client.chat.completions.create(
    #             model=self.model,
    #             messages=messages,
    #             temperature=0.7,
    #             max_tokens=512
    #         )
    #         return response.choices[0].message.content
    #     except Exception as e:
    #         return f"API调用失败: {str(e)}"

# 测试代码
if __name__ == "__main__":
    # 1. 准备测试数据 (8867号画像)
    sample_persona = {
        "persona_id": "8867",
        "ground_truth": {
            "Destination": "北京",
            "Consumption": "Economy",
            "Interest": "Culture",
            "Pace": "Relaxed"
        },
        "specific_attractions": ["故宫", "天坛"],
        "specific_dishes": ["北京烤鸭"],
        "original_goal": [[1, "景点", "评分", "4.5分以上", False]]
    }
    
    # 2. 初始化模拟器 (请确保 API Key 和 URL 正确)
    user_sim = UserSimulator(
        persona_data=sample_persona, 
        api_key="AIzaSyBC6h3wPvaBzBiug-kHN_abu2yhOlCISeU", 
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    # 3. 开启连续对话循环
    conversation_history = []
    print("系统：已成功加载画像 8867。现在开始模拟对话（输入 'quit' 退出）。")
    print("-" * 50)

    # 第一轮：User Agent 发起会话
    user_reply = user_sim.get_response(conversation_history)
    print(f"游客 (User Agent): {user_reply}")
    
    # 将 User 的发言加入历史
    # 注意：在 Assistant 的视角下，User Agent 的发言 role 是 "user"
    conversation_history.append({"role": "user", "content": user_reply})

    while True:
        # 你扮演 Assistant 进行追问
        assistant_input = input("你的回复 (Assistant): ")
        
        if assistant_input.lower() in ['quit', 'exit', '退出']:
            break
            
        conversation_history.append({"role": "assistant", "content": assistant_input})

        # 获取 User Agent 的回应
        print("等待游客思考中...")
        user_reply = user_sim.get_response(conversation_history)
        
        print(f"游客 (User Agent): {user_reply}")
        conversation_history.append({"role": "user", "content": user_reply})




