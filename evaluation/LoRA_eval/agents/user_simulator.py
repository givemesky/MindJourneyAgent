import json
import openai

# 直接在文件中定义 Prompt
USER_SIMULATOR_PROMPT_TEMPLATE = """你现在扮演一名正在计划去北京旅游的真实游客。
你手中有一份详细的【旅游意向清单】，但你性格内敛，**绝对不会主动一次性交代所有需求**。

### 你的行为准则：
1. **被动释放信息**：除非助手主动询问，否则不要提及你的预算、节奏、具体景点或菜品。
2. **循序渐进**：每一轮只回答助手提出的问题。如果助手问得宽泛，你只给出一个模糊的方向。
3. **忠于设定**：你的回答必须严格符合下方的【旅游意向清单】。如果助手问到清单外的内容，你可以随性发挥，但不能与清单冲突。
4. **口语化**：使用自然、口语化的中文，像在微信上聊天一样，可以带一点点情绪。

### 你的秘密【旅游意向清单】：
1. 目的地：{destination}
2. 期望消费水平：{consumption}
3. 核心兴趣：{interest}
4. 行程节奏：{pace}
5. **一定要去的景点**：{attractions}
6. **一定要尝的美食**：{dishes}
7. 其他具体要求（来自原始目标）：
{original_goals_str}

### 当前对话背景：
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

    def _format_original_goals(self):
        """将原始目标列表转换为易读的字符串"""
        lines = []
        for g in self.original_goal:
            # CrossWOZ格式: [ID, 领域, 属性, 值, 是否完成]
            lines.append(f"- {g[1]}的{g[2]}要求: {g[3]}")
        return "\n".join(lines) if lines else "无特殊要求"

    def get_response(self, conversation_history):
        """
        conversation_history: 之前的对话记录
        """
        # 准备系统提示词
        prompt = USER_SIMULATOR_PROMPT_TEMPLATE.format(
            destination=self.ground_truth['Destination'],
            consumption=self.ground_truth['Consumption'],
            interest=self.ground_truth['Interest'],
            pace=self.ground_truth['Pace'],
            attractions=", ".join(self.attractions) if self.attractions else "未确定",
            dishes=", ".join(self.dishes) if self.dishes else "未确定",
            original_goals_str=self._format_original_goals()
        )

        # Gemma-3 兼容性处理 (System 转 User)
        # 将系统提示词作为第一条消息发送
        messages = [{"role": "user", "content": f"【任务指令】\n{prompt}"}]
        
        # 转换历史记录角色：
        # 在模拟器看来，原本 Assistant 的回复是它的 "user" 输入，它的回复是 "assistant" 输出
        for msg in conversation_history:
            role = "user" if msg['role'] == 'assistant' else "assistant"
            messages.append({"role": role, "content": msg['content']})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=512
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API调用失败: {str(e)}"

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
        api_key="AIzaSyDgpc-WpDzv24ZEPINcLdZtgyEZeuZPb4E", 
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