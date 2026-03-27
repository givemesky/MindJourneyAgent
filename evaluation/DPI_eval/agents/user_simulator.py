# -*- coding: utf-8 -*-
import json
import os
from openai import OpenAI  # 假设使用兼容 OpenAI 格式的 API 调用 Gemma 3

class UserSimulator:
    def __init__(self, api_key="AIzaSyDclRzZ4_PT-G2A2rAefVc7MqTdqZf1YEA", base_url="https://generativelanguage.googleapis.com/v1beta/openai/"):
        """
        初始化用户模拟器
        :param api_key: 你的 API 密钥
        :param base_url: Gemma 3 27B 服务的 API 地址
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = "gemma-3-27b-it" # 根据你的实际模型部署名称修改
        
        # 定义系统内部参考的标签空间，用于在 System Prompt 中约束模拟器
        self.label_space = (
            "- 消费偏好 (Consumption): [Budget (节省型), Economy (经济型), Luxury (豪华型/高端)]\n"
            "- 兴趣特征 (Interest): [Culture (历史文化), Nature (自然风光), Entertainment (娱乐), Shopping (购物)]\n"
            "- 行程节奏 (Pace): [Fast (特种兵/紧凑), Relaxed (慢节奏/休闲)]"
        )

    def _build_system_prompt(self, case_data):
        """
        根据测试用例的 Ground Truth 构造模拟器的系统提示词
        """
        gt = case_data["ground_truth"]
        persona = gt["persona"]
        raw_desc = case_data.get("raw_pref_desc", "无")
        full_speech = case_data.get("full_user_speech", "无具体描述")
        
        system_prompt = f"""你现在扮演一个真实的游客，正在与一名旅游规划助手对话。
        你的目标是根据你内心预设的【秘密设定】与助手交流，最终获得一份满意的行程。

        ### 你的秘密设定 (必须严格遵守):
        1. **目的地**: {gt['target_city']}
        2. **时长**: {gt['days']} 天
        3. **人数**: {gt['people_count']} 人
        4. **消费预算上限**: {gt['budget_limit']} 元人民币
        5. **核心偏好标签**:
        - 消费档次: {persona['Consumption']}
        - 兴趣倾向: {persona['Interest']}
        - 旅行节奏: {persona['Pace']}
        6. **【核心参考】原始需求全景图**: "{full_speech}" (注：这段话包含你可能想吃的特定餐厅、想去的特定景点或具体的交互语气，请作为对话细节的最高参考。)

        ### 对话原则 (实验成功的关键):
        1. **第一句话**: 请直接使用预设的 `initial_query` 发起对话。
        2. **循序渐进**: 不要一次性把所有偏好（预算、兴趣、节奏）都告诉助手。
        3. **你问我答**: 助手问你什么，你就回答什么。如果助手没问某个维度，除非你觉得对话陷入僵局，否则不要主动提及。
        4. **自然表达**: 使用口语化的中文。虽然你的设定是基于标签的（例如 Luxury），但请将其转化为自然的需求描述（例如“我想住得好一点，不差钱”）。
        5. **一致性**: 严禁在对话过程中修改或违背上述秘密设定。

        ### 助手参考的偏好维度定义:
        {self.label_space}
        """
        return system_prompt

  

    def generate_response(self, case_data, chat_history):
        """
        根据当前对话历史，生成下一句用户回复
        """
        # 1. 第一轮强制返回 initial_query
        if not chat_history:
            return case_data["initial_query"]

        # 2. 获取强化的系统设定内容
        system_content = self._build_system_prompt(case_data)
        
        # --- 核心修复：角色镜像翻转逻辑 ---
        flipped_messages = []
        
        # 注入系统指令
        flipped_messages.append({
            "role": "user", 
            "content": f"【重要指令】你现在是【游客】，正在回答【导游】的提问。以下是你的设定：\n{system_content}\n\n请记住：你是客户，不要规划行程，只需回答问题。"
        })
        flipped_messages.append({
            "role": "assistant", 
            "content": "没问题。我是游客，我会根据我的秘密设定回答导游的问题，不会主动规划。"
        })
        
        # 翻转历史记录中的角色
        for msg in chat_history:
            # 如果原角色是 user (其实是模拟器自己)，在 API 看来它是 assistant (自我补全)
            # 如果原角色是 assistant (其实是 MindJourney)，在 API 看来它是 user (外部输入)
            if msg['role'] == 'user':
                flipped_messages.append({"role": "assistant", "content": msg['content']})
            else:
                flipped_messages.append({"role": "user", "content": msg['content']})
        
        # --- 确保最后一条消息是 user (即导游的提问) ---
        # 如果不是，模型可能又会开始乱说话

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=flipped_messages,
                temperature=0.7,
                max_tokens=256
            )
            content = response.choices[0].message.content
            
            if content is None:
                return f"我比较看重{case_data['ground_truth']['persona']['Interest']}方面的体验。"
                
            return content.strip()
        except Exception as e:
            print(f"❌ [User Simulator] API 调用失败: {e}")
            return "我想想... 咱们还是继续聊行程吧。"

# ================= 单元测试示例 =================
if __name__ == "__main__":
    # 测试数据路径
    DATA_PATH = "/public/home/azsy/MindJourney_LoRA/MindJourneyAgent/evaluation/DPI_eval/data/dpi_me_test_cases.json"
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    # 选取第一个用例进行测试
    test_case = test_cases[0]
    
    # 初始化模拟器 (请根据实际 API 修改参数)
    simulator = UserSimulator(api_key="AIzaSyDclRzZ4_PT-G2A2rAefVc7MqTdqZf1YEA", base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    
    # 模拟第一轮
    first_turn = simulator.generate_response(test_case, [])
    print(f"User (Turn 1): {first_turn}")
    
    # 模拟助手提问
    mock_history = [
        {"role": "user", "content": first_turn},
        {"role": "assistant", "content": "好的，去杭州2天。请问您对住宿有什么预算要求吗？"}
    ]
    
    # 模拟用户根据设定回答
    second_turn = simulator.generate_response(test_case, mock_history)
    print(f"User (Turn 2): {second_turn}")