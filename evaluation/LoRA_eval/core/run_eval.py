import sys
import os
import json
import re

# 1. 处理路径：将项目根目录加入 sys.path，确保能导入 agents 文件夹
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.user_simulator import UserSimulator
from agents.assistant_agent import MindJourneyAssistant

def is_mining_complete(assistant_output):
    """
    检查助手的输出是否已经集齐了 4 个维度的信息。
    """
    # 使用正则提取 <summary> 标签内的内容
    summary_match = re.search(r'<summary>(.*?)</summary>', assistant_output, re.DOTALL)
    if not summary_match:
        return False
    
    try:
        # 解析 JSON 字符串
        summary_str = summary_match.group(1).strip()
        summary_data = json.loads(summary_str)
        
        # 定义需要检查的 4 个核心维度
        core_keys = ["Destination", "Consumption", "Interest", "Pace"]
        
        # 统计已填充的维度
        filled_count = 0
        for key in core_keys:
            val = summary_data.get(key)
            # 只有当值不是 None (JSON的null), 且不是 "null"/"未知"/空字符串时，才认为已填充
            if val is not None and str(val).lower() not in ["null", "none", "未知", ""]:
                filled_count += 1
        
        print(f"--- 挖掘进度: {filled_count}/4 ---")
        
        # 只有 4 个维度全部集齐，才返回 True
        return filled_count == 4
    except Exception as e:
        # 如果 JSON 解析失败，说明格式还不标准，继续对话
        return False

def main():
    # --- 配置参数 ---
    API_KEY = "AIzaSyCfU8v-9LTXMT0-xWtGf503GaudDh55FDs" # 建议正式实验时从环境变量读取
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    
    DATA_PATH = "../data/processed_data/test_personas.json"
    MAX_TURNS = 6  

    # 2. 加载测试画像
    if not os.path.exists(DATA_PATH):
        print(f"错误：找不到数据文件 {DATA_PATH}")
        return

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        personas = json.load(f)

    # 提取第一组数据
    case_id = list(personas.keys())[0]
    persona_data = personas[case_id]

    print(f"系统：成功加载画像 ID {case_id}")
    print(f"系统：该游客的目标是 -> {json.dumps(persona_data['ground_truth'], ensure_ascii=False)}")
    print("-" * 50)

    # 3. 初始化 Agents
    print("系统：正在初始化 LoRA 模型 (Group C)...")
    assistant = MindJourneyAssistant(mode="lora")

    print("系统：正在初始化 User Simulator (Gemma-3)...")
    user_sim = UserSimulator(
        persona_data=persona_data,
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # 4. 对话循环
    conversation_history = []
    
    print("\n" + "="*20 + " 对话开始 " + "="*20 + "\n")

    for turn in range(1, MAX_TURNS + 1):
        print(f"--- 第 {turn} 轮 ---")

        # A. User Agent 发言
        user_reply = user_sim.get_response(conversation_history)
        print(f"游客 (User Agent):\n{user_reply}\n")
        conversation_history.append({"role": "user", "content": user_reply})

        # B. Assistant Agent (LoRA) 发言
        print("助手 (MindJourney-LoRA) 正在思考...")
        assistant_reply = assistant.generate_response(conversation_history)
        
        print(f"助手回复:\n{assistant_reply}\n")
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        # C. 判定：通过解析 JSON 判定意图是否收集完毕
        if is_mining_complete(assistant_reply):
            print(f"系统提示：检测到 <summary> 已集齐所有维度。对话在第 {turn} 轮提前结束。")
            break
        
        if turn == MAX_TURNS:
            print("系统提示：已达到最大对话轮数，停止实验。")

    print("\n" + "="*20 + " 对话结束 " + "="*20)

if __name__ == "__main__":
    main()