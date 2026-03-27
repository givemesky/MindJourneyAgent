# -*- coding: utf-8 -*-
import json
import os
import sys

# 确保能导入 agents 目录下的 UserSimulator
current_dir = os.path.dirname(os.path.abspath(__file__))
agents_path = os.path.join(current_dir, "agents")
if agents_path not in sys.path:
    sys.path.insert(0, agents_path)

from user_simulator import UserSimulator

def interactive_test():
    # 1. 加载测试数据
    DATA_PATH = os.path.join(current_dir, "data", "dpi_me_test_cases.json")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    # 2. 获取第一组数据作为测试人设
    case = test_cases[0]
    print("\n" + "="*50)
    print("🎭 [人设加载成功] 正在模拟第一组数据")
    print(f"📍 目的地: {case['ground_truth']['target_city']}")
    print(f"💰 真实预算上限: {case['ground_truth']['budget_limit']}")
    print(f"🎯 预设标签: {case['ground_truth']['persona']}")
    print(f"📝 原始愿望清单: {case['full_user_speech']}")
    print("="*50 + "\n")

    # 3. 初始化模拟器 (请确认 API Key 和 URL 正确)
    # 建议此处直接填入你之前能跑通的 API 配置
    simulator = UserSimulator(
        api_key="AIzaSyDclRzZ4_PT-G2A2rAefVc7MqTdqZf1YEA", 
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    chat_history = []
    
    # 第一轮：由 User Agent 发起
    print("系统：等待游客发起对话...")
    user_reply = simulator.generate_response(case, chat_history)
    print(f"\n🙋 [游客]: {user_reply}")
    
    # 存入历史（模拟器说的话 role 是 user）
    chat_history.append({"role": "user", "content": user_reply})

    # 开始循环对话
    while True:
        try:
            # 你扮演 Assistant (导游)
            guide_input = input("\n🤖 [你(导游)]: ").strip()
            
            if guide_input.lower() in ['quit', 'exit', 'q', '退出']:
                print("\n对话结束。")
                break
            
            if not guide_input:
                continue

            # 将你的话存入历史（导游的话 role 是 assistant）
            chat_history.append({"role": "assistant", "content": guide_input})

            # 获取 User Agent 的回应
            print("\n(游客正在思考...)")
            user_reply = simulator.generate_response(case, chat_history)
            
            print(f"🙋 [游客]: {user_reply}")
            
            # 存入历史
            chat_history.append({"role": "user", "content": user_reply})

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    interactive_test()