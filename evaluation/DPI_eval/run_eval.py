# -*- coding: utf-8 -*-
import json
import os
import sys
import time

# --- 核心路径修复 ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将本地的 agents 文件夹加入搜索路径
agents_path = os.path.join(current_dir, "agents")
if agents_path not in sys.path:
    sys.path.insert(0, agents_path)

# 定位项目根目录 (MindJourneyAgent)
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 现在可以安全导入了
try:
    from user_simulator import UserSimulator
    from assistant_agent import AssistantAgent
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# ================= 配置区 =================
DATA_PATH = os.path.join(current_dir, "data", "dpi_me_test_cases.json")
# 日志根目录
LOGS_ROOT = os.path.join(current_dir, "logs")

# 请替换为你真实的 Gemma 3 API 地址
GEMMA_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/" 
GEMMA_API_KEY = "AIzaSyDclRzZ4_PT-G2A2rAefVc7MqTdqZf1YEA" 

MAX_TURNS = 8  
# =========================================

def evaluate_persona(inferred, ground_truth):
    """对比推断结果与标准答案"""
    results = {}
    for dim in ["Consumption", "Interest", "Pace"]:
        inf_val = inferred.get(dim, {}).get('label')
        gt_val = ground_truth.get(dim)
        results[dim] = {
            "inferred": inf_val,
            "gt": gt_val,
            "match": str(inf_val).lower() == str(gt_val).lower()
        }
    return results

def run_evaluation(num_cases=2):
    # 1. 加载测试数据
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    # 2. 初始化助手 Agent，传入日志根目录
    assistant = AssistantAgent(log_dir=LOGS_ROOT)

    # 3. 初始化用户模拟器
    user_sim = UserSimulator(api_key=GEMMA_API_KEY, base_url=GEMMA_API_URL)

    all_results = []

    print(f"\n🚀 开始评估实验 (前 {num_cases} 组测试用例)\n" + "="*50)

    for i in range(min(num_cases, len(test_cases))):
        case = test_cases[i]
        uid = case['uid']
        gt_persona = case['ground_truth']['persona']
        
        print(f"\n[Case {i+1}] UID: {uid}")
        print(f"🎯 目标画像: {gt_persona}")
        
        # 重置当前 Case 的对话状态
        if hasattr(assistant.bridge.dpi_engine, 'reset'):
            assistant.bridge.dpi_engine.reset()
        
        chat_history = []
        turn = 1
        final_response = None
        start_time = time.time()

        # 开始对话循环
        while turn <= MAX_TURNS:
            # A. 用户 Agent 生成回复
            user_text = user_sim.generate_response(case, chat_history)
            print(f"   👤 User (Turn {turn}): {user_text}")
            chat_history.append({"role": "user", "content": user_text})

            # B. 助手 Agent 处理
            assistant_res = assistant.process_turn(user_text, turn)
            print(f"   🤖 Assistant: {assistant_res['reply']}")
            chat_history.append({"role": "assistant", "content": assistant_res['reply']})

            # C. 检查是否收敛或达到最大轮次
            if assistant_res['status'] == 'TERMINATE' or turn == MAX_TURNS:
                final_response = assistant_res
                
                # --- 【核心修改点：保存该 Case 日志】 ---
                # 1. 调用 Assistant 保存 DPI 内部日志 (UserProfile + Convergence)
                assistant.terminate_session(uid)
                
                # 2. 手动保存对话历史原文
                history_path = os.path.join(LOGS_ROOT, uid, "chat_history.json")
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(chat_history, f, ensure_ascii=False, indent=2)
                
                if assistant_res['status'] == 'TERMINATE':
                    print(f"\n✨ 对话收敛于第 {turn} 轮。")
                else:
                    print(f"\n⚠️ 达到最大轮次，强制结束并保存现状。")
                break
            
            turn += 1

        # 4. 结果统计
        duration = time.time() - start_time
        evaluation = evaluate_persona(final_response['persona'], gt_persona)
        
        case_summary = {
            "uid": uid,
            "turns": turn,
            "duration": duration,
            "results": evaluation
        }
        all_results.append(case_summary)

        # 打印单例结果
        print(f"\n📊 评估结果:")
        for dim, res in evaluation.items():
            status = "✅ 匹配" if res['match'] else "❌ 错误"
            print(f"   - {dim:<11}: 推断={res['inferred']:<10} | 标准={res['gt']:<10} | {status}")

    # 5. 生成总览简报
    print("\n" + "="*50)
    print("🏁 评估结束报告")
    avg_turns = sum([r['turns'] for r in all_results]) / len(all_results)
    print(f"📝 测试用例数: {len(all_results)}")
    print(f"⏱️  平均收敛轮次: {avg_turns:.2f}")
    print(f"📂 详细日志已分目录存放在: {LOGS_ROOT}")
    print("="*50)

if __name__ == "__main__":
    run_evaluation(num_cases=1)