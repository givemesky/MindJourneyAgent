# -*- coding: utf-8 -*-
import json
import os
import sys
import time

# --- 1. 路径修复 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
agents_path = os.path.join(current_dir, "agents")
if agents_path not in sys.path:
    sys.path.insert(0, agents_path)

root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from user_simulator import UserSimulator
    from assistant_Qwen import AssistantQwen
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# --- 2. 配置 ---
DATA_PATH = os.path.join(current_dir, "data", "dpi_me_test_cases.json")
LOGS_QWEN = os.path.join(current_dir, "logs_Qwen")
MAX_TURNS = 8

# Gemma API 配置
GEMMA_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/" 
GEMMA_API_KEY = "AIzaSyDclRzZ4_PT-G2A2rAefVc7MqTdqZf1YEA" 

def evaluate_persona(inferred, ground_truth):
    """对比推断结果与标准答案"""
    results = {}
    for dim in ["Consumption", "Interest", "Pace"]:
        # 获取推断标签
        inf_val = inferred.get(dim, {}).get('label', "Unknown")
        gt_val = ground_truth.get(dim)
        results[dim] = {
            "inferred": inf_val,
            "gt": gt_val,
            "match": str(inf_val).lower() == str(gt_val).lower()
        }
    return results

def run_test_qwen_case():
    # 加载测试用例
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    case = test_cases[0] # 测试第一个用例
    uid = case['uid']
    gt_persona = case['ground_truth']['persona']

    print(f"\n" + "!"*60)
    print(f"🧪 开始测试对照组：Assistant Qwen (Vanilla LLM)")
    print(f"UID: {uid}")
    print(f"Ground Truth: {gt_persona}")
    print("!"*60 + "\n")

    assistant = AssistantQwen(log_dir=LOGS_QWEN)
    user_sim = UserSimulator(api_key=GEMMA_API_KEY, base_url=GEMMA_API_URL)

    chat_history_for_user = []    
    chat_history_for_assistant = [] 
    
    turn = 1
    start_time = time.time()
    final_res = None

    while turn <= MAX_TURNS:
        # A. 用户 Agent 发言
        user_text = user_sim.generate_response(case, chat_history_for_user)
        print(f"Round {turn} | 👤 User: {user_text}")
        
        chat_history_for_user.append({"role": "user", "content": user_text})
        chat_history_for_assistant.append({"role": "user", "content": user_text})

        # B. 纯大模型助手 (Qwen) 处理
        assistant_res = assistant.process_turn(user_text, turn, chat_history_for_assistant)
        final_res = assistant_res
        
        print(f"Round {turn} | 🤖 Assistant: {assistant_res['reply']}")
        print(f"         🧠 Persona Captured: {assistant_res['persona']}")
        
        chat_history_for_user.append({"role": "assistant", "content": assistant_res['reply']})
        chat_history_for_assistant.append({"role": "assistant", "content": assistant_res['reply']})

        # C. 检查终止条件
        if assistant_res['status'] == 'TERMINATE' or turn == MAX_TURNS:
            if assistant_res['status'] == 'TERMINATE':
                print(f"\n✅ Qwen 自主决定结束对话（收敛于第 {turn} 轮）。")
            else:
                print(f"\n⚠️ 达到最大轮次，强制结束。")
            
            # 1. 保存原始日志
            assistant.terminate_session(uid)
            
            # 2. 保存对话原文
            history_path = os.path.join(LOGS_QWEN, uid, "chat_history.json")
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(chat_history_for_user, f, ensure_ascii=False, indent=2)

            # 3. --- 生成 eval_summary.txt 总结日志 ---
            evaluation = evaluate_persona(assistant_res['persona'], gt_persona)
            match_count = sum(1 for res in evaluation.values() if res['match'])
            accuracy = (match_count / 3) * 100
            
            summary_lines = [
                f"📊 评估结果 (UID: {uid})",
                "-" * 30
            ]
            for dim, res in evaluation.items():
                status = "✅ 匹配" if res['match'] else "❌ 错误"
                summary_lines.append(f"   - {dim:<11}: 推断={res['inferred']:<10} | 标准={res['gt']:<10} | {status}")
            
            summary_lines.extend([
                "=" * 50,
                f"🏁 结论报告",
                f"⏱️  收敛轮次: {turn}",
                f"标签匹配数: {match_count}/3",
                f"单例准确率: {accuracy:.1f}%"
            ])
            
            summary_text = "\n".join(summary_lines)
            
            # 保存到文件
            summary_path = os.path.join(LOGS_QWEN, uid, "eval_summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
            
            # 打印到终端
            print("\n" + summary_text)
            break
        
        turn += 1
        time.sleep(2)

    duration = time.time() - start_time
    print(f"\n🏁 测试完成，耗时: {duration:.2f}s")

if __name__ == "__main__":
    run_test_qwen_case()