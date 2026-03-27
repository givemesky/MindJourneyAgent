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
        # 获取推断标签，如果不存在则为 Unknown
        inf_val = inferred.get(dim, {}).get('label', "Unknown")
        gt_val = ground_truth.get(dim)
        results[dim] = {
            "inferred": inf_val,
            "gt": gt_val,
            "match": str(inf_val).lower() == str(gt_val).lower()
        }
    return results


def run_evaluation(num_cases=1):
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    assistant = AssistantAgent(log_dir=LOGS_ROOT)
    user_sim = UserSimulator(api_key=GEMMA_API_KEY, base_url=GEMMA_API_URL)
    all_results = []

    print(f"\n🚀 开始评估实验 (前 {num_cases} 组测试用例)\n" + "="*50)

    for i in range(min(num_cases, len(test_cases))):
        case = test_cases[i]
        uid = case['uid']
        gt_persona = case['ground_truth']['persona']
        
        print(f"\n[Case {i+1}] UID: {uid}")
        if hasattr(assistant.bridge.dpi_engine, 'reset'):
            assistant.bridge.dpi_engine.reset()
        
        chat_history = []
        turn = 1
        final_response = None
        start_time = time.time()

        while turn <= MAX_TURNS:
            user_text = user_sim.generate_response(case, chat_history)
            print(f"   👤 User (Turn {turn}): {user_text}")
            chat_history.append({"role": "user", "content": user_text})

            assistant_res = assistant.process_turn(user_text, turn)
            print(f"   🤖 Assistant: {assistant_res['reply']}")
            chat_history.append({"role": "assistant", "content": assistant_res['reply']})

            if assistant_res['status'] == 'TERMINATE' or turn == MAX_TURNS:
                final_response = assistant_res
                # 保存原始日志
                assistant.terminate_session(uid)
                # 保存对话历史
                history_path = os.path.join(LOGS_ROOT, uid, "chat_history.json")
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(chat_history, f, ensure_ascii=False, indent=2)
                break
            turn += 1

        # --- 结果统计与总结生成 ---
        evaluation = evaluate_persona(final_response['persona'], gt_persona)
        match_count = sum(1 for res in evaluation.values() if res['match'])
        accuracy = (match_count / 3) * 100
        
        # 构造总结文本
        summary_lines = []
        summary_lines.append(f"\n📊 评估结果:")
        for dim, res in evaluation.items():
            status = "✅ 匹配" if res['match'] else "❌ 错误"
            summary_lines.append(f"   - {dim:<11}: 推断={res['inferred']:<10} | 标准={res['gt']:<10} | {status}")
        
        summary_lines.append("\n" + "="*50)
        summary_lines.append("🏁 评估结束报告")
        summary_lines.append(f"📝 测试用例 UID: {uid}")
        summary_lines.append(f"⏱️  收敛轮次: {turn:.2f}")
        summary_lines.append(f"标签匹配数：{match_count}")
        summary_lines.append(f"偏好推理准确率 ：{accuracy:.1f}%")
        
        summary_text = "\n".join(summary_lines)
        
        # 保存总结日志文件
        summary_path = os.path.join(LOGS_ROOT, uid, "eval_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        # 打印到终端
        print(summary_text)
        
        all_results.append({"turns": turn, "accuracy": accuracy})

    # 5. 最终总体简报
    if len(all_results) > 1:
        avg_acc = sum(r['accuracy'] for r in all_results) / len(all_results)
        avg_turns = sum(r['turns'] for r in all_results) / len(all_results)
        print("\n" + "高级汇总" + "!"*20)
        print(f"总测试数: {len(all_results)} | 平均准确率: {avg_acc:.1f}% | 平均轮次: {avg_turns:.2f}")

if __name__ == "__main__":
    run_evaluation(num_cases=1)


