# -*- coding: utf-8 -*-
import json
import os
import sys
import time
import argparse  # 新增：用于处理命令行参数

# --- 核心路径修复 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
agents_path = os.path.join(current_dir, "agents")
if agents_path not in sys.path:
    sys.path.insert(0, agents_path)
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from user_simulator import UserSimulator
    from assistant_agent import AssistantAgent
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# ================= 配置区 =================
DATA_PATH = os.path.join(current_dir, "data", "dpi_me_test_cases.json")
LOGS_ROOT = os.path.join(current_dir, "logs")

GEMMA_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/" 
GEMMA_API_KEY = "AIzaSyDclRzZ4_PT-G2A2rAefVc7MqTdqZf1YEA" 

MAX_TURNS = 8  
# =========================================

def evaluate_persona(inferred, ground_truth):
    """对比推断结果与标准答案"""
    results = {}
    for dim in ["Consumption", "Interest", "Pace"]:
        inf_val = inferred.get(dim, {}).get('label', "Unknown")
        gt_val = ground_truth.get(dim)
        results[dim] = {
            "inferred": inf_val,
            "gt": gt_val,
            "match": str(inf_val).lower() == str(gt_val).lower()
        }
    return results

def run_evaluation(target_uid=None, num_cases=None):
    """
    运行评估
    :param target_uid: 指定的 UID 字符串
    :param num_cases: 如果没指定 UID，则运行前 N 个数据
    """
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    # --- 核心修改：筛选数据逻辑 ---
    if target_uid:
        # 寻找匹配 UID 的数据
        selected_cases = [c for c in test_cases if c['uid'] == target_uid]
        if not selected_cases:
            print(f"❌ 错误：在数据集中找不到 UID 为 [{target_uid}] 的数据。")
            return
    elif num_cases:
        selected_cases = test_cases[:num_cases]
    else:
        selected_cases = [test_cases[0]] # 默认运行第一条

    assistant = AssistantAgent(log_dir=LOGS_ROOT)
    user_sim = UserSimulator(api_key=GEMMA_API_KEY, base_url=GEMMA_API_URL)
    all_results = []

    print(f"\n🚀 开始评估实验 | 模式: {'指定UID' if target_uid else '批量测试'}\n" + "="*50)

    for i, case in enumerate(selected_cases):
        uid = case['uid']
        gt_persona = case['ground_truth']['persona']
        
        print(f"\n[Task {i+1}] UID: {uid}")
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
                assistant.terminate_session(uid)
                history_path = os.path.join(LOGS_ROOT, uid, "chat_history.json")
                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(chat_history, f, ensure_ascii=False, indent=2)
                break
            turn += 1

        evaluation = evaluate_persona(final_response['persona'], gt_persona)
        match_count = sum(1 for res in evaluation.values() if res['match'])
        accuracy = (match_count / 3) * 100
        
        summary_lines = []
        summary_lines.append(f"\n📊 评估结果:")
        for dim, res in evaluation.items():
            status = "✅ 匹配" if res['match'] else "❌ 错误"
            summary_lines.append(f"   - {dim:<11}: 推断={res['inferred']:<10} | 标准={res['gt']:<10} | {status}")
        
        summary_lines.append("\n" + "="*50)
        summary_lines.append("🏁 评估结束报告")
        summary_lines.append(f"📝 测试用例 UID: {uid}")
        summary_lines.append(f"⏱️  收敛轮次: {turn}")
        summary_lines.append(f"标签匹配数：{match_count}")
        summary_lines.append(f"偏好推理准确率 ：{accuracy:.1f}%")
        
        summary_text = "\n".join(summary_lines)
        summary_path = os.path.join(LOGS_ROOT, uid, "eval_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        print(summary_text)
        all_results.append({"turns": turn, "accuracy": accuracy})

    if len(all_results) > 1:
        avg_acc = sum(r['accuracy'] for r in all_results) / len(all_results)
        avg_turns = sum(r['turns'] for r in all_results) / len(all_results)
        print("\n" + "高级汇总" + "!"*20)
        print(f"总测试数: {len(all_results)} | 平均准确率: {avg_acc:.1f}% | 平均轮次: {avg_turns:.2f}")

if __name__ == "__main__":
    # 使用 argparse 让脚本更灵活
    parser = argparse.ArgumentParser(description="DPI-SynPT Evaluation Script")
    parser.add_argument("--uid", type=str, help="指定要运行的测试用例 UID")
    parser.add_argument("--num", type=int, default=1, help="如果没有指定UID，运行前N条数据")
    
    args = parser.parse_args()

    # 调用函数
    run_evaluation(target_uid=args.uid, num_cases=args.num)