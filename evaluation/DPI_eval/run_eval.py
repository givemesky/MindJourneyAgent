# -*- coding: utf-8 -*-
import json
import os
import sys
import time
import traceback
from tqdm import tqdm  # 导入进度条库

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

def run_evaluation():
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到测试数据: {DATA_PATH}")
        return
        
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    # 1. 过滤掉已经跑过的数据（断点续传）
    pending_cases = [c for c in test_cases if not os.path.exists(os.path.join(LOGS_ROOT, c['uid'], "eval_summary.txt"))]
    
    print(f"🚀 开始全量评估 | 总数: {len(test_cases)} | 剩余待处理: {len(pending_cases)}")

    assistant = AssistantAgent(log_dir=LOGS_ROOT)
    user_sim = UserSimulator(api_key=GEMMA_API_KEY, base_url=GEMMA_API_URL)
    
    all_summaries = []

    # 2. 使用 tqdm 包裹循环
    pbar = tqdm(pending_cases, desc="Evaluating", unit="case")

    for case in pbar:
        uid = case['uid']
        gt_persona = case['ground_truth']['persona']
        
        # 更新进度条左侧描述，实时显示当前正在处理的 UID
        pbar.set_postfix_str(f"Current UID: {uid}")

        try:
            if hasattr(assistant.bridge.dpi_engine, 'reset'):
                assistant.bridge.dpi_engine.reset()
            
            chat_history = []
            turn = 1
            final_response = None

            while turn <= MAX_TURNS:
                time.sleep(10) # 遵守速率限制
                
                user_text = user_sim.generate_response(case, chat_history)
                chat_history.append({"role": "user", "content": user_text})

                # 核心步骤：此时控制台不会打印对话，因为我们删除了 print
                assistant_res = assistant.process_turn(user_text, turn)
                chat_history.append({"role": "assistant", "content": assistant_res['reply']})

                if assistant_res['status'] == 'TERMINATE' or turn == MAX_TURNS:
                    final_response = assistant_res
                    assistant.terminate_session(uid)
                    history_path = os.path.join(LOGS_ROOT, uid, "chat_history.json")
                    with open(history_path, "w", encoding="utf-8") as f:
                        json.dump(chat_history, f, ensure_ascii=False, indent=2)
                    break
                turn += 1

            # 计算准确率并写入文件（后台静默进行）
            evaluation = evaluate_persona(final_response['persona'], gt_persona)
            match_count = sum(1 for res in evaluation.values() if res['match'])
            accuracy = (match_count / 3) * 100
            
            summary_lines = [f"📊 评估结果 (UID: {uid})", "-"*30]
            for dim, res in evaluation.items():
                status = "✅ 匹配" if res['match'] else "❌ 错误"
                summary_lines.append(f"   - {dim:<11}: 推断={res['inferred']:<10} | 标准={res['gt']:<10} | {status}")
            summary_lines.extend(["="*50, f"🏁 结论报告", f"⏱️  收敛轮次: {turn}", f"标签匹配数: {match_count}/3", f"单例准确率: {accuracy:.1f}%"])
            
            with open(os.path.join(LOGS_ROOT, uid, "eval_summary.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(summary_lines))

            all_summaries.append({"turns": turn, "accuracy": accuracy})

        except Exception:
            # 如果某个 Case 失败，我们不在进度条中间打印长长的报错，记录到专门的 error.log
            with open("eval_errors.log", "a") as err_f:
                err_f.write(f"Error UID {uid}:\n{traceback.format_exc()}\n\n")
            continue

    # 3. 最终汇总打印
    print("\n\n" + "🏆 实验汇总 " + "="*30)
    if all_summaries:
        avg_acc = sum(r['accuracy'] for r in all_summaries) / len(all_summaries)
        avg_turns = sum(r['turns'] for r in all_summaries) / len(all_summaries)
        print(f"📊 已测数量: {len(all_summaries)}")
        print(f"🎯 平均准确率: {avg_acc:.2f}%")
        print(f"⏱️  平均收敛轮次: {avg_turns:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()