# -*- coding: utf-8 -*-
import json
import os
import sys
import time
import traceback
from tqdm import tqdm

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
LOGS_QWEN_ROOT = os.path.join(current_dir, "logs_Qwen")
MAX_TURNS = 8

# Gemma API 配置 (User Simulator)
GEMMA_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/" 
GEMMA_API_KEY = "AIzaSyDclRzZ4_PT-G2A2rAefVc7MqTdqZf1YEA" 

# 环境配置：禁用 Tokenizer 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def run_evaluation():
    # 加载测试用例
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到测试数据: {DATA_PATH}")
        return
        
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    # 断点续传：过滤掉已经生成 eval_summary.txt 的 UID
    pending_cases = [c for c in test_cases if not os.path.exists(os.path.join(LOGS_QWEN_ROOT, c['uid'], "eval_summary.txt"))]
    
    print(f"🚀 开始 Qwen 对照组全量评估 | 总数: {len(test_cases)} | 待处理: {len(pending_cases)}")
    
    # 初始化用户模拟器 (单例)
    user_sim = UserSimulator(api_key=GEMMA_API_KEY, base_url=GEMMA_API_URL)
    
    # 使用 tqdm 进度条
    pbar = tqdm(pending_cases, desc="Evaluating Qwen", unit="case")

    for case in pbar:
        uid = case['uid']
        gt_persona = case['ground_truth']['persona']
        pbar.set_postfix_str(f"UID: {uid[:8]}...")

        try:
            # 每个 Case 实例化一个新的 AssistantQwen 类（但不重新加载模型权重）
            # 这样可以重置其内部的 history_evolution 和 captured_constraints
            assistant = AssistantQwen(log_dir=LOGS_QWEN_ROOT)
            
            chat_history_for_user = []    
            chat_history_for_assistant = [] 
            turn = 1
            final_res = None

            while turn <= MAX_TURNS:
                # 遵守速率限制
                time.sleep(4)

                # A. 用户发言
                user_text = user_sim.generate_response(case, chat_history_for_user)
                chat_history_for_user.append({"role": "user", "content": user_text})
                chat_history_for_assistant.append({"role": "user", "content": user_text})

                # B. 助手 (Qwen) 处理
                assistant_res = assistant.process_turn(user_text, turn, chat_history_for_assistant)
                final_res = assistant_res
                chat_history_for_user.append({"role": "assistant", "content": assistant_res['reply']})
                chat_history_for_assistant.append({"role": "assistant", "content": assistant_res['reply']})

                # C. 检查终止
                if assistant_res['status'] == 'TERMINATE' or turn == MAX_TURNS:
                    # 保存原始日志
                    assistant.terminate_session(uid)
                    
                    # 保存对话历史
                    history_path = os.path.join(LOGS_QWEN_ROOT, uid, "chat_history.json")
                    with open(history_path, "w", encoding="utf-8") as f:
                        json.dump(chat_history_for_user, f, ensure_ascii=False, indent=2)

                    # 生成 eval_summary.txt
                    evaluation = evaluate_persona(assistant_res['persona'], gt_persona)
                    match_count = sum(1 for res in evaluation.values() if res['match'])
                    accuracy = (match_count / 3) * 100
                    
                    summary = [
                        f"📊 评估结果 (UID: {uid})",
                        "-" * 30
                    ]
                    for dim, res in evaluation.items():
                        status = "✅ 匹配" if res['match'] else "❌ 错误"
                        summary.append(f"   - {dim:<11}: 推断={res['inferred']:<10} | 标准={res['gt']:<10} | {status}")
                    
                    summary.extend([
                        "=" * 50,
                        f"🏁 结论报告",
                        f"⏱️  收敛轮次: {turn}",
                        f"标签匹配数: {match_count}/3",
                        f"单例准确率: {accuracy:.1f}%"
                    ])
                    
                    with open(os.path.join(LOGS_QWEN_ROOT, uid, "eval_summary.txt"), "w", encoding="utf-8") as f:
                        f.write("\n".join(summary))
                    break
                
                turn += 1

        except Exception:
            # 错误记录到独立日志，不中断主循环
            with open("qwen_eval_errors.log", "a") as err_f:
                err_f.write(f"Error UID {uid}:\n{traceback.format_exc()}\n\n")
            continue

    print(f"\n✅ Qwen 对照组全量评估完成！日志存放在: {LOGS_QWEN_ROOT}")

if __name__ == "__main__":
    run_evaluation()