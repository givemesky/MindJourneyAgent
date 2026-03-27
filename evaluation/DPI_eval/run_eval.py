# -*- coding: utf-8 -*-
import json
import os
import sys
import time
import traceback

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
GEMMA_API_KEY = "你的API_KEY" 

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

def run_evaluation():
    # 1. 加载测试数据
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到测试数据文件: {DATA_PATH}")
        return
        
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    total_cases = len(test_cases)
    print(f"\n🚀 开始全量评估实验 (共 {total_cases} 组测试用例)")
    print("="*60)

    # 2. 初始化核心组件
    assistant = AssistantAgent(log_dir=LOGS_ROOT)
    user_sim = UserSimulator(api_key=GEMMA_API_KEY, base_url=GEMMA_API_URL)
    
    all_summaries = []

    for i, case in enumerate(test_cases):
        uid = case['uid']
        gt_persona = case['ground_truth']['persona']
        
        print(f"\n进展: [{i+1}/{total_cases}] | 处理 UID: {uid}")
        
        # --- 3. 断点续传逻辑 ---
        # 如果对应的日志文件夹已存在且包含总结文件，则跳过
        if os.path.exists(os.path.join(LOGS_ROOT, uid, "eval_summary.txt")):
            print(f"⏭️  检测到已完成，跳过。")
            continue

        try:
            # 重置 DPI 引擎状态
            if hasattr(assistant.bridge.dpi_engine, 'reset'):
                assistant.bridge.dpi_engine.reset()
            
            chat_history = []
            turn = 1
            final_response = None
            start_time = time.time()

            # --- 4. 对话循环 ---
            while turn <= MAX_TURNS:
                time.sleep(10) 
                user_text = user_sim.generate_response(case, chat_history)
                chat_history.append({"role": "user", "content": user_text})

                assistant_res = assistant.process_turn(user_text, turn)
                chat_history.append({"role": "assistant", "content": assistant_res['reply']})

                if assistant_res['status'] == 'TERMINATE' or turn == MAX_TURNS:
                    final_response = assistant_res
                    # 保存日志和对话历史
                    assistant.terminate_session(uid)
                    history_path = os.path.join(LOGS_ROOT, uid, "chat_history.json")
                    with open(history_path, "w", encoding="utf-8") as f:
                        json.dump(chat_history, f, ensure_ascii=False, indent=2)
                    break
                turn += 1

            # --- 5. 单例评估与报告生成 ---
            evaluation = evaluate_persona(final_response['persona'], gt_persona)
            match_count = sum(1 for res in evaluation.values() if res['match'])
            accuracy = (match_count / 3) * 100
            
            summary_lines = [
                f"📊 评估结果 (UID: {uid})",
                "-"*30
            ]
            for dim, res in evaluation.items():
                status = "✅ 匹配" if res['match'] else "❌ 错误"
                summary_lines.append(f"   - {dim:<11}: 推断={res['inferred']:<10} | 标准={res['gt']:<10} | {status}")
            
            summary_lines.extend([
                "="*50,
                f"🏁 结论报告",
                f"⏱️  收敛轮次: {turn}",
                f"标签匹配数: {match_count}/3",
                f"单例准确率: {accuracy:.1f}%"
            ])
            
            summary_text = "\n".join(summary_lines)
            with open(os.path.join(LOGS_ROOT, uid, "eval_summary.txt"), "w", encoding="utf-8") as f:
                f.write(summary_text)

            print(f"✅ 完成 Case {uid} | 轮次: {turn} | 准确率: {accuracy:.1f}%")
            all_summaries.append({"uid": uid, "turns": turn, "accuracy": accuracy})

        except Exception as e:
            print(f"⚠️ [Error] 处理 Case {uid} 时崩溃: {e}")
            traceback.print_exc()
            continue # 即使出错也继续跑下一个 Case

    # --- 6. 最终总体统计 ---
    if all_summaries:
        avg_acc = sum(r['accuracy'] for r in all_summaries) / len(all_summaries)
        avg_turns = sum(r['turns'] for r in all_summaries) / len(all_summaries)
        print("\n" + "🏆 全量实验汇总统计 " + "="*30)
        print(f"📊 总测试数: {len(all_summaries)}")
        print(f"🎯 平均偏好准确率: {avg_acc:.2f}%")
        print(f"⏱️  平均收敛轮次: {avg_turns:.2f}")
        print(f"📂 原始日志路径: {LOGS_ROOT}")
        print("="*50)
    


if __name__ == "__main__":
    run_evaluation()