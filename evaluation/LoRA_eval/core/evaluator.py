import sys
import os
import json
import re
import argparse
from tqdm import tqdm

# 1. 处理路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.user_simulator import UserSimulator
from agents.assistant_agent import MindJourneyAssistant

# --- 配置参数 ---
DATA_PATH = "/public/home/azsy/MindJourney_LoRA/MindJourneyAgent/evaluation/LoRA_eval/data/processed_data/test_personas.json"
RESULTS_BASE_DIR = "/public/home/azsy/MindJourney_LoRA/MindJourneyAgent/evaluation/LoRA_eval/results"
MAX_TURNS = 8

# API 配置 (建议替换为你的实际配置)
API_KEY = "AIzaSyCfU8v-9LTXMT0-xWtGf503GaudDh55FDs"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

def is_mining_complete(assistant_output):
    """判定 4 个维度是否已集齐"""
    summary_match = re.search(r'<summary>(.*?)</summary>', assistant_output, re.DOTALL)
    if not summary_match:
        return False
    try:
        summary_data = json.loads(summary_match.group(1).strip())
        core_keys = ["Destination", "Consumption", "Interest", "Pace"]
        for key in core_keys:
            val = summary_data.get(key)
            if val is None or str(val).lower() in ["null", "none", "未知", ""]:
                return False
        return True
    except:
        return False

def parse_summary(assistant_output):
    """解析最后一轮的 summary 用于后续分析"""
    summary_match = re.search(r'<summary>(.*?)</summary>', assistant_output, re.DOTALL)
    if summary_match:
        try:
            return json.loads(summary_match.group(1).strip())
        except:
            return {}
    return {}

def run_evaluation(mode, limit=5):
    # 2. 检查并创建结果目录
    output_dir = os.path.join(RESULTS_BASE_DIR, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. 加载测试画像
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        all_personas = json.load(f)
    
    # 获取前 limit 个 Case ID
    case_ids = list(all_personas.keys())[:limit]
    print(f"--- 启动实验组: {mode.upper()} | 样本数: {len(case_ids)} ---")

    # 4. 初始化助手模型 (只加载一次)
    assistant = MindJourneyAssistant(mode=mode)

    # 5. 开始批处理循环
    for cid in tqdm(case_ids, desc=f"Processing {mode}"):
        persona_data = all_personas[cid]
        
        # 初始化 User Simulator
        user_sim = UserSimulator(
            persona_data=persona_data,
            api_key=API_KEY,
            base_url=BASE_URL
        )

        conversation_history = []
        trajectory = []
        is_completed = False
        final_ttc = MAX_TURNS

        # 模拟对话循环
        for turn in range(1, MAX_TURNS + 1):
            # A. User 发言
            user_reply = user_sim.get_response(conversation_history)
            
            # B. Assistant 发言
            # 注意：在获取回复前，先把 user_reply 加进 history
            temp_history = conversation_history + [{"role": "user", "content": user_reply}]
            assistant_reply = assistant.generate_response(temp_history)
            
            # 记录轨迹
            turn_data = {
                "turn": turn,
                "user_content": user_reply,
                "assistant_raw": assistant_reply,
                "parsed_summary": parse_summary(assistant_reply)
            }
            trajectory.append(turn_data)
            
            # 更新历史用于下一轮
            conversation_history.append({"role": "user", "content": user_reply})
            conversation_history.append({"role": "assistant", "content": assistant_reply})

            # C. 判定结束
            if is_mining_complete(assistant_reply):
                is_completed = True
                final_ttc = turn
                break
        
        # 6. 保存单例日志
        log_data = {
            "persona_id": cid,
            "mode": mode,
            "ground_truth": persona_data['ground_truth'],
            "final_stats": {
                "is_completed": is_completed,
                "ttc": final_ttc
            },
            "trajectory": trajectory
        }
        
        log_file = os.path.join(output_dir, f"log_{cid}.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)

    print(f"\n[DONE] {mode} 组测试完成，结果已保存至: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindJourney 评估主程序")
    parser.add_argument("--mode", type=str, required=True, choices=["base", "few_shot", "lora"], help="实验组选择")
    parser.add_argument("--limit", type=int, default=5, help="运行的样本数量")
    
    args = parser.parse_args()
    
    run_evaluation(mode=args.mode, limit=args.limit)