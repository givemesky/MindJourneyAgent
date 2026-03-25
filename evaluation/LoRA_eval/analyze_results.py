import os
import json
import re
import pandas as pd
from collections import defaultdict

# --- 配置路径 ---
BASE_DIR = "/public/home/azsy/MindJourney_LoRA/MindJourneyAgent/evaluation/LoRA_eval"
PERSONA_PATH = os.path.join(BASE_DIR, "data/processed_data/test_personas.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODES = ["base", "few_shot", "lora"]
CORE_KEYS = ["Destination", "Consumption", "Interest", "Pace"]

def normalize_label(val):
    """
    标签归一化：处理模型输出的随机性
    1. 处理 JSON 的 null 和 字符串 "null"
    2. 处理中英文对齐 (Beijing -> 北京)
    """
    if val is None:
        return "null"
    s = str(val).strip().lower()
    if s in ["null", "none", "未知", ""]:
        return "null"
    if s in ["beijing", "北京"]:
        return "北京"
    return s

class MindJourneyAnalyzer:
    def __init__(self):
        # 加载画像原始数据
        with open(PERSONA_PATH, 'r', encoding='utf-8') as f:
            self.personas = json.load(f)
        print(f"成功加载 {len(self.personas)} 条画像基准数据。")

    def analyze_case(self, file_path):
        """分析单个对话 JSON 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            log = json.load(f)
        
        case_id = log['persona_id']
        gt = log['ground_truth']
        trajectory = log['trajectory']
        
        # 1. 格式稳定性检查 (检查最后一轮是否有 summary)
        last_turn = trajectory[-1]
        has_summary = bool(last_turn.get('parsed_summary'))
        
        # 2. 计算 TTC (第一次集齐 4 个维度的轮次)
        ttc = 8  # 默认最大轮次
        for turn_data in trajectory:
            summary = turn_data.get('parsed_summary', {})
            filled_count = 0
            for k in CORE_KEYS:
                if normalize_label(summary.get(k)) != "null":
                    filled_count += 1
            if filled_count == 4:
                ttc = turn_data['turn']
                break
        
        # 3. 计算最后一轮的 Recall 和 Accuracy
        final_summary = last_turn.get('parsed_summary', {})
        recall_count = 0
        acc_count = 0
        
        for k in CORE_KEYS:
            pred = normalize_label(final_summary.get(k))
            target = normalize_label(gt.get(k))
            
            if pred != "null":
                recall_count += 1
            if pred == target:
                acc_count += 1

        # 4. 主动性检查 (第一轮是否发起了追问)
        # 逻辑：第一轮的 assistant_raw 中是否包含思考过程，且 summary 至少识别了目的地
        is_proactive = False
        if len(trajectory) > 0:
            first_summary = trajectory[0].get('parsed_summary', {})
            if normalize_label(first_summary.get('Destination')) == "北京":
                is_proactive = True

        return {
            "case_id": case_id,
            "recall": recall_count / 4.0,
            "accuracy": acc_count / 4.0,
            "ttc": ttc,
            "is_completed": log['final_stats']['is_completed'],
            "is_proactive": is_proactive,
            "has_summary": has_summary
        }

    def run(self):
        comparison_table = []

        for mode in MODES:
            mode_dir = os.path.join(RESULTS_DIR, mode)
            if not os.path.exists(mode_dir):
                print(f"警告：目录 {mode_dir} 不存在，跳过。")
                continue

            case_files = [f for f in os.listdir(mode_dir) if f.endswith('.json')]
            mode_results = []
            
            for fname in case_files:
                fpath = os.path.join(mode_dir, fname)
                try:
                    res = self.analyze_case(fpath)
                    mode_results.append(res)
                except Exception as e:
                    print(f"解析文件 {fname} 失败: {e}")

            if not mode_results:
                continue

            # 计算平均指标
            df = pd.DataFrame(mode_results)
            comparison_table.append({
                "实验组": mode.upper(),
                "样本量": len(df),
                "平均召回率(Recall)": f"{df['recall'].mean():.2%}",
                "平均准确率(Accuracy)": f"{df['accuracy'].mean():.2%}",
                "平均收敛步数(TTC)": f"{df['ttc'].mean():.2f}",
                "任务完成率": f"{(df['ttc'] < 8).mean():.2%}",
                "格式合格率": f"{df['has_summary'].mean():.2%}",
                "首轮主动性": f"{df['is_proactive'].mean():.2%}"
            })

        # 打印对比报表
        final_df = pd.DataFrame(comparison_table)
        print("\n" + "="*30 + " MindJourney 实验对比报表 " + "="*30)
        print(final_df.to_markdown(index=False))
        print("="*86)
        
        # 保存为 CSV 方便画图
        output_csv = os.path.join(BASE_DIR, "evaluation_report.csv")
        final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n分析报告已保存至: {output_csv}")

if __name__ == "__main__":
    analyzer = MindJourneyAnalyzer()
    analyzer.run()