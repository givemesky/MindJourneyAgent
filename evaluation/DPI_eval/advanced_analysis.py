# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# --- Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
LOGS_ROOT = os.path.join(current_dir, "logs")
GT_DATA_PATH = os.path.join(current_dir, "data", "dpi_me_test_cases.json")
OUTPUT_DIR = os.path.join(current_dir, "analysis_report")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.switch_backend('agg')
sns.set_theme(style="white")

def calculate_csr(uid, gt_info, log_path):
    """计算约束满足率 (CSR)"""
    constraints_file = os.path.join(log_path, "constraints_log.json")
    if not os.path.exists(constraints_file):
        return 0.0
    
    try:
        with open(constraints_file, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
        
        all_extracted_text = ""
        for turn in extracted_data:
            all_extracted_text += " ".join(turn.get("constraints", [])) + " "
        
        checks = {
            "city": str(gt_info['target_city']),
            "days": str(gt_info['days']),
            "people": str(gt_info['people_count'])
        }
        
        satisfied = 0
        for key, val in checks.items():
            if val in all_extracted_text:
                satisfied += 1
        return satisfied / len(checks)
    except:
        return 0.0

def run_advanced_analysis():
    # 1. 加载 Ground Truth 数据
    if not os.path.exists(GT_DATA_PATH):
        print(f"❌ Error: Ground Truth file not found at {GT_DATA_PATH}")
        return

    with open(GT_DATA_PATH, 'r', encoding='utf-8') as f:
        gt_cases = {c['uid']: c for c in json.load(f)}

    if not os.path.exists(LOGS_ROOT):
        print(f"❌ Error: Logs directory not found at {LOGS_ROOT}")
        return

    case_dirs = [d for d in os.listdir(LOGS_ROOT) if os.path.isdir(os.path.join(LOGS_ROOT, d))]
    print(f"🔎 Found {len(case_dirs)} potential case folders in logs.")

    metrics_list = []
    conf_evolution_matrix = [] 

    for uid in case_dirs:
        if uid not in gt_cases:
            continue
        
        case_path = os.path.join(LOGS_ROOT, uid)
        evo_json = os.path.join(case_path, "persona_evolution.json")
        sum_txt = os.path.join(case_path, "eval_summary.txt")
        
        if not os.path.exists(evo_json) or not os.path.exists(sum_txt):
            continue

        # A. 计算 CSR
        csr = calculate_csr(uid, gt_cases[uid]['ground_truth'], case_path)

        # B. 提取置信度演化
        try:
            with open(evo_json, 'r', encoding='utf-8') as f:
                evo_data = json.load(f)
            gt_labels = gt_cases[uid]['ground_truth']['persona']
            case_conf_history = []
            for turn_idx in range(1, 9):
                turn_key = str(turn_idx)
                if turn_key in evo_data:
                    conf_sum = 0
                    for dim in ["Consumption", "Interest", "Pace"]:
                        target_label = gt_labels[dim]
                        dist = evo_data[turn_key].get(dim, {}).get("distribution", {})
                        conf_sum += dist.get(target_label, 0.33) # 默认 1/3 均匀分布
                    case_conf_history.append(conf_sum / 3)
                else:
                    last_val = case_conf_history[-1] if case_conf_history else 0.33
                    case_conf_history.append(last_val)
            conf_evolution_matrix.append(case_conf_history)
        except Exception as e:
            print(f"⚠️ Error parsing evolution for {uid}: {e}")
            continue

        # C. 提取收敛轮次和准确率 (兼容中英文)
        try:
            with open(sum_txt, 'r', encoding='utf-8') as f:
                content = f.read()
                # 兼容 "收敛轮次" 或 "Turns"
                t_match = re.search(r'(?:收敛轮次|Turns|turn|收敛轮次)\s*[:：]\s*(\d+)', content)
                turns = int(t_match.group(1)) if t_match else 8
                
                # 兼容 "准确率" 或 "Accuracy"
                a_match = re.search(r'(?:准确率|Accuracy|偏好推理准确率)\s*[:：]\s*(\d+(?:\.\d+)?)', content)
                acc = float(a_match.group(1)) / 100 if a_match else 0.0

            metrics_list.append({"uid": uid, "ttc": turns, "accuracy": acc, "csr": csr})
        except:
            continue

    if not metrics_list:
        print("❌ No valid experiment data parsed. Check if eval_summary.txt is empty or mismatched.")
        return

    # --- Data Aggregation ---
    df = pd.DataFrame(metrics_list)
    avg_accuracy = df['accuracy'].mean()
    avg_ttc = df['ttc'].mean()
    avg_csr = df['csr'].mean()
    conv_efficiency = avg_accuracy / avg_ttc if avg_ttc > 0 else 0

    print("\n" + "="*50)
    print("📈 Advanced Performance Report")
    print("-" * 50)
    print(f"Test Cases Processed            : {len(df)}")
    print(f"Overall Accuracy (Persona)      : {avg_accuracy:.2%}")
    print(f"Constraint Satisfaction (CSR)   : {avg_csr:.2%}")
    print(f"Avg Turns to Convergence (TTC)  : {avg_ttc:.2f} turns")
    print(f"Convergence Efficiency (Acc/TTC): {conv_efficiency:.4f}")
    print("="*50)

    # --- Plot 1: Heatmap ---
    plt.figure(figsize=(12, 8))
    heat_df = pd.DataFrame(conf_evolution_matrix, columns=[f"Turn {i}" for i in range(1, 9)])
    heat_df = heat_df.sort_values(by=f"Turn 8", ascending=False)
    sns.heatmap(heat_df, cmap="YlGnBu", annot=False, cbar_kws={'label': 'Confidence'})
    plt.title("System Confidence Evolution Heatmap (EN)", fontsize=14)
    plt.xlabel("Dialogue Turns")
    plt.ylabel("Test Cases")
    plt.savefig(os.path.join(OUTPUT_DIR, "confidence_heatmap.png"), dpi=300)

    # --- Plot 2: Bar Chart ---
    plt.figure(figsize=(8, 6))
    summary_data = {"Metric": ["Accuracy", "CSR"], "Score": [avg_accuracy * 100, avg_csr * 100]}
    ax = sns.barplot(x="Metric", y="Score", data=summary_data, palette="viridis")
    plt.ylim(0, 110)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 8),
                    textcoords='offset points', fontweight='bold')
    plt.title("Overall Accuracy vs. Constraint Satisfaction")
    plt.savefig(os.path.join(OUTPUT_DIR, "overall_performance.png"), dpi=300)

    print(f"\n✅ Done! Check results in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_advanced_analysis()