# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Path Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
LOGS_QWEN_ROOT = os.path.join(current_dir, "logs_Qwen")
GT_DATA_PATH = os.path.join(current_dir, "data", "dpi_me_test_cases.json")
OUTPUT_DIR = os.path.join(current_dir, "analysis_report_qwen")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Visualization Settings ---
plt.switch_backend('agg')
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

def calculate_csr(uid, gt_info, log_path):
    """
    Calculate Constraint Satisfaction Rate (CSR) 
    by checking captured constraints in logs against GT.
    """
    constraints_file = os.path.join(log_path, "constraints_log.json")
    if not os.path.exists(constraints_file):
        return 0.0
    
    try:
        with open(constraints_file, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        
        # In AssistantQwen, we saved constraints as a list in a simulated format
        # Combine all strings in constraints list for fuzzy matching
        captured_text = ""
        for entry in log_data:
            captured_text += " ".join(entry.get("constraints", [])) + " "
        
        # Hard constraints to check
        checks = [
            str(gt_info['target_city']),
            str(gt_info['days']),
            str(gt_info['people_count'])
        ]
        
        satisfied = sum(1 for val in checks if val.lower() in captured_text.lower())
        return satisfied / len(checks)
    except:
        return 0.0

def parse_eval_summary(file_path):
    """Extract metrics from eval_summary.txt"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Robust regex for TTC and Accuracy
        turns = int(re.search(r'(?:Turns|收敛轮次)\s*[:：]\s*(\d+)', content).group(1))
        acc = float(re.search(r'(?:Accuracy|准确率)\s*[:：]\s*(\d+(?:\.\d+)?)', content).group(1)) / 100
        
        dim_results = {}
        for dim in ["Consumption", "Interest", "Pace"]:
            pattern = rf"-\s*{dim}\s*[:：].*?\|\s*(✅|❌)"
            m = re.search(pattern, content)
            dim_results[dim] = 1 if m and m.group(1) == "✅" else 0
            
        return turns, acc, dim_results
    except:
        return None

def run_analysis():
    # 1. Load Ground Truth
    with open(GT_DATA_PATH, 'r', encoding='utf-8') as f:
        gt_cases = {c['uid']: c for c in json.load(f)}

    case_dirs = [d for d in os.listdir(LOGS_QWEN_ROOT) if os.path.isdir(os.path.join(LOGS_QWEN_ROOT, d))]
    
    results = []
    
    print(f"Starting analysis of {len(case_dirs)} Qwen baseline logs...")

    for uid in case_dirs:
        if uid not in gt_cases: continue
        
        case_path = os.path.join(LOGS_QWEN_ROOT, uid)
        summary_txt = os.path.join(case_path, "eval_summary.txt")
        
        if not os.path.exists(summary_txt): continue

        # Parse metrics
        parsed = parse_eval_summary(summary_txt)
        if not parsed: continue
        turns, acc, dim_matches = parsed

        # Calculate CSR
        csr = calculate_csr(uid, gt_cases[uid]['ground_truth'], case_path)

        res_item = {
            "uid": uid,
            "ttc": turns,
            "accuracy": acc,
            "csr": csr
        }
        res_item.update(dim_matches)
        results.append(res_item)

    if not results:
        print("❌ No valid data found to analyze.")
        return

    df = pd.DataFrame(results)

    # --- 2. Calculate Final Metrics ---
    cases_processed = len(df)
    overall_acc = df['accuracy'].mean()
    avg_ttc = df['ttc'].mean()
    avg_csr = df['csr'].mean()
    # Efficiency defined as Accuracy per Turn
    conv_efficiency = overall_acc / avg_ttc if avg_ttc > 0 else 0
    
    dim_acc = {d: df[d].mean() * 100 for d in ["Consumption", "Interest", "Pace"]}

    # --- 3. Print Text Report ---
    print("\n" + "="*50)
    print("📊 Qwen Baseline (Vanilla LLM) Performance Report")
    print("-" * 50)
    print(f"Test Cases Processed            : {cases_processed}")
    print(f"Overall Accuracy (Persona)      : {overall_acc:.2%}")
    print(f"Constraint Satisfaction (CSR)   : {avg_csr:.2%}")
    print(f"Avg Turns to Convergence (TTC)  : {avg_ttc:.2f} turns")
    print(f"Convergence Efficiency (Acc/TTC): {conv_efficiency:.4f}")
    print("="*50)

    # --- 4. Plotting ---
    plt.figure(figsize=(14, 6))

    # Plot A: TTC Distribution
    plt.subplot(1, 2, 1)
    sns.histplot(df['ttc'], bins=range(1, 10), color='salmon', discrete=True)
    plt.title("Turns to Convergence (TTC) Distribution", fontsize=13)
    plt.xlabel("Number of Turns")
    plt.ylabel("Count")

    # Plot B: Accuracy by Dimension
    plt.subplot(1, 2, 2)
    acc_data = pd.DataFrame({"Dimension": list(dim_acc.keys()), "Accuracy": list(dim_acc.values())})
    ax = sns.barplot(x="Dimension", y="Accuracy", data=acc_data, palette="magma")
    plt.ylim(0, 110)
    plt.title("Accuracy by Preference Dimension", fontsize=13)
    plt.ylabel("Accuracy (%)")
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qwen_baseline_performance.png"), dpi=300)
    
    # Save CSV
    df.to_csv(os.path.join(OUTPUT_DIR, "qwen_metrics_raw.csv"), index=False)
    print(f"\n✅ Analysis complete! Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_analysis()