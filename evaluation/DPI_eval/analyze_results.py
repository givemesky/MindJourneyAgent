# -*- coding: utf-8 -*-
import os
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Path Configuration ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
LOGS_ROOT = os.path.join(current_script_dir, "logs")
OUTPUT_DIR = os.path.join(current_script_dir, "analysis_report")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Visualization Configuration (English Only) ---
plt.switch_backend('agg')  # For headless server environments
sns.set_theme(style="whitegrid")
# Standard fonts usually available on most Linux systems
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

def parse_summary_txt(file_path):
    """Parse eval_summary.txt with robust regex for both EN/CN punctuation"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract Turns to Convergence
        turns_match = re.search(r'(?:收敛轮次|Turns to Convergence)\s*[:：]\s*(\d+)', content)
        turns = int(turns_match.group(1)) if turns_match else 0
        
        # Extract Accuracy
        acc_match = re.search(r'(?:偏好推理准确率|Inference Accuracy)\s*[:：]\s*(\d+\.\d+)%', content)
        accuracy = float(acc_match.group(1)) if acc_match else 0.0

        # Extract dimension match status
        dim_results = {"Consumption": 0, "Interest": 0, "Pace": 0}
        for dim in ["Consumption", "Interest", "Pace"]:
            # Pattern matches "- Dimension : ... | ✅" or "- Dimension: ... | ❌"
            pattern = rf"-\s*{dim}\s*[:：].*?\|\s*(✅|❌)"
            m = re.search(pattern, content)
            if m:
                dim_results[dim] = 1 if m.group(1) == "✅" else 0
                
        return turns, accuracy, dim_results
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return 0, 0.0, {"Consumption": 0, "Interest": 0, "Pace": 0}

def run_analysis():
    if not os.path.exists(LOGS_ROOT):
        print(f"Error: Log directory not found at {LOGS_ROOT}")
        return

    all_case_dirs = [d for d in os.listdir(LOGS_ROOT) if os.path.isdir(os.path.join(LOGS_ROOT, d))]
    
    summary_data = []
    entropy_series = []

    print(f"Start analyzing {len(all_case_dirs)} experiment logs...")

    for uid in all_case_dirs:
        case_path = os.path.join(LOGS_ROOT, uid)
        summary_txt = os.path.join(case_path, "eval_summary.txt")
        convergence_csv = os.path.join(case_path, "persona_convergence.csv")

        if not os.path.exists(summary_txt):
            continue

        # 1. Process Text Summary
        turns, acc, dim_matches = parse_summary_txt(summary_txt)
        res = {"uid": uid, "turns": turns, "accuracy": acc}
        res.update(dim_matches)
        summary_data.append(res)

        # 2. Process Convergence CSV for Entropy Decay
        if os.path.exists(convergence_csv):
            try:
                df_conv = pd.read_csv(convergence_csv)
                cols = ['Consumption_entropy', 'Interest_entropy', 'Pace_entropy']
                if all(c in df_conv.columns for c in cols):
                    df_conv['total_entropy'] = df_conv[cols].sum(axis=1)
                    for _, row in df_conv.iterrows():
                        entropy_series.append({
                            "turn": int(row['turn']),
                            "entropy": row['total_entropy']
                        })
            except Exception as e:
                print(f"Warning: Failed to process {convergence_csv}: {e}")

    if not summary_data:
        print("Error: No valid data found in logs.")
        return

    # --- Data Aggregation ---
    df_results = pd.DataFrame(summary_data)
    df_entropy = pd.DataFrame(entropy_series)

    # Calculate global metrics
    avg_ttc = df_results['turns'].mean()
    total_acc = df_results['accuracy'].mean()
    dim_acc = {d: df_results[d].mean() * 100 for d in ["Consumption", "Interest", "Pace"]}

    # --- Print English Report to Terminal ---
    print("\n" + "="*50)
    print("📊 MindJourney DPI-ME Experiment Summary")
    print("-" * 50)
    print(f"Total Test Cases      : {len(df_results)}")
    print(f"Avg Turns to Converge : {avg_ttc:.2f}")
    print(f"Overall Accuracy      : {total_acc:.2f}%")
    print("-" * 50)
    for d, a in dim_acc.items():
        print(f"Accuracy [{d:<11}] : {a:.2f}%")
    print("="*50)

    # --- Visualization Rendering ---
    fig = plt.figure(figsize=(14, 10))

    # Plot 1: TTC Distribution (Histogram)
    ax1 = plt.subplot(2, 2, 1)
    if not df_results['turns'].empty:
        sns.histplot(data=df_results, x='turns', bins=range(1, 10), color='skyblue', discrete=True, ax=ax1)
    ax1.set_title("Turns to Convergence (TTC) Distribution", fontsize=14)
    ax1.set_xlabel("Number of Turns")
    ax1.set_ylabel("Count of Cases")

    # Plot 2: Dimensional Accuracy (Bar Chart)
    ax2 = plt.subplot(2, 2, 2)
    acc_df = pd.DataFrame({"Dimension": list(dim_acc.keys()), "Accuracy": list(dim_acc.values())})
    sns.barplot(data=acc_df, x="Dimension", y="Accuracy", palette="viridis", ax=ax2)
    ax2.set_ylim(0, 110)
    ax2.set_title("Accuracy by Preference Dimension", fontsize=14)
    ax2.set_ylabel("Accuracy (%)")
    for i, v in enumerate(acc_df['Accuracy']):
        ax2.text(i, v + 1.5, f"{v:.1f}%", ha='center', fontweight='bold')

    # Plot 3: Entropy Decay Curve (The core proof of active learning)
    ax3 = plt.subplot(2, 1, 2)
    if not df_entropy.empty:
        sns.lineplot(data=df_entropy, x='turn', y='entropy', marker='o', errorbar='sd', color='navy', ax=ax3)
        ax3.set_title("Average System Total Entropy Decay over Turns", fontsize=14)
        ax3.set_xlabel("Dialogue Turn")
        ax3.set_ylabel("Total Entropy (bits)")
        max_turn = int(df_entropy['turn'].max())
        ax3.set_xticks(range(1, max_turn + 1))
    else:
        ax3.text(0.5, 0.5, "Missing Entropy Data", ha='center')

    plt.tight_layout()
    plot_filename = "dpi_experiment_results_en.png"
    plt.savefig(os.path.join(OUTPUT_DIR, plot_filename), dpi=300)
    
    # Save raw metrics to CSV for further reference
    df_results.to_csv(os.path.join(OUTPUT_DIR, "final_metrics_summary.csv"), index=False)
    
    print(f"\nAnalysis complete!")
    print(f"Charts saved to: {os.path.join(OUTPUT_DIR, plot_filename)}")
    print(f"CSV data saved to: {os.path.join(OUTPUT_DIR, 'final_metrics_summary.csv')}")

if __name__ == "__main__":
    run_analysis()