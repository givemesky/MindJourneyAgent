# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- 绘图配置 ---
plt.switch_backend('agg')
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# 配色方案 (参照图1: 浅蓝、中蓝、绿、珊瑚橙)
# DPI-ME 使用蓝色系，Baseline 使用橙红色系或对比色
COLORS_DPI = ["#3d5a80", "#2a9d8f", "#90be6d"] # 深蓝、青绿、嫩绿
COLORS_BASE = ["#ee6c4d", "#f9c74f", "#f94144"] # 珊瑚橙、淡黄、红
DPI_THEME_COLOR = "#98d8ed" # 图1左侧的浅蓝色
BASE_THEME_COLOR = "#fca5a5" # 图2左侧的浅红色

OUTPUT_DIR = "evaluation/DPI_eval/analysis_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_combined_results():
    # 1. 准备 Accuracy 维度对比数据 (基于您的实验结果)
    acc_data = {
        "Dimension": ["Consumption", "Interest", "Pace", "Consumption", "Interest", "Pace"],
        "Accuracy": [84.0, 98.0, 76.0, 52.0, 66.0, 60.0],
        "Model": ["MindJourney (DPI-ME)", "MindJourney (DPI-ME)", "MindJourney (DPI-ME)", 
                  "Qwen Baseline", "Qwen Baseline", "Qwen Baseline"]
    }
    df_acc = pd.DataFrame(acc_data)

    # 2. 模拟 TTC 分布数据 (根据您的直方图趋势还原)
    # DPI-ME: 集中在 4 轮左右
    dpi_ttc = [3]*9 + [4]*27 + [5]*10 + [6]*4
    # Baseline: 分散在 2, 3, 4, 8 轮
    base_ttc = [2]*14 + [3]*12 + [4]*7 + [5]*1 + [6]*2 + [8]*14 
    
    df_ttc = pd.DataFrame({
        "Turns": dpi_ttc + base_ttc,
        "Model": ["MindJourney (DPI-ME)"]*50 + ["Qwen Baseline"]*50
    })

    # --- 开始绘图 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 图 A: TTC Distribution (对比直方图)
    sns.histplot(data=df_ttc, x="Turns", hue="Model", multiple="dodge", 
                 shrink=0.8, palette=[DPI_THEME_COLOR, BASE_THEME_COLOR], ax=axes[0])
    axes[0].set_title("Comparison of Turns to Convergence (TTC) Distribution", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Number of Dialogue Turns")
    axes[0].set_ylabel("Case Count")

    # 图 B: Dimensional Accuracy (分组柱状图)
    ax_b = sns.barplot(data=df_acc, x="Dimension", y="Accuracy", hue="Model", 
                       palette=[COLORS_DPI[0], COLORS_BASE[0]], ax=axes[1])
    
    # 添加数值标签
    for p in ax_b.patches:
        if p.get_height() > 0:
            ax_b.annotate(f'{p.get_height():.1f}%', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', fontsize=11, color='black', 
                        xytext=(0, 7), textcoords='offset points', fontweight='bold')

    axes[1].set_title("Accuracy Comparison by Preference Dimension", fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 115)
    axes[1].set_ylabel("Accuracy (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "system_comparison_en.png"), dpi=300)
    print(f"✅ Comparison charts saved to {OUTPUT_DIR}/system_comparison_en.png")

if __name__ == "__main__":
    plot_combined_results()