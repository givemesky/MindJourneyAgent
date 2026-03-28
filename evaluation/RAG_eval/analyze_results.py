# -*- coding: utf-8 -*-
import pandas as pd
import os

# --- 1. 路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
summary_path = os.path.join(current_dir, "logs", "rag_eval_summary.csv")

def analyze():
    if not os.path.exists(summary_path):
        print(f"❌ 错误：找不到汇总文件 {summary_path}，请先运行评测脚本。")
        return

    # 加载汇总数据
    df = pd.read_csv(summary_path)

    # 定义难度顺序
    diff_order = ['Easy', 'Standard', 'Hard']
    
    print("="*60)
    print("📊 MindJourney RAG 规划系统实验数据分析报告")
    print("="*60)

    results_table = []

    for diff in diff_order:
        # 1. 筛选当前难度的数据
        sub_df = df[df['difficulty'] == diff]
        if sub_df.empty:
            continue

        total_count = len(sub_df)

        # 2. [实验 A] 全集约束满足率 (Global CSR) - 直接计算平均值
        avg_csr = sub_df['global_csr'].mean()

        # 3. [实验 C] 方案二：兴趣画像契合度
        # 规则：排除 -1.0 (无体现/未推荐额外景点) 的样本
        interest_valid = sub_df[sub_df['interest_precision'] != -1.0]
        if not interest_valid.empty:
            avg_interest = interest_valid['interest_precision'].mean()
            interest_sample_size = len(interest_valid)
        else:
            avg_interest = 0.0
            interest_sample_size = 0

        # 4. [实验 D] 方案三：消费定制测试符合率
        # 规则：我们在批量脚本中，如果正则失败 cons_ok=0，daily_avg=0，符合“按不符合处理”的要求
        # 计算 cons_ok 为 1 的比例
        success_cons_count = sub_df['cons_ok'].sum()
        cons_success_rate = success_cons_count / total_count

        # 5. 存储结果
        results_table.append({
            "难度等级": diff,
            "测试样本数": total_count,
            "Exp A:平均CSR": f"{avg_csr:.2%}",
            "Exp C:有效样本数": interest_sample_size,
            "Exp C:平均兴趣契合度": f"{avg_interest:.2%}",
            "Exp D:消费符合率": f"{cons_success_rate:.2%}"
        })

    # --- 打印结果表格 ---
    final_df = pd.DataFrame(results_table)
    print(final_df.to_string(index=False))
    
    print("-" * 60)
    # 计算总平均 (不区分难度)
    total_avg_csr = df['global_csr'].mean()
    
    total_interest_valid = df[df['interest_precision'] != -1.0]
    total_avg_interest = total_interest_valid['interest_precision'].mean() if not total_interest_valid.empty else 0
    
    total_cons_rate = df['cons_ok'].sum() / len(df)

    print(f"🌍 全局平均统计 (Total):")
    print(f"   > [实验 A] 全集约束满足率: {total_avg_csr:.2%}")
    print(f"   > [实验 C] 兴趣契合度 (有效样本): {total_avg_interest:.2%}")
    print(f"   > [实验 D] 消费定制符合率: {total_cons_rate:.2%}")
    print("="*60)

if __name__ == "__main__":
    analyze()