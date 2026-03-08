# ===================================================
#  贝叶斯与信息熵算法测试脚本
# ===================================================

import os
import sys
import pprint
import time

# ===================================================
# 1. 环境准备
# ===================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from dpi_reason.manager import DPISystemManager
    from dpi_reason.evaluator import PreferenceEvaluator  # <--- 导入新写的评估器
    from dpi_likelihood.reasoning.schema import PREFERENCE_SCHEMA
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def run_persona_evolution_test():
    """
    集成测试：验证贝叶斯算法在多轮对话中的偏好收敛与熵减过程
    """
    print("\n" + "="*70)
    print("🚀 [集成测试] 开始验证：DPI-ME 偏好推理与收敛性分析")
    print("="*70)

    # 1. 初始化系统
    manager = DPISystemManager(model_name="Qwen2.5-7B-Instruct")
    # 2. 初始化评估器（用于记录熵和确定度）
    evaluator = PreferenceEvaluator(log_dir="dpi_me/logs")

    # 3. 测试场景
    test_scenario = [
        {"input": "我想从深圳到成都玩5天，希望总预算不超过5000。"},
        {"input": "我想去太古里和杜甫草堂，全程公共交通为主。"},
        {"input": "休闲一点，不要太赶。"}
    ]

    print(f"\n初始信念状态 (Prior): 均匀分布，不确定度最高")

    # 4. 开始模拟对话
    for i, turn in enumerate(test_scenario):
        user_text = turn["input"]
        turn_id = i + 1
        
        start_time = time.time()
        # 执行核心全链路逻辑 (Perception -> Reasoning -> State Update)
        manager.process_turn(user_text, turn_id)
        elapsed = time.time() - start_time

        # --- 【新增逻辑】计算收敛性指标 ---
        # 记录本轮各维度的熵值和置信度
        turn_metrics = evaluator.record_turn(turn_id, manager.persona)
        
        # 5. 打印详细推理报告
        print(f"\n>>> Turn {turn_id} 处理完成 (耗时: {elapsed:.2f}s)")
        print(f"用户表达: \"{user_text}\"")
        
        # 表格表头增加 熵(Entropy) 和 确定度(Confidence %)
        header = f"{'维度 (Dimension)':<12} | {'推断标签':<10} | {'置信度':<8} | {'熵(Entropy)':<10} | {'收敛确定度'}"
        print("-" * 75)
        print(header)
        print("-" * 75)

        for dim in ["Consumption", "Interest", "Pace"]:
            # 从 manager.persona.beliefs 获取原始概率分布计算
            prob_dist = manager.persona.beliefs[dim]
            entropy = evaluator.calculate_entropy(prob_dist)
            certainty = evaluator.calculate_certainty(prob_dist)
            
            # 获取当前最大概率的标签
            top_label = manager.persona.get_top_label(dim) 
            top_prob = max(prob_dist)

            status_icon = "🎯" if certainty > 0.7 else "⏳"
            print(f"{dim:<12} | {top_label:<10} | {top_prob:<8.3f} | {entropy:<10.4f} | {certainty:>8.2%} {status_icon}")

    # 6. 保存实验数据并生成报告
    evaluator.save_to_csv("persona_convergence_test.csv")
    final_report = manager.generate_final_report()
    
    print("\n" + "="*70)
    print("🎉 测试结束：")
    print(f"1. 画像演化路径已记录。")
    print(f"2. 收敛性分析报表已生成: dpi_me/logs/persona_convergence_test.csv")
    print("="*70)

if __name__ == "__main__":
    run_persona_evolution_test()