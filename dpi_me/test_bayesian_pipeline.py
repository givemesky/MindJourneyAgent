# ===================================================
#  贝叶斯算法测试脚本
# ===================================================

import os
import sys
import pprint
import time

# ===================================================
# 1. 环境准备：确保能找到本地模块
# ===================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入你刚才编写的系统中枢
try:
    from dpi_reason.manager import DPISystemManager
    from dpi_likelihood.reasoning.schema import PREFERENCE_SCHEMA
except ImportError as e:
    print(f"❌ 导入失败，请检查目录结构和 __init__.py 文件: {e}")
    sys.exit(1)

def run_persona_evolution_test():
    """
    测试目标：验证贝叶斯算法是否能根据多轮对话正确累积证据并收敛画像
    """
    print("\n" + "="*60)
    print("🚀 [集成测试] 开始验证：贝叶斯偏好推理全链路")
    print("="*60)

    # 1. 初始化系统（此步会加载 Qwen 模型）
    manager = DPISystemManager(model_name="Qwen2.5-7B-Instruct")

    # 2. 定义一个测试场景：一个“省钱、想看风景、节奏慢”的用户
    test_scenario = [
        # {"input": "我想从杭州去上海玩3天，预算比较吃紧，想找200元左右的住处。", "goal": "Consumption -> Budget"},
        # {"input": "我想去看看外滩和自然公园，不打算打车，全程走路或者地铁。", "goal": "Interest -> Nature/Culture, Pace -> Relaxed"},
        # {"input": "我希望行程不要太赶，一天逛一两个地方就行了。", "goal": "Pace -> Relaxed (强化)"}
        {"input": "我想从深圳到成都玩5天，希望总预算不超过5000。"},
        {"input": "我想去太古里和杜甫草堂，全程公共交通为主。"},
        {"input": "休闲一点。"}
    ]

    print(f"\n初始信念状态 (Prior):")
    # 打印初始状态（应该是均匀分布）
    pprint.pprint(manager.persona.get_top_persona())

    # 3. 开始模拟对话
    for i, turn in enumerate(test_scenario):
        user_text = turn["input"]
        turn_id = i + 1
        
        start_time = time.time()
        # 执行核心全链路逻辑
        current_persona = manager.process_turn(user_text, turn_id)
        elapsed = time.time() - start_time

        # 4. 打印本轮推理结论
        print(f"\n>>> Turn {turn_id} 处理完成 (耗时: {elapsed:.2f}s)")
        print(f"用户表达: {user_text}")
        
        print(f"{'维度 (Dimension)':<15} | {'推断标签 (Label)':<12} | {'置信度 (Prob)':<8} | {'状态'}")
        print("-" * 60)
        for dim, info in current_persona.items():
            status = "✅ 已收敛" if info['is_stable'] else "⏳ 探测中"
            print(f"{dim:<15} | {info['label']:<12} | {info['confidence']:<8.3f} | {status}")

    # 5. 生成实验报告
    final_report = manager.generate_final_report()
    print("\n" + "="*60)
    print("🎉 测试结束：偏好画像演化路径已记录。")
    print("="*60)

if __name__ == "__main__":
    # 执行测试
    run_persona_evolution_test()