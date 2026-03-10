# -*- coding: utf-8 -*-
"""
DPI-SynPT Full System Debugger (全流程推荐版)
功能：对话 -> 分析 -> 策略 -> 收敛 -> 画像总结 -> 最终推荐
"""

import sys
import io
import os
import time

# 强制设置标准输入输出为 UTF-8
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 确保导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from integrated_bridge import DPI_SynPT_Bridge
from rag_system.rag_interface import run_rag_recommendation

def print_dashboard(turn, result):
    """打印算法监控面板"""
    print("\n" + "="*80)
    print(f"📊  [DPI-ME 监控面板] 第 {turn} 轮对话分析")
    print("-" * 80)
    
    header = f"{'偏好维度':<12} | {'当前推断标签':<12} | {'置信度':<10} | {'熵(Entropy)':<12} | {'状态'}"
    print(header)
    print("-" * 80)
    
    metrics = result['metrics']
    persona = result['persona']
    
    for dim in ["Consumption", "Interest", "Pace"]:
        label = persona[dim]['label']
        conf = persona[dim]['confidence']
        ent = metrics.get(f"{dim}_entropy", 0.0)
        status = "🎯 已收敛" if persona[dim]['is_stable'] else "⏳ 探索中"
        print(f"{dim:<12} | {label:<12} | {conf:<10.2%} | {ent:<12.4f} | {status}")

    print("-" * 80)
    decision = result['decision']
    print(f"🧠 [策略决策] 动作: {decision['action']} | 理由: {decision['reason']}")
    print(f"⏱️  [当前轮次耗时] {result['duration']}s")
    print("="*80 + "\n")


def generate_final_recommendation(bridge, final_result):
    """
    [修正版]：基于收敛的画像和 RAG 引擎生成最终推荐方案
    """
    print("\n" + "✨" * 30)
    print("🎯 [阶段：画像识别与方案生成]")
    print("✨" * 30)

    # 1. 提取最终画像信息
    persona = final_result['persona']
    summary = []
    for dim, info in persona.items():
        summary.append(f"- {dim}: {info['label']} (置信度: {info['confidence']:.2%})")
    
    persona_str = "\n".join(summary)
    print(f"\n👤 识别到的最终用户画像：\n{persona_str}\n")

    # 2. 调用 RAG 引擎获取知识图谱中的具体景点和餐厅
    print("🚀 正在从知识图谱中检索深度定制化方案...")
    
    # 【核心调用】获取你之前写好的 GML 图谱检索结果
    kg_recommendations = run_rag_recommendation()
    print(kg_recommendations)

    # 3. 如果你还想让 Qwen 模型说一段总结性的导游词（可选）
    print("\n智能导游正在为您规划行程说明...")
    final_prompt = "用户偏好已确认，请根据以上画像生成一段简短的欢迎词和行程寄语。"
    # 【修复点】去掉 turn=999，确保不报错
    final_response = bridge.chat_step(final_prompt, 999)
    
    print("\n" + "🏆" * 10 + " 导游寄语 " + "🏆" * 10)
    print(final_response['response'])
    print("🏆" * 32 + "\n")



def main():
    try:
        bridge = DPI_SynPT_Bridge()
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return

    print("\n🌟 [Qwen-PT + DPI-ME] 旅游定制助手已就绪！")
    print("您好！我是您的智能旅游助手，请问您最近有去哪里旅行的计划吗？")

    turn = 1
    try:
        while True:
            user_input = input("\n用户: ").strip()
            if not user_input: continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                bridge.finalize()
                break

            print("\n系统思考中...")
            # 1. 先执行当前轮次的推理和对话回复
            result = bridge.chat_step(user_input, turn)

            # 2. 【改进】立刻打印当前轮次的监控面板和助手回复
            print_dashboard(turn, result)
            print(f"助手: {result['response']}")

            # 3. 检测收敛性，如果动作是 TERMINATE，则进入最终推荐流程
            if result['decision']['action'] == 'TERMINATE':
                print("\n" + "!"*30)
                print("📢 [系统提示] 用户偏好已收敛。系统正在生成最终知识图谱方案...")
                print("!"*30)
                
                # 保存实验日志
                bridge.finalize()
                
                # 4. 执行 RAG 推荐（内部已包含 Hard/Soft 信息打印）
                start_time = time.time()
                recommendation_report = run_rag_recommendation()
                print(recommendation_report)
                end_time = time.time()
                
                print(f"✅ 任务完成！推荐生成耗时: {end_time - start_time:.2f}s")
                
                # 推荐完后直接结束或询问是否继续
                print("\n(对话已锁定，如需开始新的一轮规划请重启程序)")
                break

            turn += 1

    except KeyboardInterrupt:
        bridge.finalize()
    except Exception as e:
        print(f"⚠️ 运行中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()