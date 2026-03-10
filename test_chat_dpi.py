# [新建] 最终集成测试：运行“对话-分析-策略-生成”全流程

# -*- coding: utf-8 -*-
"""
DPI-SynPT Full System Debugger (总调试脚本)
功能：启动集成系统，展示“对话-分析-熵减-决策”全链路过程。
"""

import sys
import io

# 强制设置标准输入输出为 UTF-8 编码，防止终端乱码报错
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import time

# 确保导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from integrated_bridge import DPI_SynPT_Bridge

def print_dashboard(turn, result):
    """打印算法监控面板，展示当前偏好推断的内部状态"""
    print("\n" + "="*80)
    print(f"📊  [DPI-ME 监控面板] 第 {turn} 轮对话分析")
    print("-" * 80)
    
    # 打印画像总结与置信度指标
    header = f"{'偏好维度':<12} | {'当前推断标签':<12} | {'置信度':<10} | {'熵(Entropy)':<12} | {'状态'}"
    print(header)
    print("-" * 80)
    
    metrics = result['metrics']
    persona = result['persona']
    
    for dim in ["Consumption", "Interest", "Pace"]:
        label = persona[dim]['label']
        conf = persona[dim]['confidence']
        # 这里的 key 名需与 evaluator.py 记录的保持一致
        ent = metrics.get(f"{dim}_entropy", 0.0)
        
        # 确定度展示
        status = "🎯 已收敛" if persona[dim]['is_stable'] else "⏳ 探索中"
        print(f"{dim:<12} | {label:<12} | {conf:<10.2%} | {ent:<12.4f} | {status}")

    print("-" * 80)
    # 打印策略决策
    decision = result['decision']
    print(f"🧠 [策略决策] 动作: {decision['action']} | 理由: {decision['reason']}")
    print(f"⏱️  [生成耗时] {result['duration']}s")
    print("="*80 + "\n")

def main():
    # 1. 初始化集成中枢 (此处会触发 model_provider 加载模型)
    try:
        bridge = DPI_SynPT_Bridge()
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return

    print("\n🌟 [Qwen-PT + DPI-ME] 集成助手已就绪！")
    print("输入 'quit' 退出系统并生成报表。")

    # 初始引导语 (可选)
    print("\n助手: 您好！我是您的智能旅游助手，请问您最近有去哪里旅行的计划吗？")

    turn = 1
    try:
        while True:
            # 2. 获取用户输入
            user_input = input("\n用户: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("\n🚀 正在结束对话并生成实验数据报告...")
                report = bridge.finalize()
                print("✅ 报告已保存至 dpi_me/logs/ 目录下。")
                break

            # 3. 运行集成步进：分析 -> 决策 -> 回复
            # 这里是模型共享的核心，DPI分析和SynPT回复会顺次执行
            print("\n系统思考中...")
            result = bridge.chat_step(user_input, turn)

            # 4. 展示监控面板 (用于毕设调试和论文数据观察)
            print_dashboard(turn, result)

            # 5. 输出助手回复
            print(f"助手: {result['response']}")

            # 6. 如果检测到对话应结束，给出提示
            # if result['decision']['action'] == 'TERMINATE':
            #     print("\n📢 [系统提示] 用户偏好已收敛。接下来系统将锁定此偏好，为您生成详细方案。")

            # turn += 1

            if result['decision']['action'] == 'TERMINATE':
                print("\n" + "!"*30)
                print("📢 [系统提示] 用户偏好已收敛。接下来系统将锁定此偏好，为您生成详细方案。")
                print("🚀 正在自动保存实验数据并退出系统...")
                print("!"*30)
                
                # 执行保存逻辑
                bridge.finalize()
                
                print(f"\n✅ 任务完成！耗时: {result['duration']}s")
                return # 或者使用 sys.exit(0) 彻底退出程序

            turn += 1

    except KeyboardInterrupt:
        print("\n检测到强制退出，正在保存数据...")
        bridge.finalize()
    except Exception as e:
        print(f"⚠️ 运行中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()