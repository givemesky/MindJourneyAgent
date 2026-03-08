# integrated_bridge.py [中枢协调器]：
# 唯一职责：编排业务流。
# 流程：用户输入 -> 调 dpi_interface 分析 -> 调 strategy 决策 -> 调 model_provider 生成回复。

# -*- coding: utf-8 -*-
"""
DPI-SynPT Integrated Bridge (中枢桥接器)
功能：协调全局模型共享、DPI偏好分析、策略决策与助手对话生成。
"""

import os
import sys
import time

# 确保导入路径包含根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入自定义组件
from model_provider import SharedModelProvider
from dpi_me.dpi_interface import DPIAlgorithmInterface
from dpi_me.dpi_reason.strategy import DialogueStrategy

class DPI_SynPT_Bridge:
    def __init__(self):
        print("\n" + "="*50)
        print("🤖 [System] 正在启动 DPI-ME + SynPT 集成系统...")
        print("="*50)

        # 1. 初始化模型提供者 (加载 Qwen + LoRA)
        self.model_provider = SharedModelProvider.get_instance()
        
        # 2. 初始化 DPI 算法接口 (感知/推理/评估)
        self.dpi_engine = DPIAlgorithmInterface()
        
        # 3. 初始化决策策略 (基于熵值的判断)
        self.strategy = DialogueStrategy(max_turns=6)
        
        # 4. 初始化对话历史 (参照 chat_proactive.py)
        self.messages = [
            {"role": "system", "content": "你是一个专业的旅游助手，擅长通过对话挖掘用户意图并提供行程建议。"}
        ]
        
        print("✅ [System] 所有模块初始化完成，显存已优化。")

    def chat_step(self, user_input, turn_id):
        """
        处理一轮对话：分析 -> 决策 -> 生成
        """
        # --- A. DPI-ME 实时偏好分析 ---
        # 运行全链路推理，返回画像总结和数学指标 (Entropy/Confidence)
        persona_summary, metrics = self.dpi_engine.analyze_turn(user_input, turn_id)
        
        # --- B. Strategy 行为决策 ---
        # 根据熵值判断下一步：是继续追问某个维度，还是直接结束并输出方案
        decision = self.strategy.get_decision(metrics, turn_id)
        
        # 获取给 LLM 的引导性提示词
        system_hint = self.strategy.get_prompt_hint(decision, self.dpi_engine.manager.persona)
        # 添加防复读指令
        system_hint += " 注意：如果之前已经问过类似问题，请换一种委婉的方式追问，或者结合已有的信息进行确认。"

        # --- C. 构造对话上下文与助手回复 ---
        # 1. 将用户输入加入历史
        self.messages.append({"role": "user", "content": user_input})
        
        # 2. 注入“决策提示”：这是一个技巧，通过临时插入指令引导模型生成
        # 我们不把这个指令存入长期记忆，只用于当前这一轮的回复生成
        temp_messages = self.messages.copy()
        temp_messages.append({"role": "system", "content": f"注意：{system_hint}"})

        # 3. 调用共享模型生成对话回复 (参照 chat_proactive.py 逻辑)
        start_time = time.time()
        assistant_response = SharedModelProvider.chat_generate(temp_messages)
        duration = time.time() - start_time

        # 4. 将助手的真实回复存入正式历史
        self.messages.append({"role": "assistant", "content": assistant_response})

        # --- D. 封装返回结果 ---
        return {
            "response": assistant_response,
            "metrics": metrics,
            "decision": decision,
            "persona": persona_summary,
            "duration": round(duration, 2)
        }

    def finalize(self):
        """生成最终实验报表"""
        return self.dpi_engine.get_final_report()

# ===================================================
# 快速测试代码
# ===================================================
if __name__ == "__main__":
    bridge = DPI_SynPT_Bridge()
    print("\n🌟 系统就绪！请输入您对旅行的想法。")
    
    turn = 1
    while True:
        user_in = input("\n用户: ")
        if user_in.lower() in ['quit', 'exit', '退出']:
            bridge.finalize()
            break
            
        result = bridge.chat_step(user_in, turn)
        
        print(f"\n[DPI 分析] 确定度: {result['metrics'].get('Interest_confidence', 0):.2%} | 决策: {result['decision']['action']}")
        print(f"助手: {result['response']}")
        
        if result['decision']['action'] == 'TERMINATE':
            print("\n🎯 [系统提示] 检测到偏好已收敛，系统已自动切换至方案输出模式。")
        
        turn += 1