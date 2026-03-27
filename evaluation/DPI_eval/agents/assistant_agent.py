# -*- coding: utf-8 -*-
import sys
import io
import os
import time

# 自动寻找根目录并加入 path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from integrated_bridge import DPI_SynPT_Bridge

# 强制设置标准输入输出为 UTF-8
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class AssistantAgent:
    def __init__(self, log_dir="evaluation/DPI_eval/logs"):
        print("🤖 [Assistant Agent] 正在初始化 DPI-SynPT 引擎...")
        self.log_dir = log_dir
        try:
            self.bridge = DPI_SynPT_Bridge()
            print("✅ [Assistant Agent] 引擎就绪。")
        except Exception as e:
            print(f"❌ [Assistant Agent] 初始化失败: {e}")
            raise


    def process_turn(self, user_input, turn):
        """
        处理一轮对话
        """
        result = self.bridge.chat_step(user_input, turn)
        
        # --- 核心修复点：先定义 decision 变量 ---
        decision = result.get('decision', {}) 
        
        # self._print_mini_dashboard(turn, result)
        
        return {
            "reply": result['response'],
            "status": decision.get('action', 'ASK'), # 安全获取 action
            "persona": result['persona'],
            "metrics": result['metrics'],
            "reason": decision.get('reason', "")    # 安全获取 reason
        }
  
    def terminate_session(self, case_uid):
        """
        将所有 DPI 产生的日志直接保存在 UID 文件夹根目录下
        """
        # 1. 构造该 Case 专属的日志路径
        case_log_path = os.path.join(self.log_dir, case_uid)
        os.makedirs(case_log_path, exist_ok=True)

        print(f"\n💾 [Log] 正在保存 Case {case_uid} 的实验数据至根目录...")

        # 2. 修改 bridge 内部引擎的路径
        if hasattr(self.bridge, 'dpi_engine'):
            engine = self.bridge.dpi_engine
            
            # 更新引擎的目录变量
            engine.user_profile_dir = case_log_path
            engine.convergence_dir = case_log_path
            
            # 【核心修复点】：必须同时更新 evaluator 对象内部的路径！
            if hasattr(engine, 'evaluator'):
                # 强制覆盖评估器的日志目录
                engine.evaluator.log_dir = case_log_path
            
            # 执行 finalize 逻辑：
            # 这内部会调用 engine.evaluator.save_to_csv("persona_convergence.csv")
            self.bridge.finalize()
            print(f"✅ [Assistant Agent] DPI 原始日志（含 CSV）已保存。")
        else:
            print("⚠️ [Warning] 未找到 dpi_engine 引用。")

    def _print_mini_dashboard(self, turn, result):
        persona = result['persona']
        print(f"\n📊 [DPI Monitor] Round {turn}")
        print(f"  - Action: {result['decision']['action']} | Reason: {result['decision']['reason']}")
        for dim in ["Consumption", "Interest", "Pace"]:
            p = persona[dim]
            status = "🎯" if p['is_stable'] else "⏳"
            print(f"  - {dim:<11}: {p['label']:<10} | Conf: {p['confidence']:.2%} {status}")
        print("-" * 40)