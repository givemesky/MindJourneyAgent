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
        """
        初始化旅游定制助手 Agent
        :param log_dir: 实验日志存放的总目录
        """
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
        decision = result['decision']
        action = decision['action']
        
        self._print_mini_dashboard(turn, result)
        
        response_data = {
            "reply": result['response'],
            "status": action,
            "persona": result['persona'],
            "metrics": result['metrics'],
            "reason": decision['reason']
        }

        # 注意：这里不再自动在 Terminate 时调用 terminate_session
        # 改为由 run_eval.py 控制，以便传入 Case ID
        return response_data

    def terminate_session(self, case_uid):
        """
        偏好收敛，保存日志到指定 Case 目录
        :param case_uid: 当前测试用例的唯一 ID
        """
        # 1. 构造该 Case 专属的日志路径
        # 路径示例: evaluation/DPI_eval/logs/p0e20241028.../
        case_log_path = os.path.join(self.log_dir, case_uid)
        user_profile_dir = os.path.join(case_log_path, "UserProfile")
        convergence_dir = os.path.join(case_log_path, "Convergence")

        # 2. 创建文件夹
        os.makedirs(user_profile_dir, exist_ok=True)
        os.makedirs(convergence_dir, exist_ok=True)

        print(f"\n💾 [Log] 正在保存 Case {case_uid} 的实验数据...")

        # 3. 动态修改 bridge 内部引擎的路径并落盘
        if hasattr(self.bridge, 'dpi_engine'):
            # 强制重定向路径
            self.bridge.dpi_engine.user_profile_dir = user_profile_dir
            self.bridge.dpi_engine.convergence_dir = convergence_dir
            # 执行 finalize 逻辑
            self.bridge.finalize()
            print(f"✅ [Assistant Agent] 数据已保存至: {case_log_path}")
        else:
            print("⚠️ [Warning] 未找到 dpi_engine 引用，无法重定向路径。")

    def _print_mini_dashboard(self, turn, result):
        persona = result['persona']
        print(f"\n📊 [DPI Monitor] Round {turn}")
        print(f"  - Action: {result['decision']['action']} | Reason: {result['decision']['reason']}")
        for dim in ["Consumption", "Interest", "Pace"]:
            p = persona[dim]
            status = "🎯" if p['is_stable'] else "⏳"
            print(f"  - {dim:<11}: {p['label']:<10} | Conf: {p['confidence']:.2%} {status}")
        print("-" * 40)