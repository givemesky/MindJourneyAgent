# dpi_me/dpi_interface.py [DPI 算法入口]：
# [新建] 算法接口层：封装感知、推理与评估，对外提供单次分析 API
# 功能：输入一段话，输出当前的“用户画像”和“信息熵指标”。

# -*- coding: utf-8 -*-
"""
DPI-ME Algorithm Interface Layer
功能：封装感知、推理、状态更新与评估全流程。
      输入用户文本，输出当前画像状态与数学收敛指标。
"""

import os
import sys

# 确保能找到根目录下的 model_provider
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from dpi_me.dpi_reason.manager import DPISystemManager
    from dpi_me.dpi_reason.evaluator import PreferenceEvaluator
except ImportError as e:
    print(f"❌ [DPI Interface] 导入内部模块失败: {e}")
    raise

class DPIAlgorithmInterface:
    # def __init__(self, log_dir="/public/home/azsy/SynPT-main/dpi_me/logs/UserProfile"):
    #     """
    #     初始化 DPI 算法插件
    #     :param log_dir: 实验日志保存路径
    #     """
    #     print("🛠️  [DPI Interface] 正在初始化 DPI-ME 算法引擎...")
        
    #     # 1. 初始化系统中枢 (内部已配置为引用全局共享模型)
    #     # 注意：这里传入 model_name="shared" 是为了告诉管理器不要自己加载模型
    #     self.manager = DPISystemManager(model_name="shared")
        
    #     # 2. 初始化评估器
    #     self.evaluator = PreferenceEvaluator(log_dir=log_dir)
    #     self.log_dir = log_dir 
    #     print("✅ [DPI Interface] 算法引擎初始化完成。")

    def __init__(self, 
                 user_profile_dir="/public/home/azsy/SynPT-main/dpi_me/logs/UserProfile",
                 convergence_dir="/public/home/azsy/SynPT-main/dpi_me/logs/Convergence"): # 新增路径参数
        """
        初始化 DPI 算法插件
        """
        print("🛠️  [DPI Interface] 正在初始化引擎...")
        
        self.manager = DPISystemManager(model_name="shared")
        
        # --- 修改这里：分别为两个组件指定不同的目录 ---
        # 1. 评估器存到 Convergence
        self.evaluator = PreferenceEvaluator(log_dir=convergence_dir)
        print("✅ [DPI Interface] 算法引擎初始化完成。")

        # 2. 接口层保存 UserProfile 的引用
        self.user_profile_dir = user_profile_dir
        self.convergence_dir = convergence_dir
        
        print(f"✅ 日志配置：\n   - 画像轨迹: {user_profile_dir}\n   - 收敛报表: {convergence_dir}")



    def analyze_turn(self, user_input, turn_id):
        """
        核心接口：执行一轮完整的偏好推理过程
        :param user_input: 用户本轮输入的自然语言字符串
        :param turn_id: 当前对话轮次 (从1开始)
        :return: (persona_summary, metrics)
                 - persona_summary: 包含各维度推断标签和置信度的字典
                 - metrics: 包含熵和确定度的数学指标字典
        """
        
        # --- 1. 执行推理链路 (Perception -> Reasoning -> State Update) ---
        # manager.process_turn 内部会调用 nl2sl, EvidenceMapper 和 BayesianUpdater
        self.manager.process_turn(user_input, turn_id)
        # 【关键修改】从 manager 中获取刚刚识别到的约束
        current_constraints = getattr(self.manager, 'last_constraints', [])
        
        # --- 2. 计算并记录评估指标 ---
        # 记录本轮各维度的熵值、确定度，并保存到 evaluator.history
        metrics = self.evaluator.record_turn(turn_id, self.manager.persona)
        
        # --- 3. 获取当前画像的顶层总结 ---
        # 包含每个维度的 Label, Confidence, is_stable
        persona_summary = self.manager.persona.get_top_persona()
        
        # 记录本次演化到本地存储 (json)
        self.manager.persona.record_step(turn_id)
        
        
        return persona_summary, metrics

    # def get_final_report(self):
    #     """保存并获取最终实验数据报表"""
    #     # 1. 保存 CSV
    #     self.evaluator.save_to_csv("persona_convergence.csv")
        
    #     # 2. 保存画像 JSON (纯数学分布)
    #     json_path = os.path.join(self.log_dir, "persona_evolution.json")
    #     self.manager.persona.save_logs(path=json_path)
        
    #     # 3. 保存独立约束 JSON (纯语义约束)
    #     constraints_path = os.path.join(self.log_dir, "constraints_log.json")
    #     self.manager.save_constraints(path=constraints_path)
        
    #     # 这里建议直接 return 而不是调用 manager 的 generate_final_report
    #     # 因为 manager 里的路径是硬编码的，可能会产生重复或路径不一的文件
    #     return self.manager.persona.get_top_persona()

    def get_final_report(self):
        """保存并获取最终实验数据报表"""
        # 1. 保存 CSV (它会存入初始化时指定的 Convergence 目录)
        self.evaluator.save_to_csv("persona_convergence.csv")
        
        # 2. 保存画像 JSON (明确指向 user_profile_dir)
        json_path = os.path.join(self.user_profile_dir, "persona_evolution.json")
        self.manager.persona.save_logs(path=json_path)
        
        # 3. 保存独立约束 JSON (明确指向 user_profile_dir)
        constraints_path = os.path.join(self.user_profile_dir, "constraints_log.json")
        self.manager.save_constraints(path=constraints_path)
        
        # 返回最终画像总结
        return self.manager.persona.get_top_persona()

    def reset(self):
        """重置画像状态 (用于开始新对话)"""
        from dpi_me.dpi_likelihood.reasoning.schema import PREFERENCE_SCHEMA
        from dpi_me.dpi_reason.persona import UserPersona
        self.manager.persona = UserPersona(PREFERENCE_SCHEMA)
        print("🔄 [DPI Interface] 用户画像已重置。")