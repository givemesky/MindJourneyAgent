# dpi_me/evaluator.py (收敛性分析)
# 核心指标：计算 信息熵 (Entropy) 的下降速度，记录 置信度 (Confidence) 的上升曲线。
# 输出：保存为 .csv 文件，方便你用 Python 画出论文里的那种偏好演化图。

# dpi_me/engine.py
import numpy as np
import csv
import os

class PreferenceEvaluator:
    def __init__(self, log_dir="dpi_me/logs"):
        """
        初始化评估器
        log_dir: 结果保存路径
        """
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.history = [] # 用于存储每一轮的指标

    @staticmethod
    def calculate_entropy(probs):
        """计算给定概率分布的信息熵 (单位: bits)"""
        probs = np.array(probs)
        # 过滤掉概率为 0 的项，避免 log2(0) 报错
        probs = probs[probs > 1e-12]
        if len(probs) <= 1:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    @staticmethod
    def calculate_certainty(probs):
        """
        计算确定度 (Confidence Score)
        公式: 1 - (当前熵 / 最大可能熵)
        0 表示完全随机，1 表示完全确定
        """
        n_categories = len(probs)
        if n_categories <= 1:
            return 1.0
        
        max_entropy = np.log2(n_categories)
        current_entropy = PreferenceEvaluator.calculate_entropy(probs)
        
        # 防止微小的数值误差导致结果超出 [0, 1]
        certainty = 1.0 - (current_entropy / max_entropy)
        return max(0.0, min(1.0, certainty))

    def record_turn(self, turn_idx, persona):
        """
        记录当前轮次各维度的评估指标
        persona: UserPersona 实例，其中包含 beliefs 字典
        """
        entry = {"turn": turn_idx}
        
        for dim, probs in persona.beliefs.items():
            entropy = self.calculate_entropy(probs)
            certainty = self.calculate_certainty(probs)
            top_label = persona.get_top_label(dim) # 假设 persona 有此方法

            entry[f"{dim}_entropy"] = round(entropy, 4)
            entry[f"{dim}_confidence"] = round(certainty, 4)
            entry[f"{dim}_top"] = top_label
            
        self.history.append(entry)
        return entry

    def save_to_csv(self, filename="convergence_analysis.csv"):
        """将记录的所有历史指标保存为 CSV"""
        if not self.history:
            print("Warning: No history to save.")
            return

        filepath = os.path.join(self.log_dir, filename)
        keys = self.history[0].keys()
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.history)
        
        print(f"Convergence analysis saved to {filepath}")