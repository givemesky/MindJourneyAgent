# dpi_me/persona.py (用户信念状态)
# 定义一个类，保存每个维度的概率分布。
# 属性：self.beliefs (字典), self.entropy_history (列表)。
# 方法：get_top_labels() (返回当前概率最高的偏好)。

# dpi_me/persona.py
import numpy as np
import json
import os

class UserPersona:
    def __init__(self, schema):
        self.schema = schema
        # 1. 初始化信念分布 (Belief States)：初始为均匀分布
        self.beliefs = {
            dim: np.full(len(labels), 1.0 / len(labels)) 
            for dim, labels in schema.items()
        }
        # 2. 演化历史记录 (用于论文绘图)
        self.history = []  # 存储每一轮的概率分布
        self.metrics_history = [] # 存储每一轮的熵和置信度

    def get_entropy(self, dim):
        """计算特定维度的信息熵 (熵越低，画像越确定)"""
        p = self.beliefs[dim]
        return -np.sum(p * np.log2(p + 1e-9))

    def get_confidence(self, dim):
        """计算置信度 (当前最高概率项的概率值)"""
        return np.max(self.beliefs[dim])

    def record_step(self, turn_id):
        """记录当前状态到历史"""
        step_data = {dim: prob.tolist() for dim, prob in self.beliefs.items()}
        self.history.append({"turn": turn_id, "beliefs": step_data})
        
        metrics = {
            dim: {
                "entropy": self.get_entropy(dim),
                "confidence": self.get_confidence(dim)
            } for dim in self.beliefs
        }
        self.metrics_history.append({"turn": turn_id, "metrics": metrics})

    def get_top_persona(self):
        """获取当前最可能的画像标签总结"""
        summary = {}
        for dim, prob in self.beliefs.items():
            idx = np.argmax(prob)
            summary[dim] = {
                "label": self.schema[dim][idx],
                "confidence": round(float(prob[idx]), 3),
                "is_stable": self.get_entropy(dim) < 0.5 # 熵阈值判定是否收敛
            }
        return summary

    def save_logs(self, path="logs/persona_evolution.json"):
        """保存演化日志"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({"history": self.history, "metrics": self.metrics_history}, f, indent=2)