# mapper.py核心算法：将 Constraints 映射为 D

import re
from .schema import PREFERENCE_SCHEMA

class EvidenceMapper:
    def __init__(self):
        # 映射规则：关键词 -> (维度, 标签, 似然分数)
        self.rules = {
            "Price <= 200": ("Consumption", "Budget", 0.95),
            "Price <= 500": ("Consumption", "Economy", 0.8),
            "Luxury": ("Consumption", "Luxury", 0.9),
            "人文景观": ("Interest", "Culture", 0.8),
            "自然风光": ("Interest", "Nature", 0.9),
            "博物馆": ("Interest", "Culture", 0.95),
            "walk": ("Pace", "Relaxed", 0.7),
            "metro": ("Pace", "Fast", 0.6),
        }

    def get_likelihood_D(self, constraints):
        # 初始化 D 向量 (全 0.5 表示无证据)
        D = {dim: [0.5] * len(labels) for dim, labels in PREFERENCE_SCHEMA.items()}
        
        for c in constraints:
            for key, (dim, label, score) in self.rules.items():
                if key in c:
                    labels = PREFERENCE_SCHEMA[dim]
                    idx = labels.index(label)
                    D[dim][idx] = max(D[dim][idx], score) # 取最强证据
        return D