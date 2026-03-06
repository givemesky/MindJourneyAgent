# mapper.py核心算法：将 Constraints 映射为 D

import re
from .schema import PREFERENCE_SCHEMA


class EvidenceMapper:
    def __init__(self):
        # 这里的规则改为正则模式，增加容错性
        self.rules = {
            # 1. 消费维度 (Consumption)
            r"Budget.*<=?\s*200": ("Consumption", "Budget", 0.95),
            r"Price.*<=?\s*200":  ("Consumption", "Budget", 0.95),
            r"Budget":            ("Consumption", "Budget", 0.7),
            r"Luxury":            ("Consumption", "Luxury", 0.9),
            r"五星":              ("Consumption", "Luxury", 0.9),

            # 2. 兴趣维度 (Interest)
            r"外滩":              ("Interest", "Culture", 0.8),
            r"博物馆":            ("Interest", "Culture", 0.9),
            r"美术馆":            ("Interest", "Culture", 0.9),
            r"自然":              ("Interest", "Nature", 0.9),
            r"公园":              ("Interest", "Nature", 0.7),
            r"乐园":              ("Interest", "Entertainment", 0.8),
            r"购物":              ("Interest", "Shopping", 0.8),

            # 3. 节奏维度 (Pace)
            r"慢节奏":            ("Pace", "Relaxed", 0.9),
            r"不.*赶":            ("Pace", "Relaxed", 0.8),
            r"走路":              ("Pace", "Relaxed", 0.7),
            r"步行":              ("Pace", "Relaxed", 0.7),
            r"metro":            ("Pace", "Fast", 0.6),
            r"地铁":              ("Pace", "Fast", 0.6),
        }

    def get_likelihood_D(self, constraints):
        # 初始化 D 向量
        D = {dim: [0.5] * len(labels) for dim, labels in PREFERENCE_SCHEMA.items()}
        
        for c in constraints:
            c_str = str(c).lower()
            for pattern, (dim, label, score) in self.rules.items():
                # 使用正则表达式进行模糊搜索
                if re.search(pattern.lower(), c_str):
                    labels = PREFERENCE_SCHEMA[dim]
                    if label in labels:
                        idx = labels.index(label)
                        # 核心逻辑：累积似然。如果一轮出现多个证据，取最强的
                        D[dim][idx] = max(D[dim][idx], score)
        return D