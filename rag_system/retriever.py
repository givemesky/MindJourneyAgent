# 核心检索器：解析日志、加载图谱、执行节点扩展检索

import json
import pandas as pd
import networkx as nx
import os
import re
from config import *

class TravelRetriever:
    def __init__(self):
        self.hard_filters = {'city': '', 'must_visit': [], 'budget': ''}
        self.soft_priors = {}
        self.kg = None

    def load_logs(self):
        # 逻辑保持不变，确保提取关键词
        with open(CONSTRAINTS_PATH, 'r', encoding='utf-8') as f:
            logs = json.load(f)
            for turn in logs:
                for c in turn['constraints']:
                    if "Target City ==" in c:
                        val = c.split('"')[1]
                        if val: self.hard_filters['city'] = val
                    if "Budget per person ==" in c:
                        self.hard_filters['budget'] = c.split('==')[1].strip()
                    if "Position ==" in c:
                        pos = c.split('"')[1]
                        clean_pos = re.sub(r'与|相关区域|这些地方|这些|附近|去看|看看|逛|一下|玩|看', '', pos)
                        if clean_pos and clean_pos not in self.hard_filters['must_visit']:
                            self.hard_filters['must_visit'].append(clean_pos)

        df = pd.read_csv(CONVERGENCE_PATH)
        last_turn = df.iloc[-1]
        self.soft_priors = {
            "interest": last_turn['Interest_top'],
            "consumption": last_turn['Consumption_top'],
            "pace": last_turn['Pace_top']
        }

    def load_kg(self):
        city_en = {"北京": "beijing", "成都": "chengdu", "上海": "shanghai", "广州": "guangzhou", "深圳": "shenzhen", "杭州": "hangzhou", "苏州": "suzhou", "南京": "nanjing", "武汉": "wuhan", "重庆": "chongqing"}
        city_name = self.hard_filters.get('city')
        city_key = city_en.get(city_name)
        if not city_key: return False
        kg_path = os.path.join(KG_DIR, f"{city_key}_unified_kg.gml")
        if os.path.exists(kg_path):
            self.kg = nx.read_gml(kg_path)
            return True
        return False

    def get_recommendations(self):
        # 使用字典来确保每个关键词都有对应的匹配
        must_matches = {kw: [] for kw in self.hard_filters['must_visit']}
        interest_attrs = []
        recommend_res = []

        target_interest = self.soft_priors.get('interest')

        for node, data in self.kg.nodes(data=True):
            if 'attr_id' in data:
                # 1. 检查是否匹配必去关键词
                matched_any_kw = False
                for kw in self.hard_filters['must_visit']:
                    if kw in node:
                        must_matches[kw].append(node)
                        matched_any_kw = True
                
                if matched_any_kw:
                    continue # 已经是必选，跳过兴趣匹配

                # 2. 检查是否匹配软偏好兴趣
                node_type = data.get('type', '')
                if target_interest == "Nature" and node_type in ["自然风光", "公园", "动物园"]:
                    interest_attrs.append(node)
                elif target_interest == "Culture" and node_type in ["历史古迹", "博物馆/纪念馆", "文化旅游区"]:
                    interest_attrs.append(node)
                elif target_interest == "Entertainment" and node_type in ["游乐园/体育娱乐", "人文景观"]:
                    interest_attrs.append(node)
                elif target_interest == "Shopping" and node_type in ["商业街区"]:
                    interest_attrs.append(node)

        # 整理必去景点：每个关键词至少取其最匹配的（名字最短的通常是主景区）
        final_must_attrs = []
        for kw, matches in must_matches.items():
            if matches:
                # 按名称长度排序，比如“太古里”比“太古里川剧院”短，优先排前面
                matches.sort(key=len)
                final_must_attrs.extend(matches)

        # 合并列表
        all_attrs = final_must_attrs + interest_attrs

        # 3. 关联餐厅扩展
        for attr in all_attrs:
            if attr in self.kg:
                for neighbor in self.kg.neighbors(attr):
                    if self.kg.get_edge_data(attr, neighbor).get('relation') == 'attr_dining':
                        recommend_res.append(neighbor)

        seen = set()
        unique_res = [x for x in recommend_res if not (x in seen or seen.add(x))]

        return final_must_attrs, interest_attrs, unique_res