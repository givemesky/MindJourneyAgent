import json
import pandas as pd
import networkx as nx
import os
import re
from .config import *

class TravelRetriever:
    def __init__(self):
        self.hard_filters = {'city': '', 'must_visit': [], 'budget': '未知', 'food': []}
        self.soft_priors = {'interest': '', 'consumption': '', 'pace': ''}
        self.kg = None

    def load_logs(self):
        try:
            with open(CONSTRAINTS_PATH, 'r', encoding='utf-8') as f:
                logs = json.load(f)
                for turn in logs:
                    for c in turn['constraints']:
                        # 1. 提取城市
                        if "Target City ==" in c:
                            res = re.findall(r'\"([^\"]+)\"', c)
                            if res: self.hard_filters['city'] = res[0]
                        
                        # 2. 提取预算
                        if "Budget" in c:
                            nums = re.findall(r'\d+', c)
                            if nums:
                                b_val = max([int(n) for n in nums])
                                if b_val > 100: self.hard_filters['budget'] = str(b_val)

                        # 3. 提取必去点 (只抓取 Position 后面具体的名称)
                        if "Position ==" in c:
                            # 匹配 Position == "..." 中的内容
                            pos_res = re.findall(r'Position == \"([^\"]+)\"', c)
                            for pos in pos_res:
                                # 清洗动词和无意义词
                                clean = re.sub(r'与|相关区域|这些地方|这些|附近|去看|看看|逛|一下|玩|看', '', pos)
                                if clean and clean not in self.hard_filters['must_visit']:
                                    # 排除掉 "公园"、"历史古迹" 等类型词作为景点名
                                    if clean not in ["公园", "历史古迹", "商业街区", "博物馆/纪念馆"]:
                                        self.hard_filters['must_visit'].append(clean)
                        
                        # 4. 提取菜品 (重点修复)
                        if any(kw in c for kw in ["Cuisine", "Restaurant", "Type"]):
                            food_res = re.findall(r'\"([^\"]+)\"', c)
                            for f_item in food_res:
                                # 排除掉城市名和已知的类型词
                                if f_item not in ["南京", "成都", "江苏菜", "公园"] and f_item not in self.hard_filters['food']:
                                    self.hard_filters['food'].append(f_item)
        except Exception as e:
            print(f"解析日志出错: {e}")

        # 加载软偏好
        try:
            df = pd.read_csv(CONVERGENCE_PATH)
            if not df.empty:
                last = df.iloc[-1]
                self.soft_priors = {
                    "interest": last.get('Interest_top', ''),
                    "consumption": last.get('Consumption_top', ''),
                    "pace": last.get('Pace_top', '')
                }
        except:
            pass

    def load_kg(self):
        city_en = {"北京": "beijing", "成都": "chengdu", "上海": "shanghai", "广州": "guangzhou", "深圳": "shenzhen", "杭州": "hangzhou", "苏州": "suzhou", "南京": "nanjing", "武汉": "wuhan", "重庆": "chongqing"}
        name = self.hard_filters.get('city')
        key = city_en.get(name)
        if not key: return False
        
        path = os.path.join(KG_DIR, f"{key}_unified_kg.gml")
        if os.path.exists(path):
            self.kg = nx.read_gml(path)
            return True
        return False

    def get_recommendations(self):
        """核心检索逻辑，确保永远返回 3 个对象"""
        must_attrs, int_attrs, final_res = [], [], []
        if self.kg is None: return must_attrs, int_attrs, final_res

        kw_buckets = {kw: [] for kw in self.hard_filters['must_visit']}
        others = []
        target_int = self.soft_priors['interest']

        # 遍历节点
        for node, data in self.kg.nodes(data=True):
            if 'attr_id' in data: # 景点
                matched = False
                for kw in self.hard_filters['must_visit']:
                    if kw in node:
                        kw_buckets[kw].append(node)
                        matched = True
                        break
                if not matched:
                    # 匹配软偏好
                    ntype = data.get('type', '')
                    if (target_int == "Nature" and ntype in ["自然风光", "公园", "动物园"]) or \
                       (target_int == "Culture" and ntype in ["历史古迹", "博物馆/纪念馆"]):
                        others.append(node)

        # 必选：每个关键词取前2个
        for kw in self.hard_filters['must_visit']:
            matches = kw_buckets[kw]
            matches.sort(key=len)
            must_attrs.extend(matches[:2])
            if len(matches) > 2: others.extend(matches[2:])
        
        # 餐厅关联
        raw_res_list = []
        for attr in (must_attrs + others[:10]):
            if attr in self.kg:
                for n in self.kg.neighbors(attr):
                    if self.kg.get_edge_data(attr, n).get('relation') == 'attr_dining':
                        raw_res_list.append(n)

        # 餐厅重排 (菜品优先)
        food_kws = self.hard_filters['food']
        f_matched, f_others, seen = [], [], set()
        for r in raw_res_list:
            if r in seen: continue
            seen.add(r)
            r_data = self.kg.nodes[r]
            # 检查推荐菜或名称是否包含目标食物
            content = str(r_data.get('recommend', '')) + r
            if any(fk in content for fk in food_kws):
                f_matched.append(r)
            else:
                f_others.append(r)
        
        final_res = f_matched + f_others
        return must_attrs, others, final_res