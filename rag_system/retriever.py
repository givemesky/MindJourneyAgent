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
        # 定义支持的城市列表，防止被“任意城市”覆盖
        self.valid_cities = ["北京", "成都", "上海", "广州", "深圳", "杭州", "苏州", "南京", "武汉", "重庆"]

    # def load_logs(self):
    #     try:
    #         with open(CONSTRAINTS_PATH, 'r', encoding='utf-8') as f:
    #             logs = json.load(f)
    #             for turn in logs:
    #                 # A. 优先从 constraints 提取硬约束
    #                 for c in turn['constraints']:
    #                     # 1. 提取城市 (只有在列表中才更新)
    #                     if "Target City ==" in c:
    #                         res = re.findall(r'\"([^\"]+)\"', c)
    #                         if res and res[0] in self.valid_cities:
    #                             self.hard_filters['city'] = res[0]
                        
    #                     # 2. 提取预算 (识别 4 位及以上的数字)
    #                     if "Budget" in c or "Cost" in c:
    #                         nums = re.findall(r'\d{3,}', c) # 找 3 位及以上的数字
    #                         if nums:
    #                             b_val = max([int(n) for n in nums])
    #                             self.hard_filters['budget'] = str(b_val)

    #                     # 3. 提取景点 (Position)
    #                     if "Position ==" in c:
    #                         pos_res = re.findall(r'Position == \"([^\"]+)\"', c)
    #                         for pos in pos_res:
    #                             clean = re.sub(r'与|相关区域|这些地方|这些|附近|去看|看看|逛|一下|玩|看', '', pos)
    #                             if clean and clean not in self.hard_filters['must_visit']:
    #                                 if len(clean) > 1 and clean not in ["公园", "历史古迹", "商业街区"]:
    #                                     self.hard_filters['must_visit'].append(clean)
                        
    #                     # 4. 提取菜品
    #                     if any(kw in c for kw in ["Cuisine", "Restaurant", "Type"]):
    #                         food_res = re.findall(r'\"([^\"]+)\"', c)
    #                         for f_item in food_res:
    #                             if f_item not in self.valid_cities and f_item not in ["公园", "历史古迹"]:
    #                                 if f_item not in self.hard_filters['food']:
    #                                     self.hard_filters['food'].append(f_item)
                    
    #                 # B. 【新增兜底】直接从 user_text 提取漏掉的关键词 (大熊猫/太古里等)
    #                 user_txt = turn.get('user_text', '')
    #                 for key_spot in ["大熊猫", "太古里", "春熙路", "锦里", "宽窄巷子", "武侯祠"]:
    #                     if key_spot in user_txt and key_spot not in self.hard_filters['must_visit']:
    #                         self.hard_filters['must_visit'].append(key_spot)

    #     except Exception as e:
    #         print(f"解析日志出错: {e}")

    #     # 加载软偏好
    #     try:
    #         df = pd.read_csv(CONVERGENCE_PATH)
    #         if not df.empty:
    #             last = df.iloc[-1]
    #             self.soft_priors = {
    #                 "interest": last.get('Interest_top', 'Culture'),
    #                 "consumption": last.get('Consumption_top', 'Economy'),
    #                 "pace": last.get('Pace_top', 'Fast')
    #             }
    #     except:
    #         pass

    def load_logs(self):
        try:
            with open(CONSTRAINTS_PATH, 'r', encoding='utf-8') as f:
                logs = json.load(f)
                for turn in logs:
                    user_txt = turn.get('user_text', '')
                    for c in turn['constraints']:
                        # 1. 提取城市
                        if "Target City ==" in c:
                            res = re.findall(r'\"([^\"]+)\"', c)
                            if res and res[0] in self.valid_cities:
                                self.hard_filters['city'] = res[0]
                        
                        # 2. 提取预算
                        if "Budget" in c:
                            nums = re.findall(r'\d{3,}', c)
                            if nums: self.hard_filters['budget'] = str(max([int(n) for n in nums]))

                        # 3. 提取景点 (Position 或 Attraction Type)
                        # 如果约束里提到 Type == "园林"，我们也把它当做必去关键词
                        if "Position ==" in c or ("Attraction" in c and "Type ==" in c):
                            pos_res = re.findall(r'\"([^\"]+)\"', c)
                            for pos in pos_res:
                                clean = re.sub(r'与|相关区域|这些地方|这些|附近|去看|看看|逛|一下|玩|看', '', pos)
                                # 排除掉“其它”这种无意义词
                                if clean and clean not in ["其它", "其他"]:
                                    if clean not in self.hard_filters['must_visit']:
                                        self.hard_filters['must_visit'].append(clean)
                        
                        # 4. 【精准修复】提取菜品 (排除 Attraction 干扰)
                        # 只有当约束明确提到 Restaurant 或 Cuisine 时才认为是菜
                        if "Restaurant" in c or "Cuisine" in c:
                            food_res = re.findall(r'\"([^\"]+)\"', c)
                            for f_item in food_res:
                                if f_item not in self.valid_cities and f_item not in ["园林", "公园", "其它"]:
                                    if f_item not in self.hard_filters['food']:
                                        self.hard_filters['food'].append(f_item)
                    
                    # 兜底：从文本提取苏州特有的“园林”关键词
                    if "园林" in user_txt and "园林" not in self.hard_filters['must_visit']:
                        self.hard_filters['must_visit'].append("园林")

        except Exception as e:
            print(f"解析日志出错: {e}")

    def load_kg(self):
        city_en = {"北京": "beijing", "成都": "chengdu", "上海": "shanghai", "广州": "guangzhou", "深圳": "shenzhen", "杭州": "hangzhou", "苏州": "suzhou", "南京": "nanjing", "武汉": "wuhan", "重庆": "chongqing"}
        name = self.hard_filters.get('city')
        key = city_en.get(name)
        if not key:
            print(f"DEBUG: 当前识别城市为 '{name}'，不在映射表中。")
            return False
        
        path = os.path.join(KG_DIR, f"{key}_unified_kg.gml")
        if os.path.exists(path):
            self.kg = nx.read_gml(path)
            return True
        return False

    def get_recommendations(self):
        must_attrs, others, final_res = [], [], []
        if self.kg is None: return must_attrs, others, final_res

        # 对必去点进行桶排序，确保每个关键词至少有一个
        kw_buckets = {kw: [] for kw in self.hard_filters['must_visit']}
        target_int = self.soft_priors['interest']

        for node, data in self.kg.nodes(data=True):
            if 'attr_id' in data: # 景点
                matched = False
                for kw in self.hard_filters['must_visit']:
                    if kw in node:
                        kw_buckets[kw].append(node)
                        matched = True
                        break
                if not matched:
                    # 匹配软偏好 (例如 Nature -> 自然风光/公园)
                    ntype = data.get('type', '')
                    if (target_int == "Nature" and ntype in ["自然风光", "公园", "动物园"]) or \
                       (target_int == "Culture" and ntype in ["历史古迹", "博物馆/纪念馆"]) or \
                       (target_int == "Shopping" and ntype in ["商业街区"]):
                        others.append(node)

        # 每个关键词取前 2 个，确保“大熊猫”和“太古里”都有
        for kw in self.hard_filters['must_visit']:
            matches = kw_buckets[kw]
            matches.sort(key=len)
            must_attrs.extend(matches[:2])
            if len(matches) > 2: others.extend(matches[2:])
        
        # 关联餐厅检索
        raw_res_list = []
        for attr in (must_attrs + others[:10]):
            if attr in self.kg:
                for n in self.kg.neighbors(attr):
                    if self.kg.get_edge_data(attr, n).get('relation') == 'attr_dining':
                        raw_res_list.append(n)

        # 餐厅重排 (关键词匹配优先)
        food_kws = self.hard_filters['food']
        f_matched, f_others, seen = [], [], set()
        for r in raw_res_list:
            if r in seen: continue
            seen.add(r)
            r_data = self.kg.nodes[r]
            content = str(r_data.get('recommend', '')) + r
            if any(fk in content for fk in food_kws):
                f_matched.append(r)
            else:
                f_others.append(r)
        
        return must_attrs, others, (f_matched + f_others)