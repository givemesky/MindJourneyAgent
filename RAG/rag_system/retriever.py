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
                        # if "Budget" in c:
                        #     nums = re.findall(r'\d{3,}', c)
                        #     if nums: self.hard_filters['budget'] = str(max([int(n) for n in nums]))

                        # 触发词库：Budget, Cost, Total, Price, 预算, 花费
                        if any(kw in c for kw in ["Budget", "Cost", "Total", "Price", "预算", "花费"]):
                            # 提取约束字符串中所有的数字
                            nums = re.findall(r'\d+', c)
                            if nums:
                                # 转换为整数列表
                                val_list = [int(n) for n in nums]
                                # 核心过滤：旅游预算通常是一个较大的数
                                # 过滤掉 1 (人数), 3 (天数), 5 (ID) 等干扰项
                                # 假设有效预算通常 >= 200 元
                                money_candidates = [v for v in val_list if v >= 200]
                                
                                if money_candidates:
                                    # 如果有多个大数字（比如人均和总额），取最大值作为硬约束
                                    self.hard_filters['budget'] = str(max(money_candidates))

                        # 3. 提取景点 (Position 或 Attraction Type)
                        # 如果约束里提到 Type == "园林"，我们也把它当做必去关键词
                        if "Position ==" in c or ("Attraction" in c and "Type ==" in c):
                            pos_res = re.findall(r'\"([^\"]+)\"', c)
                            for pos in pos_res:
                                # clean = re.sub(r'与|相关区域|这些地方|这些|附近|去看|看看|逛|一下|玩|看', '', pos)
                                clean = re.sub(r'与|相关区域|这些地方|这些|附近|去看|看看|逛|一下|玩|看|景区|基地', '', pos)
                                if clean and clean not in self.hard_filters['must_visit']:
                                    if len(clean) >= 2: # 至少两个字，防止提取出无意义的单字
                                        self.hard_filters['must_visit'].append(clean)
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

    # def get_recommendations(self):
    #     must_attrs, others, final_res = [], [], []
    #     if self.kg is None: return must_attrs, others, final_res

    #     # 对必去点进行桶排序，确保每个关键词至少有一个
    #     kw_buckets = {kw: [] for kw in self.hard_filters['must_visit']}
    #     target_int = self.soft_priors['interest']

    #     for node, data in self.kg.nodes(data=True):
    #         if 'attr_id' in data: # 景点
    #             matched = False
    #             for kw in self.hard_filters['must_visit']:
    #                 if kw in node:
    #                     kw_buckets[kw].append(node)
    #                     matched = True
    #                     break
    #             if not matched:
    #                 # 匹配软偏好 (例如 Nature -> 自然风光/公园)
    #                 ntype = data.get('type', '')
    #                 if (target_int == "Nature" and ntype in ["自然风光", "公园", "动物园"]) or \
    #                    (target_int == "Culture" and ntype in ["历史古迹", "博物馆/纪念馆"]) or \
    #                    (target_int == "Shopping" and ntype in ["商业街区"]):
    #                     others.append(node)

    #     # 每个关键词取前 2 个，确保“大熊猫”和“太古里”都有
    #     for kw in self.hard_filters['must_visit']:
    #         matches = kw_buckets[kw]
    #         matches.sort(key=len)
    #         must_attrs.extend(matches[:2])
    #         if len(matches) > 2: others.extend(matches[2:])
        
    #     # 关联餐厅检索
    #     raw_res_list = []
    #     for attr in (must_attrs + others[:10]):
    #         if attr in self.kg:
    #             for n in self.kg.neighbors(attr):
    #                 if self.kg.get_edge_data(attr, n).get('relation') == 'attr_dining':
    #                     raw_res_list.append(n)

    #     # 餐厅重排 (关键词匹配优先)
    #     food_kws = self.hard_filters['food']
    #     f_matched, f_others, seen = [], [], set()
    #     for r in raw_res_list:
    #         if r in seen: continue
    #         seen.add(r)
    #         r_data = self.kg.nodes[r]
    #         content = str(r_data.get('recommend', '')) + r
    #         if any(fk in content for fk in food_kws):
    #             f_matched.append(r)
    #         else:
    #             f_others.append(r)
        
    #     return must_attrs, others, (f_matched + f_others)

    # def get_recommendations(self):
    #     must_attrs, interest_pool, final_res = [], [], []
    #     if self.kg is None: return must_attrs, interest_pool, final_res

    #     must_visit_keywords = self.hard_filters['must_visit']
    #     target_int = self.soft_priors['interest']

    #     # 为每个关键词准备一个桶
    #     kw_buckets = {kw: [] for kw in must_visit_keywords}
        
    #     # 拓宽类型匹配范围
    #     nature_types = ["自然风光", "公园", "动物园", "风景名胜", "水利工程", "文化旅游区"]
    #     culture_types = ["历史古迹", "博物馆/纪念馆", "文化旅游区", "人文景观", "寺庙", "古镇"]

    #     for node, data in self.kg.nodes(data=True):
    #         if 'attr_id' in data: 
    #             node_name = str(node)
    #             node_type = data.get('type', '')
                
    #             # A. 灵活匹配必去点
    #             is_must = False
    #             for kw in must_visit_keywords:
    #                 # 只要核心关键词在名字里，或者名字在关键词里（处理“太古里” vs “太古里店”）
    #                 if kw in node_name or node_name in kw:
    #                     kw_buckets[kw].append(node)
    #                     is_must = True
    #                     break
                
    #             # B. 如果不是必选，则按兴趣匹配进入推荐池
    #             if not is_must:
    #                 if target_int == "Nature" and any(t in node_type for t in nature_types):
    #                     interest_pool.append(node)
    #                 elif target_int == "Culture" and any(t in node_type for t in culture_types):
    #                     interest_pool.append(node)
    #                 elif target_int == "Shopping" and node_type == "商业街区":
    #                     interest_pool.append(node)

    #     # 3. 核心改进：配额制筛选必去景点
    #     for kw in must_visit_keywords:
    #         matches = kw_buckets[kw]
    #         if matches:
    #             # 名字短的通常是主景区（如“都江堰景区”比“都江堰南门”短），排前面
    #             matches.sort(key=len)
    #             must_attrs.append(matches[0]) # 取最匹配的一个作为“核心打卡”
    #             # 如果还有其他匹配项，塞进普通推荐池，防止浪费
    #             if len(matches) > 1:
    #                 interest_pool = matches[1:3] + interest_pool 

    #     # 4. 餐厅关联检索 (保持原逻辑，但使用合并后的景点列表)
    #     raw_res_list = []
    #     # 搜索范围：必去点 + 前10个推荐点
    #     search_nodes = must_attrs + interest_pool[:10]
    #     for attr in search_nodes:
    #         if attr in self.kg:
    #             for n in self.kg.neighbors(attr):
    #                 if self.kg.get_edge_data(attr, n).get('relation') == 'attr_dining':
    #                     raw_res_list.append(n)

    #     # 餐厅重排 (关键词匹配优先)
    #     food_kws = self.hard_filters['food']
    #     f_matched, f_others, seen = [], [], set()
    #     for r in raw_res_list:
    #         if r in seen: continue
    #         seen.add(r)
    #         r_data = self.kg.nodes[r]
    #         content = str(r_data.get('recommend', '')) + r + str(r_data.get('cuisine', ''))
    #         if any(fk in content for fk in food_kws):
    #             f_matched.append(r)
    #         else:
    #             f_others.append(r)
        
    #     return must_attrs, interest_pool, (f_matched + f_others)

    # def get_recommendations(self):
    #     """
    #     核心推荐引擎（增强版）：具备菜品驱动的动态范围扩展功能
    #     """
    #     must_attrs, interest_pool, final_res = [], [], []
    #     if self.kg is None: return must_attrs, interest_pool, final_res

    #     # --- 1. 景点识别与匹配逻辑 (保持不变) ---
    #     must_visit_keywords = self.hard_filters['must_visit']
    #     target_int = self.soft_priors['interest']
    #     kw_buckets = {kw: [] for kw in must_visit_keywords}
        
    #     nature_types = ["自然风光", "公园", "动物园", "风景名胜", "水利工程", "文化旅游区"]
    #     culture_types = ["历史古迹", "博物馆/纪念馆", "文化旅游区", "人文景观", "寺庙", "古镇"]

    #     for node, data in self.kg.nodes(data=True):
    #         if 'attr_id' in data: 
    #             node_name = str(node)
    #             node_type = data.get('type', '')
    #             is_must = False
    #             for kw in must_visit_keywords:
    #                 if kw in node_name or node_name in kw:
    #                     kw_buckets[kw].append(node)
    #                     is_must = True
    #                     break
    #             if not is_must:
    #                 if target_int == "Nature" and any(t in node_type for t in nature_types):
    #                     interest_pool.append(node)
    #                 elif target_int == "Culture" and any(t in node_type for t in culture_types):
    #                     interest_pool.append(node)
    #                 elif target_int == "Shopping" and node_type == "商业街区":
    #                     interest_pool.append(node)

    #     for kw in must_visit_keywords:
    #         matches = kw_buckets[kw]
    #         if matches:
    #             matches.sort(key=len)
    #             must_attrs.append(matches[0]) 
    #             if len(matches) > 1:
    #                 interest_pool = matches[1:3] + interest_pool 

    #     # --- 2. 餐厅检索逻辑 (动态范围扩展核心) ---
    #     food_kws = self.hard_filters.get('food', [])
    #     local_res_pool = set() # 存放 1km 内的餐厅
        
    #     # A. 首先获取 1km 内的所有“局部餐厅”
    #     search_nodes = must_attrs + interest_pool[:10]
    #     for attr in search_nodes:
    #         if attr in self.kg:
    #             for n in self.kg.neighbors(attr):
    #                 if self.kg.get_edge_data(attr, n).get('relation') == 'attr_dining':
    #                     local_res_pool.add(n)

    #     # B. 检查局部餐厅对“想吃的菜”的覆盖情况
    #     local_food_matches = []
    #     other_local_res = []
        
    #     for r in local_res_pool:
    #         r_data = self.kg.nodes[r]
    #         # 搜索餐厅名、菜系和推荐菜
    #         content = str(r_data.get('recommend', '')) + str(r) + str(r_data.get('cuisine', ''))
    #         if food_kws and any(fk in content for fk in food_kws):
    #             local_food_matches.append(r)
    #         else:
    #             other_local_res.append(r)

    #     # C. 判断是否需要“扩大范围”进行全局搜索
    #     # 触发条件：用户指定了菜品，但在 1km 范围内没有任何餐厅命中该菜品
    #     if food_kws and not local_food_matches:
    #         print(f"DEBUG: 1km内未找到{food_kws}，正在全城扩大范围搜索...")
    #         global_food_matches = []
            
    #         # 遍历图谱中所有的餐厅节点
    #         for node, data in self.kg.nodes(data=True):
    #             if 'res_id' in data: # 识别餐厅
    #                 content = str(data.get('recommend', '')) + str(node) + str(data.get('cuisine', ''))
    #                 if any(fk in content for fk in food_kws):
    #                     global_food_matches.append(node)
            
    #         # 此时的返回结果：全城命中的餐厅 + 原本周边的普通餐厅
    #         # 排序：匹配菜品的全局餐厅排最前(🔥)
    #         final_res = global_food_matches + other_local_res
    #     else:
    #         # 如果局部已经有匹配的店，或者用户根本没搜菜品，则维持 1km 范围
    #         # 排序：匹配菜品的局部餐厅排最前(🔥)，然后是其他局部餐厅
    #         final_res = local_food_matches + other_local_res

    #     return must_attrs, interest_pool, final_res

    def get_recommendations(self):
        """
        核心推荐引擎（深度优化版）：引入分值评估系统与多维语义匹配
        """
        must_attrs, interest_pool, final_res = [], [], []
        if self.kg is None: return must_attrs, interest_pool, final_res

        must_visit_keywords = self.hard_filters.get('must_visit', [])
        target_int = self.soft_priors.get('interest', '')
        
        # 1. 初始化每个关键词的“潜力桶” (存储：(node_name, score))
        kw_buckets = {kw: [] for kw in must_visit_keywords}
        
        # 2. 拓宽分类映射表
        category_map = {
            "Nature": ["自然风光", "公园", "动物园", "风景名胜", "水利工程", "园林", "文化旅游区"],
            "Culture": ["历史古迹", "博物馆/纪念馆", "文化旅游区", "人文景观", "寺庙", "古镇", "大学校园"],
            "Shopping": ["商业街区", "步行街"],
            "Entertainment": ["游乐园/体育娱乐", "美术馆/艺术馆", "图书馆"]
        }
        target_types = category_map.get(target_int, [])

        # 3. 遍历图谱进行打分匹配
        for node, data in self.kg.nodes(data=True):
            if 'attr_id' not in data: continue # 跳过非景点节点
            
            node_name = str(node)
            node_type = data.get('type', '')
            is_matched_as_must = False
            
            # --- A. 硬约束关键词评分逻辑 ---
            for kw in must_visit_keywords:
                # 过滤掉干扰词（如模型生成的“其它”、“附近”）
                if kw in ["其它", "其他", "附近"]: continue
                
                score = -1
                # 匹配策略与评分 (分值越小越优先)
                if kw == node_name:
                    score = 0   # 完美匹配
                elif node_name.startswith(kw):
                    score = 10  # 前缀匹配 (如: 太古里 -> 太古里广场)
                elif kw in node_name:
                    score = 20  # 包含匹配 (如: 大熊猫 -> 大熊猫繁育基地)
                elif kw in node_type:
                    score = 30  # 类型匹配 (如: 园林 -> 拙政园[类型:园林])

                if score != -1:
                    # 惩罚项：名字过长通常是附属设施（如“剧院”、“店”、“停车场”）
                    if any(x in node_name for x in ["店", "剧院", "停车场", "售票处"]):
                        score += 50
                    score += len(node_name) * 0.1 # 名字越短权重略高
                    
                    kw_buckets[kw].append((node, score))
                    is_matched_as_must = True

            # --- B. 软偏好画像匹配 ---
            if not is_matched_as_must:
                if any(t in node_type for t in target_types):
                    interest_pool.append(node)

        # 4. 配额筛选：每个关键词强制选出最优质的结果
        for kw, matches in kw_buckets.items():
            if matches:
                # 按得分升序排序
                matches.sort(key=lambda x: x[1])
                # 取第一名作为必打卡点
                best_node = matches[0][0]
                if best_node not in must_attrs:
                    must_attrs.append(best_node)
                # 将该桶的其他优质匹配（前3名）放入推荐池前列
                for extra_node, _ in matches[1:3]:
                    if extra_node not in interest_pool:
                        interest_pool.insert(0, extra_node)

        # 5. 餐厅检索逻辑 (保持你原有的动态范围扩展逻辑)
        food_kws = self.hard_filters.get('food', [])
        local_res_pool = set()
        
        # 选取必打卡 + 前8个推荐景点作为餐饮搜索锚点
        search_nodes = must_attrs + interest_pool[:8]
        for attr in search_nodes:
            if attr in self.kg:
                for n in self.kg.neighbors(attr):
                    edge = self.kg.get_edge_data(attr, n)
                    if edge and edge.get('relation') == 'attr_dining':
                        local_res_pool.add(n)

        local_food_matches, other_local_res = [], []
        for r in local_res_pool:
            r_data = self.kg.nodes[r]
            content = str(r_data.get('recommend', '')) + str(r) + str(r_data.get('cuisine', ''))
            if food_kws and any(fk in content for fk in food_kws):
                local_food_matches.append(r)
            else:
                other_local_res.append(r)

        # 触发全局扩展
        if food_kws and not local_food_matches:
            global_food_matches = []
            for node, data in self.kg.nodes(data=True):
                if 'res_id' in data:
                    content = str(data.get('recommend', '')) + str(node) + str(data.get('cuisine', ''))
                    if any(fk in content for fk in food_kws):
                        global_food_matches.append(node)
            final_res = global_food_matches + other_local_res
        else:
            final_res = local_food_matches + other_local_res

        return must_attrs, interest_pool, final_res

