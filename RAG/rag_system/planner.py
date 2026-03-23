# -*- coding: utf-8 -*-
import json
import os
import networkx as nx
import numpy as np
from scipy.cluster.vq import kmeans2
import re
import sys
import torch
import math
import random

# ================= 规划器重构版 =================

class TravelPlanner:
    def __init__(self, user_data, model=None, tokenizer=None):
        """
        初始化规划器
        :param user_data: 可以是 JSON 文件路径，也可以是预先构造好的 dict 对象
        :param model: 注入的共享模型实例
        :param tokenizer: 注入的共享分词器实例
        """
        # 1. 兼容性数据加载
        if isinstance(user_data, str):
            with open(user_data, 'r', encoding='utf-8') as f:
                self.user_data = json.load(f)
        else:
            self.user_data = user_data
        
        # 2. 城市 Key 映射
        city_mapping = {
            "北京": "beijing", "成都": "chengdu", "深圳": "shenzhen", 
            "上海": "shanghai", "苏州": "suzhou", "武汉": "wuhan", 
            "南京": "nanjing", "广州": "guangzhou", "重庆": "chongqing", "杭州": "hangzhou"
        }
        target_city_name = self.user_data['basic_info']['target_city']
        self.city_key = city_mapping.get(target_city_name, "beijing")
        
        # 3. 加载知识图谱 (使用绝对路径确保 API 环境可用)
        self.kg = self._load_kg()
        self.budget_thresholds = {"restaurant": 100, "accommodation": 400}
        
        # 4. 【关键】注入模型资源，不再自行加载
        self.model = model
        self.tokenizer = tokenizer
        if self.model is None or self.tokenizer is None:
            print("⚠️ [Warning] Planner 未注入共享模型，将无法执行最终的文本生成任务。")

    def _load_kg(self):
        # 确保路径指向你图谱存放的绝对位置
        kg_path = f"/public/home/azsy/MindJourney_LoRA/MindJourneyAgent/RAG/kg_travel/city_kg/output_city_kg/{self.city_key}_unified_kg.gml"
        if not os.path.exists(kg_path):
            raise FileNotFoundError(f"找不到图谱文件: {kg_path}")
        return nx.read_gml(kg_path)

    def _clean_name(self, name):
        return re.sub(r'\s*\(.*?\)', '', name).strip()

    def _get_node_full_info(self, name):
        clean_n = self._clean_name(name)
        if clean_n in self.kg:
            data = dict(self.kg.nodes[clean_n])
            data['name'] = clean_n
            raw_recommend = data.get('recommendedfood') or data.get('recommend') or "店长推荐特色菜"
            data['recommend'] = str(raw_recommend) if str(raw_recommend).lower() != 'nan' else "特色苏帮菜"
            data['node_type'] = 'attraction' if 'max_t' in data else ('restaurant' if 'cuisine' in data else 'accommodation')
            try: data['price'] = float(data.get('price', 0))
            except: data['price'] = 0.0
            return data
            
        for node, d in self.kg.nodes(data=True):
            if clean_n in node or node in clean_n:
                res = dict(d)
                res['name'] = node
                raw_recommend = res.get('recommendedfood') or res.get('recommend') or "店长推荐特色菜"
                res['recommend'] = str(raw_recommend) if str(raw_recommend).lower() != 'nan' else "特色苏帮菜"
                res['node_type'] = 'attraction' if 'max_t' in res else ('restaurant' if 'cuisine' in res else 'accommodation')
                try: res['price'] = float(res.get('price', 0))
                except: res['price'] = 0.0
                return res
        return None

    def _find_smart_node(self, ref_node_name, target_type, exclude_names=[], focus="constraint", keyword=None):
        if ref_node_name not in self.kg:
            # 容错：如果参考点不在图谱里，随机选一个
            ref_node_name = random.choice(list(self.kg.nodes))
            
        ref_data = self.kg.nodes[ref_node_name]
        ref_coords = (ref_data['lat'], ref_data['lon'])
        pref = self.user_data['soft_constraints_persona']
        
        candidates = []
        for node, data in self.kg.nodes(data=True):
            if node in exclude_names: continue
            
            n_type = 'attraction' if 'max_t' in data else ('restaurant' if 'cuisine' in data else 'accommodation')
            if n_type == target_type:
                kw_bonus = 0
                if keyword and (keyword in node or node in keyword):
                    kw_bonus = 5000 
                
                price = float(data.get('price', 1.0))
                dist = np.sqrt((data['lat'] - ref_coords[0])**2 + (data['lon'] - ref_coords[1])**2)
                score = dist * 3000 - kw_bonus 

                if focus == "consumption":
                    if "Luxury" in str(pref.get('consumption_level','')): score += (20000 / (price + 1))
                    else: score += (price * 0.5)
                elif focus == "interest":
                    interest_pref = str(pref.get('interest_preference','')).split('(')[0].strip()
                    type_str = str(data.get('type', '')) + str(data.get('cuisine', ''))
                    if interest_pref.lower() in type_str.lower(): score -= 800 
                
                score += price * 0.1
                candidates.append((node, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1])
            return self._get_node_full_info(random.choice(candidates[:min(10, len(candidates))])[0])
        return None

    def solve_plan(self, strategy_name, focus_mode, forbidden_names=set()):
        hards = self.user_data['hard_constraints']
        basic = self.user_data['basic_info']
        day_count, people_count = basic['day_count'], basic['people_count']
        pace = self.user_data['soft_constraints_persona'].get('travel_pace', 'Relaxed')
        is_fast = "Fast" in str(pace)
        
        must_attr_names = hards['must_visit_attractions']
        must_rest_names = hards['must_eat_restaurants']
        used_here = set() | forbidden_names
        
        all_attr_pool = []
        for n in must_attr_names:
            info = self._get_node_full_info(n)
            if info: 
                all_attr_pool.append(info)
                used_here.add(info['name'])

        attr_per_day = 3 if is_fast else 2
        while len(all_attr_pool) < day_count * attr_per_day:
            ref = all_attr_pool[-1]['name'] if all_attr_pool else random.choice(list(self.kg.nodes))
            extra = self._find_smart_node(ref, 'attraction', exclude_names=used_here, focus=focus_mode)
            if extra:
                all_attr_pool.append(extra)
                used_here.add(extra['name'])
            else: break

        # 聚类分配天数
        coords = np.array([[a['lat'], a['lon']] for a in all_attr_pool])
        k_val = min(day_count, len(all_attr_pool))
        centroids, labels = kmeans2(coords, k=k_val, minit='points')
        
        itinerary = []
        for d in range(day_count):
            day_nodes = [all_attr_pool[i] for i, l in enumerate(labels) if l == d % k_val]
            if not day_nodes: continue
            ref_node = day_nodes[0]['name']

            # 餐厅与住宿
            lunch = None
            for mr_name in must_rest_names:
                clean_mr = self._clean_name(mr_name)
                if clean_mr in self.kg and clean_mr not in used_here:
                    lunch = self._get_node_full_info(clean_mr)
                    break
            
            if not lunch:
                lunch = self._find_smart_node(ref_node, 'restaurant', exclude_names=used_here, focus=focus_mode)
            if lunch: used_here.add(lunch['name'])
            
            dinner = self._find_smart_node(day_nodes[-1]['name'], 'restaurant', exclude_names=used_here, focus=focus_mode)
            if dinner: used_here.add(dinner['name'])
            
            hotel = self._find_smart_node(ref_node, 'accommodation', exclude_names=used_here, focus=focus_mode)
            if hotel: used_here.add(hotel['name'])

            day_data = {
                "day": d + 1,
                "morning_attr": day_nodes[0],
                "lunch_res": lunch,
                "afternoon_attr": day_nodes[1] if len(day_nodes) > 1 else self._find_smart_node(ref_node, 'attraction', exclude_names=used_here, focus=focus_mode),
                "dinner_res": dinner,
                "accommodation": hotel
            }
            if is_fast:
                ev = day_nodes[2] if len(day_nodes) > 2 else self._find_smart_node(day_data["afternoon_attr"]['name'] if day_data["afternoon_attr"] else ref_node, 'attraction', exclude_names=used_here, focus=focus_mode)
                day_data["evening_attr"] = ev
                if ev: used_here.add(ev['name'])
            else:
                day_data["evening_attr"] = None

            itinerary.append(day_data)

        # 价格计算
        total = 0
        rooms = math.ceil(people_count / 2)
        for dy in itinerary:
            for k in ["morning_attr", "afternoon_attr", "evening_attr", "lunch_res", "dinner_res"]:
                if dy[k]: total += dy[k]['price'] * people_count
            if dy["accommodation"]: total += dy["accommodation"]['price'] * rooms

        return {"strategy": strategy_name, "itinerary": itinerary, "total": round(total, 2)}

    def run(self):
        """
        核心运行逻辑
        :return: Qwen 生成的文本报告
        """
        b, h, p = self.user_data['basic_info'], self.user_data['hard_constraints'], self.user_data['soft_constraints_persona']
        
        # 1. 运行空间规划算法
        plan_1 = self.solve_plan("硬约束忠实版", "constraint")
        used_p1 = {n['name'] for d in plan_1['itinerary'] for n in [d['morning_attr'], d['afternoon_attr'], d.get('evening_attr'), d['lunch_res'], d['dinner_res'], d['accommodation']] if n}
        plan_2 = self.solve_plan("兴趣偏好深度版", "interest", forbidden_names=used_p1)
        used_p2 = {n['name'] for d in plan_2['itinerary'] for n in [d['morning_attr'], d['afternoon_attr'], d.get('evening_attr'), d['lunch_res'], d['dinner_res'], d['accommodation']] if n}
        plan_3 = self.solve_plan("消费水平定制版", "consumption", forbidden_names=used_p1 | used_p2)

        # 2. 构造推理 Prompt
        prompt = f"""你是一名专业的旅游定制师。请为用户写三套{b['target_city']}旅游方案。
        总人数：{b['people_count']}人。总天数：{b['day_count']}天。用户画像：{json.dumps(p, ensure_ascii=False)}。
        
        ### ⚠️ 核心任务要求：
        1. 必须完整列出每一套方案中包含的所有天数（直到第{b['day_count']}天）。
        2. 严禁只写第一天，严禁使用“同上”、“无”。必须完整列出餐厅的【招牌推荐】。
        3. 当数据中有 evening_attr 且节奏为 Fast 时要列出【晚上浏览】。
        4. 节奏为 Relaxed 时严禁列出【晚上浏览】。
        5. 严禁在同一个方案内多次出现同一家餐厅。
        
        要求格式：
        #### [方案名称]
        第1天：
        - 上午浏览：[景点名] (票价: [price]元/人)。特色：[介绍]。
        - 午餐：[餐厅名] (人均: [price]元)。招牌：[recommend]。
        - 下午浏览：[景点名] (票价: [price]元/人)。特色：[介绍]。
        - 晚上浏览：(仅当数据中有 evening_attr 且节奏为 Fast 时列出)。
        - 晚餐：[餐厅名] (人均: [price]元)。招牌：[recommend]。
        - 住宿：[酒店名] (房价: [price]元/间)。服务：[特色]。
        
        【该方案总预算预估：XXXX 元】

        [数据源-方案1]: {json.dumps(plan_1, ensure_ascii=False)}
        [数据源-方案2]: {json.dumps(plan_2, ensure_ascii=False)}
        [数据源-方案3]: {json.dumps(plan_3, ensure_ascii=False)}
        """

        # 3. 调用注入的模型进行文本渲染
        if self.model and self.tokenizer:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            print(f">>> [Planner] Qwen 正在渲染 {b['day_count']} 天行程报告...")
            with torch.no_grad():
                generated_ids = self.model.generate(**model_inputs, max_new_tokens=4096, temperature=0.1)
                response = self.tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
            return response
        else:
            return "❌ 错误: 模型未成功注入，无法生成行程报告。"
          
          


'''
# -*- coding: utf-8 -*-
import json
import os
import networkx as nx
import numpy as np
from scipy.cluster.vq import kmeans2
import re
import sys
import torch
import math
import random

# ================= 规划器重构版 =================

class TravelPlanner:
    def __init__(self, user_data, model=None, tokenizer=None):
        """
        初始化规划器
        :param user_data: 可以是 JSON 文件路径，也可以是预先构造好的 dict 对象
        :param model: 注入的共享模型实例
        :param tokenizer: 注入的共享分词器实例
        """
        # 1. 兼容性数据加载
        if isinstance(user_data, str):
            with open(user_data, 'r', encoding='utf-8') as f:
                self.user_data = json.load(f)
        else:
            self.user_data = user_data
        
        # 2. 城市 Key 映射
        city_mapping = {
            "北京": "beijing", "成都": "chengdu", "深圳": "shenzhen", 
            "上海": "shanghai", "苏州": "suzhou", "武汉": "wuhan", 
            "南京": "nanjing", "广州": "guangzhou", "重庆": "chongqing", "杭州": "hangzhou"
        }
        target_city_name = self.user_data['basic_info']['target_city']
        self.city_key = city_mapping.get(target_city_name, "beijing")
        
        # 3. 加载知识图谱 (使用绝对路径确保 API 环境可用)
        self.kg = self._load_kg()
        self.budget_thresholds = {"restaurant": 100, "accommodation": 400}
        
        # 4. 【关键】注入模型资源，不再自行加载
        self.model = model
        self.tokenizer = tokenizer
        if self.model is None or self.tokenizer is None:
            print("⚠️ [Warning] Planner 未注入共享模型，将无法执行最终的文本生成任务。")

    def _load_kg(self):
        # 确保路径指向你图谱存放的绝对位置
        kg_path = f"/public/home/azsy/MindJourney_LoRA/MindJourneyAgent/RAG/kg_travel/city_kg/output_city_kg/{self.city_key}_unified_kg.gml"
        if not os.path.exists(kg_path):
            raise FileNotFoundError(f"找不到图谱文件: {kg_path}")
        return nx.read_gml(kg_path)

    def _clean_name(self, name):
        return re.sub(r'\s*\(.*?\)', '', name).strip()

    def _get_node_full_info(self, name):
        clean_n = self._clean_name(name)
        if clean_n in self.kg:
            data = dict(self.kg.nodes[clean_n])
            data['name'] = clean_n
            raw_recommend = data.get('recommendedfood') or data.get('recommend') or "店长推荐特色菜"
            data['recommend'] = str(raw_recommend) if str(raw_recommend).lower() != 'nan' else "特色苏帮菜"
            data['node_type'] = 'attraction' if 'max_t' in data else ('restaurant' if 'cuisine' in data else 'accommodation')
            try: data['price'] = float(data.get('price', 0))
            except: data['price'] = 0.0
            return data
            
        for node, d in self.kg.nodes(data=True):
            if clean_n in node or node in clean_n:
                res = dict(d)
                res['name'] = node
                raw_recommend = res.get('recommendedfood') or res.get('recommend') or "店长推荐特色菜"
                res['recommend'] = str(raw_recommend) if str(raw_recommend).lower() != 'nan' else "特色苏帮菜"
                res['node_type'] = 'attraction' if 'max_t' in res else ('restaurant' if 'cuisine' in res else 'accommodation')
                try: res['price'] = float(res.get('price', 0))
                except: res['price'] = 0.0
                return res
        return None

    def _find_smart_node(self, ref_node_name, target_type, exclude_names=[], focus="constraint", keyword=None):
        if ref_node_name not in self.kg:
            # 容错：如果参考点不在图谱里，随机选一个
            ref_node_name = random.choice(list(self.kg.nodes))
            
        ref_data = self.kg.nodes[ref_node_name]
        ref_coords = (ref_data['lat'], ref_data['lon'])
        pref = self.user_data['soft_constraints_persona']
        
        candidates = []
        for node, data in self.kg.nodes(data=True):
            if node in exclude_names: continue
            
            n_type = 'attraction' if 'max_t' in data else ('restaurant' if 'cuisine' in data else 'accommodation')
            if n_type == target_type:
                kw_bonus = 0
                if keyword and (keyword in node or node in keyword):
                    kw_bonus = 5000 
                
                price = float(data.get('price', 1.0))
                dist = np.sqrt((data['lat'] - ref_coords[0])**2 + (data['lon'] - ref_coords[1])**2)
                score = dist * 3000 - kw_bonus 

                if focus == "consumption":
                    if "Luxury" in str(pref.get('consumption_level','')): score += (20000 / (price + 1))
                    else: score += (price * 0.5)
                elif focus == "interest":
                    interest_pref = str(pref.get('interest_preference','')).split('(')[0].strip()
                    type_str = str(data.get('type', '')) + str(data.get('cuisine', ''))
                    if interest_pref.lower() in type_str.lower(): score -= 800 
                
                score += price * 0.1
                candidates.append((node, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1])
            return self._get_node_full_info(random.choice(candidates[:min(10, len(candidates))])[0])
        return None

    def solve_plan(self, strategy_name, focus_mode, forbidden_names=set()):
        hards = self.user_data['hard_constraints']
        basic = self.user_data['basic_info']
        day_count, people_count = basic['day_count'], basic['people_count']
        pace = self.user_data['soft_constraints_persona'].get('travel_pace', 'Relaxed')
        is_fast = "Fast" in str(pace)
        
        must_attr_names = hards['must_visit_attractions']
        must_rest_names = hards['must_eat_restaurants']
        used_here = set() | forbidden_names
        
        all_attr_pool = []
        for n in must_attr_names:
            info = self._get_node_full_info(n)
            if info: 
                all_attr_pool.append(info)
                used_here.add(info['name'])

        attr_per_day = 3 if is_fast else 2
        while len(all_attr_pool) < day_count * attr_per_day:
            ref = all_attr_pool[-1]['name'] if all_attr_pool else random.choice(list(self.kg.nodes))
            extra = self._find_smart_node(ref, 'attraction', exclude_names=used_here, focus=focus_mode)
            if extra:
                all_attr_pool.append(extra)
                used_here.add(extra['name'])
            else: break

        # 聚类分配天数
        coords = np.array([[a['lat'], a['lon']] for a in all_attr_pool])
        k_val = min(day_count, len(all_attr_pool))
        centroids, labels = kmeans2(coords, k=k_val, minit='points')
        
        itinerary = []
        for d in range(day_count):
            day_nodes = [all_attr_pool[i] for i, l in enumerate(labels) if l == d % k_val]
            if not day_nodes: continue
            ref_node = day_nodes[0]['name']

            # 餐厅与住宿
            lunch = None
            for mr_name in must_rest_names:
                clean_mr = self._clean_name(mr_name)
                if clean_mr in self.kg and clean_mr not in used_here:
                    lunch = self._get_node_full_info(clean_mr)
                    break
            
            if not lunch:
                lunch = self._find_smart_node(ref_node, 'restaurant', exclude_names=used_here, focus=focus_mode)
            if lunch: used_here.add(lunch['name'])
            
            dinner = self._find_smart_node(day_nodes[-1]['name'], 'restaurant', exclude_names=used_here, focus=focus_mode)
            if dinner: used_here.add(dinner['name'])
            
            hotel = self._find_smart_node(ref_node, 'accommodation', exclude_names=used_here, focus=focus_mode)
            if hotel: used_here.add(hotel['name'])

            day_data = {
                "day": d + 1,
                "morning_attr": day_nodes[0],
                "lunch_res": lunch,
                "afternoon_attr": day_nodes[1] if len(day_nodes) > 1 else self._find_smart_node(ref_node, 'attraction', exclude_names=used_here, focus=focus_mode),
                "dinner_res": dinner,
                "accommodation": hotel
            }
            if is_fast:
                ev = day_nodes[2] if len(day_nodes) > 2 else self._find_smart_node(day_data["afternoon_attr"]['name'] if day_data["afternoon_attr"] else ref_node, 'attraction', exclude_names=used_here, focus=focus_mode)
                day_data["evening_attr"] = ev
                if ev: used_here.add(ev['name'])
            else:
                day_data["evening_attr"] = None

            itinerary.append(day_data)

        # 价格计算
        total = 0
        rooms = math.ceil(people_count / 2)
        for dy in itinerary:
            for k in ["morning_attr", "afternoon_attr", "evening_attr", "lunch_res", "dinner_res"]:
                if dy[k]: total += dy[k]['price'] * people_count
            if dy["accommodation"]: total += dy["accommodation"]['price'] * rooms

        return {"strategy": strategy_name, "itinerary": itinerary, "total": round(total, 2)}

    def run(self):
        """
        核心运行逻辑
        :return: Qwen 生成的文本报告
        """
        b, h, p = self.user_data['basic_info'], self.user_data['hard_constraints'], self.user_data['soft_constraints_persona']
        
        # 1. 运行空间规划算法
        plan_1 = self.solve_plan("硬约束忠实版", "constraint")
        used_p1 = {n['name'] for d in plan_1['itinerary'] for n in [d['morning_attr'], d['afternoon_attr'], d.get('evening_attr'), d['lunch_res'], d['dinner_res'], d['accommodation']] if n}
        plan_2 = self.solve_plan("兴趣偏好深度版", "interest", forbidden_names=used_p1)
        used_p2 = {n['name'] for d in plan_2['itinerary'] for n in [d['morning_attr'], d['afternoon_attr'], d.get('evening_attr'), d['lunch_res'], d['dinner_res'], d['accommodation']] if n}
        plan_3 = self.solve_plan("消费水平定制版", "consumption", forbidden_names=used_p1 | used_p2)

        # 2. 构造推理 Prompt
        prompt = f"""你是一名专业的旅游定制师。请为用户写三套{b['target_city']}旅游方案。
        总人数：{b['people_count']}人。总天数：{b['day_count']}天。用户画像：{json.dumps(p, ensure_ascii=False)}。
        
        ### ⚠️ 核心任务要求：
        1. 必须完整列出每一套方案中包含的所有天数（直到第{b['day_count']}天）。
        2. 严禁只写第一天，严禁使用“同上”。必须完整列出餐厅的【招牌推荐】。
        
        要求格式：
        #### [方案名称]
        第1天：
        - 上午：[景点名] (票价: [price]元/人)。[介绍]。
        - 午餐：[餐厅名] (人均: [price]元)。招牌：[recommend]。
        - 下午：[景点名] (票价: [price]元/人)。[介绍]。
        - 晚餐：[餐厅名] (人均: [price]元)。招牌：[recommend]。
        - 住宿：[酒店名] (房价: [price]元/间)。
        
        【该方案总预算预估：XXXX 元】

        [数据源-方案1]: {json.dumps(plan_1, ensure_ascii=False)}
        [数据源-方案2]: {json.dumps(plan_2, ensure_ascii=False)}
        [数据源-方案3]: {json.dumps(plan_3, ensure_ascii=False)}
        """

        # 3. 调用注入的模型进行文本渲染
        if self.model and self.tokenizer:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            print(f">>> [Planner] Qwen 正在渲染 {b['day_count']} 天行程报告...")
            with torch.no_grad():
                generated_ids = self.model.generate(**model_inputs, max_new_tokens=4096, temperature=0.1)
                response = self.tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
            return response
        else:
            return "❌ 错误: 模型未成功注入，无法生成行程报告。"
          
          
'''


'''
import json
import os
import networkx as nx
import numpy as np
from scipy.cluster.vq import kmeans2
import re
import sys
import torch
import math
import random
from transformers import AutoModelForCausalLM, AutoTokenizer




# ================= 配置路径 =================
MODEL_PATH = "/public/home/azsy/MindJourney/MindJourneyAgent/model/Qwen2.5-7B-Instruct"

class TravelPlanner:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.user_data = json.load(f)
        
        city_mapping = {
            "北京": "beijing", "成都": "chengdu", "深圳": "shenzhen", 
            "上海": "shanghai", "苏州": "suzhou", "武汉": "wuhan", 
            "南京": "nanjing", "广州": "guangzhou", "重庆": "chongqing", "杭州": "hangzhou"
        }
        target_city_name = self.user_data['basic_info']['target_city']
        self.city_key = city_mapping.get(target_city_name, "beijing")
        
        self.kg = self._load_kg()
        self.budget_thresholds = {"restaurant": 100, "accommodation": 400}
        
        print(f"正在加载基座模型: {os.path.basename(MODEL_PATH)}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.model.eval()

    def _load_kg(self):
        kg_path = f"/public/home/azsy/MindJourney_LoRA/MindJourneyAgent/RAG/kg_travel/city_kg/output_city_kg/{self.city_key}_unified_kg.gml"
        return nx.read_gml(kg_path)

    def _clean_name(self, name):
        return re.sub(r'\s*\(.*?\)', '', name).strip()

    

    def _get_node_full_info(self, name):
        clean_n = self._clean_name(name)
        if clean_n in self.kg:
            data = dict(self.kg.nodes[clean_n])
            data['name'] = clean_n
            # --- 优化特色菜字段映射 ---
            # 确保将数据库中的 recommendedfood 映射到 Prompt 需要的 recommend 键上
            raw_recommend = data.get('recommendedfood') or data.get('recommend') or "店长推荐特色菜"
            # 处理数据中的 'nan' 或空值
            data['recommend'] = str(raw_recommend) if str(raw_recommend).lower() != 'nan' else "特色苏帮菜"
            
            data['node_type'] = 'attraction' if 'max_t' in data else ('restaurant' if 'cuisine' in data else 'accommodation')
            try: data['price'] = float(data.get('price', 0))
            except: data['price'] = 0.0
            return data
            
        # 模糊匹配
        for node, d in self.kg.nodes(data=True):
            if clean_n in node or node in clean_n:
                res = dict(d)
                res['name'] = node
                # --- 同样的优化处理 ---
                raw_recommend = res.get('recommendedfood') or res.get('recommend') or "店长推荐特色菜"
                res['recommend'] = str(raw_recommend) if str(raw_recommend).lower() != 'nan' else "特色苏帮菜"
                
                res['node_type'] = 'attraction' if 'max_t' in res else ('restaurant' if 'cuisine' in res else 'accommodation')
                try: res['price'] = float(res.get('price', 0))
                except: res['price'] = 0.0
                return res
        return None

    def _find_smart_node(self, ref_node_name, target_type, exclude_names=[], focus="constraint", keyword=None):
        ref_data = self.kg.nodes[ref_node_name]
        ref_coords = (ref_data['lat'], ref_data['lon'])
        pref = self.user_data['soft_constraints_persona']
        
        candidates = []
        for node, data in self.kg.nodes(data=True):
            if node in exclude_names: continue
            
            n_type = 'attraction' if 'max_t' in data else ('restaurant' if 'cuisine' in data else 'accommodation')
            if n_type == target_type:
                kw_bonus = 0
                if keyword and (keyword in node or node in keyword):
                    kw_bonus = 5000 
                
                price = float(data.get('price', 1.0))
                dist = np.sqrt((data['lat'] - ref_coords[0])**2 + (data['lon'] - ref_coords[1])**2)
                score = dist * 3000 - kw_bonus 

                if focus == "consumption":
                    if "Luxury" in pref['consumption_level']: score += (20000 / (price + 1))
                    else: score += (price * 0.5)
                elif focus == "interest":
                    interest_pref = pref['interest_preference'].split('(')[0].strip()
                    type_str = str(data.get('type', '')) + str(data.get('cuisine', ''))
                    if interest_pref.lower() in type_str.lower(): score -= 800 
                
                score += price * 0.1
                candidates.append((node, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1])
            return self._get_node_full_info(random.choice(candidates[:min(10, len(candidates))])[0])
        return None

    def solve_plan(self, strategy_name, focus_mode, forbidden_names=set()):
        hards = self.user_data['hard_constraints']
        basic = self.user_data['basic_info']
        day_count, people_count = basic['day_count'], basic['people_count']
        pace = self.user_data['soft_constraints_persona'].get('travel_pace', 'Relaxed')
        is_fast = "Fast" in pace
        
        must_attr_names = hards['must_visit_attractions']
        must_rest_names = hards['must_eat_restaurants']
        used_here = set() | forbidden_names
        
        all_attr_pool = []
        for n in must_attr_names:
            info = self._get_node_full_info(n)
            if info: 
                all_attr_pool.append(info)
                used_here.add(info['name'])

        attr_per_day = 3 if is_fast else 2
        while len(all_attr_pool) < day_count * attr_per_day:
            ref = all_attr_pool[-1]['name'] if all_attr_pool else random.choice(list(self.kg.nodes))
            extra = self._find_smart_node(ref, 'attraction', exclude_names=used_here, focus=focus_mode)
            if extra:
                all_attr_pool.append(extra)
                used_here.add(extra['name'])
            else: break

        coords = np.array([[a['lat'], a['lon']] for a in all_attr_pool])
        k_val = min(day_count, len(all_attr_pool))
        centroids, labels = kmeans2(coords, k=k_val, minit='points')
        
        itinerary = []
        for d in range(day_count):
            day_nodes = [all_attr_pool[i] for i, l in enumerate(labels) if l == d % k_val]
            if not day_nodes: continue
            ref_node = day_nodes[0]['name']

            lunch = None
            for mr_name in must_rest_names:
                clean_mr = self._clean_name(mr_name)
                if clean_mr in self.kg and clean_mr not in used_here:
                    lunch = self._get_node_full_info(clean_mr)
                    break
                elif clean_mr not in self.kg or clean_mr in used_here:
                    alt_res = self._find_smart_node(ref_node, 'restaurant', exclude_names=used_here, focus=focus_mode, keyword=clean_mr)
                    if alt_res:
                        lunch = alt_res
                        break
            
            if not lunch:
                lunch = self._find_smart_node(ref_node, 'restaurant', exclude_names=used_here, focus=focus_mode)
            if lunch: used_here.add(lunch['name'])
            
            dinner = self._find_smart_node(day_nodes[-1]['name'], 'restaurant', exclude_names=used_here, focus=focus_mode)
            if dinner: used_here.add(dinner['name'])
            
            hotel = self._find_smart_node(ref_node, 'accommodation', exclude_names=used_here, focus=focus_mode)
            if hotel: used_here.add(hotel['name'])

            day_data = {
                "day": d + 1,
                "morning_attr": day_nodes[0],
                "lunch_res": lunch,
                "afternoon_attr": day_nodes[1] if len(day_nodes) > 1 else self._find_smart_node(ref_node, 'attraction', exclude_names=used_here, focus=focus_mode),
                "dinner_res": dinner,
                "accommodation": hotel
            }
            if is_fast:
                ev = day_nodes[2] if len(day_nodes) > 2 else self._find_smart_node(day_data["afternoon_attr"]['name'], 'attraction', exclude_names=used_here, focus=focus_mode)
                day_data["evening_attr"] = ev
                if ev: used_here.add(ev['name'])
            else:
                day_data["evening_attr"] = None

            itinerary.append(day_data)

        total = 0
        rooms = math.ceil(people_count / 2)
        for dy in itinerary:
            for k in ["morning_attr", "afternoon_attr", "evening_attr", "lunch_res", "dinner_res"]:
                if dy[k]: total += dy[k]['price'] * people_count
            if dy["accommodation"]: total += dy["accommodation"]['price'] * rooms

        return {"strategy": strategy_name, "itinerary": itinerary, "total": round(total, 2)}

    

    def run(self):
        b, h, p = self.user_data['basic_info'], self.user_data['hard_constraints'], self.user_data['soft_constraints_persona']
        
        # 链式去重：方案1的所有选点，方案2和3尽量避开
        plan_1 = self.solve_plan("硬约束忠实版", "constraint")
        used_p1 = {n['name'] for d in plan_1['itinerary'] for n in [d['morning_attr'], d['afternoon_attr'], d.get('evening_attr'), d['lunch_res'], d['dinner_res'], d['accommodation']] if n}
        
        plan_2 = self.solve_plan("兴趣偏好深度版", "interest", forbidden_names=used_p1)
        used_p2 = {n['name'] for d in plan_2['itinerary'] for n in [d['morning_attr'], d['afternoon_attr'], d.get('evening_attr'), d['lunch_res'], d['dinner_res'], d['accommodation']] if n}
        
        plan_3 = self.solve_plan("消费水平定制版", "consumption", forbidden_names=used_p1 | used_p2)

        # 【优化重点】：在 Prompt 中强化“全天数输出”的指令
        prompt = f"""你是一名专业的旅游定制师。请为用户写三套{b['target_city']}旅游方案。
        总人数：{b['people_count']}人。总天数：{b['day_count']}天。用户画像：{json.dumps(p, ensure_ascii=False)}。
        
        ### ⚠️ 核心任务要求（非常重要）：
        1. **必须完整列出每一套方案中包含的所有天数（第1天、第2天...直到第{b['day_count']}天）**。
        2. **严禁只写第一天。** 如果数据源中有两天的行程，你必须全部写出来。
        3. 严禁在同一个方案内多次出现同一家餐厅。
        4. 严禁使用“同上”、“见前文”等缩写。必须完整列出餐厅的【招牌推荐】。
        
        要求格式：
        #### [方案名称]
        第1天：
        - 上午浏览：[景点名] (票价: [price]元/人)。特色：[介绍]。
        - 午餐：[餐厅名] (人均: [price]元)。招牌：[recommend]。
        - 下午浏览：[景点名] (票价: [price]元/人)。特色：[介绍]。
        - 晚上浏览：(仅当数据中有 evening_attr 且节奏为 Fast 时列出)。
        - 晚餐：[餐厅名] (人均: [price]元)。招牌：[recommend]。
        - 住宿：[酒店名] (房价: [price]元/间)。服务：[特色]。

        第2天：
        - (必须按照第1天的格式完整列出第2天的行程内容，不得省略)

        [数据源-方案1]: {json.dumps(plan_1, ensure_ascii=False)}
        [数据源-方案2]: {json.dumps(plan_2, ensure_ascii=False)}
        [数据源-方案3]: {json.dumps(plan_3, ensure_ascii=False)}
        """

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        print(f"\n>>> Qwen 模型渲染中（当前任务：{b['day_count']}天行程全量输出）...")
        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=5000, temperature=0.1)
            response = self.tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
        
        print(f"文件将保存至: {os.path.abspath('travel_itinerary_report.txt')}")
        with open("travel_itinerary_report.txt", "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write("           🌟 旅游规划输入背景 (Context) 🌟\n")
            f.write("==================================================\n")
            f.write(f"【基本信息】\n")
            f.write(f"  - 目标城市: {b['target_city']}\n")
            f.write(f"  - 游玩天数: {b['day_count']} 天\n")
            f.write(f"  - 出行人数: {b['people_count']} 人\n\n")

            f.write(f"【硬约束条件 (Hard Constraints)】\n")
            f.write(f"  - 必去景点: {', '.join(h['must_visit_attractions'])}\n")
            f.write(f"  - 必吃餐厅: {', '.join(h['must_eat_restaurants'])}\n")
            f.write(f"  - 最小活动数: {h['min_activities_count']} 个\n\n")

            f.write(f"【用户偏好画像 (Soft Persona)】\n")
            f.write(f"  - 消费水平: {p['consumption_level']}\n")
            f.write(f"  - 兴趣偏好: {p['interest_preference']}\n")
            f.write(f"  - 旅行节奏: {p['travel_pace']}\n\n")

            f.write(f"【方案预估总价摘要 ({b['people_count']}人总计)】\n")
            f.write(f"  - 方案一 ({plan_1['strategy']}): {plan_1['total']} 元\n")
            f.write(f"  - 方案二 ({plan_2['strategy']}): {plan_2['total']} 元\n")
            f.write(f"  - 方案三 ({plan_3['strategy']}): {plan_3['total']} 元\n")
            f.write("==================================================\n\n")
            
            f.write(response)
        
        return f"\n✅ 成功！已强制要求全量天数输出。详见: travel_itinerary_report.txt"

if __name__ == "__main__":
    planner = TravelPlanner("/public/home/azsy/MindJourney_LoRA/MindJourneyAgent/test_json.json")
    print(planner.run())
    
    
'''