import json
import os
import networkx as nx
import numpy as np
from scipy.cluster.vq import kmeans2
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 配置路径 =================
MODEL_PATH = "/public/home/azsy/SynPT-main/model/Qwen2.5-7B-Instruct"

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
        
        # 预算阈值设定
        self.budget_thresholds = {
            "restaurant": 100,    
            "accommodation": 400, 
            "attraction": 50      
        }
        
        print(f"正在加载基座模型: {os.path.basename(MODEL_PATH)}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self.model.eval()

    def _load_kg(self):
        kg_path = f"/public/home/azsy/SynPT-main/kg_travel/city_kg/output_city_kg/{self.city_key}_unified_kg.gml"
        return nx.read_gml(kg_path)

    def _clean_name(self, name):
        """清洗输入，处理如 '粤海荟 (粤菜)' -> '粤海荟'"""
        return re.sub(r'\s*\(.*?\)', '', name).strip()

    def _get_node_full_info(self, name):
        """增强版匹配逻辑"""
        clean_n = self._clean_name(name)
        if clean_n in self.kg:
            data = dict(self.kg.nodes[clean_n])
            data['name'] = clean_n
            # 识别类型
            if 'max_t' in data: data['node_type'] = 'attraction'
            elif 'cuisine' in data: data['node_type'] = 'restaurant'
            else: data['node_type'] = 'accommodation'
            # 价格转换
            try: data['price'] = float(data.get('price', 0))
            except: data['price'] = 0.0
            return data
        
        # 模糊搜索
        for node, d in self.kg.nodes(data=True):
            if clean_n in node or node in clean_n:
                res = dict(d)
                res['name'] = node
                if 'max_t' in res: res['node_type'] = 'attraction'
                elif 'cuisine' in res: res['node_type'] = 'restaurant'
                else: res['node_type'] = 'accommodation'
                try: res['price'] = float(res.get('price', 0))
                except: res['price'] = 0.0
                return res
        return None

    def _find_smart_node(self, ref_node_name, target_type, exclude_names=[], strategy="loyalty"):
        """价格-距离平衡搜索"""
        ref_data = self.kg.nodes[ref_node_name]
        ref_coords = (ref_data['lat'], ref_data['lon'])
        max_p = self.budget_thresholds.get(target_type) if strategy == "budget" else None

        candidates = []
        for node, data in self.kg.nodes(data=True):
            if node in exclude_names: continue
            
            # 判断节点类型
            n_type = 'attraction' if 'max_t' in data else ('restaurant' if 'cuisine' in data else 'accommodation')
            if n_type == target_type:
                price = float(data.get('price', 0))
                if strategy == "budget" and max_p and price > max_p: continue
                
                dist = np.sqrt((data['lat'] - ref_coords[0])**2 + (data['lon'] - ref_coords[1])**2)
                score = price * 0.1 + dist * 500  # 综合分：越小越好
                candidates.append((node, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1])
            return self._get_node_full_info(candidates[0][0])
        
        # 全城最便宜兜底
        all_of_type = [n for n, d in self.kg.nodes(data=True) if ('max_t' in d if target_type=='attraction' else ('cuisine' in d if target_type=='restaurant' else 'lat' in d))]
        all_of_type.sort(key=lambda x: float(self.kg.nodes[x].get('price', 9999)))
        return self._get_node_full_info(all_of_type[0])

    def solve_plan(self, strategy="loyalty"):
        hards = self.user_data['hard_constraints']
        day_count = self.user_data['basic_info']['day_count']
        pace = self.user_data['soft_constraints_persona'].get('travel_pace', 'Relaxed')
        
        # 1. 强制提取硬约束
        must_attrs = [self._get_node_full_info(n) for n in hards['must_visit_attractions'] if self._get_node_full_info(n)]
        must_rests = [self._get_node_full_info(n) for n in hards['must_eat_restaurants'] if self._get_node_full_info(n)]

        # 2. 补全景点 (Fast: 3个/天, Relaxed: 2个/天)
        per_day = 3 if "Fast" in pace else 2
        total_needed = day_count * per_day
        all_attr_pool = list(must_attrs)
        used_names = {a['name'] for a in all_attr_pool}

        while len(all_attr_pool) < total_needed:
            ref = all_attr_pool[-1]['name']
            extra = self._find_smart_node(ref, 'attraction', exclude_names=used_names, strategy=strategy)
            if extra:
                all_attr_pool.append(extra)
                used_names.add(extra['name'])
            else: break

        # 3. 地理聚类
        coords = np.array([[a['lat'], a['lon']] for a in all_attr_pool])
        k_val = min(day_count, len(all_attr_pool))
        centroids, labels = kmeans2(coords, k=k_val, minit='points')
        
        itinerary = []
        for d in range(day_count):
            day_nodes = [all_attr_pool[i] for i, l in enumerate(labels) if l == d % k_val]
            if not day_nodes: continue
            ref_node = day_nodes[0]['name']

            # 午餐：忠实版必须用原选
            res_idx = d % len(must_rests) if must_rests else 0
            base_lunch = must_rests[res_idx] if must_rests else None
            
            if strategy == "budget" and base_lunch and base_lunch['price'] > self.budget_thresholds['restaurant']:
                lunch = self._find_smart_node(ref_node, 'restaurant', strategy="budget", exclude_names=used_names)
                change_reason = f"原选{base_lunch['name']}超出预算，已替换"
            else:
                lunch = base_lunch or self._find_smart_node(ref_node, 'restaurant', exclude_names=used_names)
                change_reason = ""

            # 晚餐/住宿/补位景点
            dinner = self._find_smart_node(day_nodes[-1]['name'], 'restaurant', exclude_names=used_names | {lunch['name'] if lunch else ""}, strategy=strategy)
            hotel = self._find_smart_node(ref_node, 'accommodation', strategy=strategy)
            
            # 分段
            morning = day_nodes[0]
            afternoon = day_nodes[1] if len(day_nodes) > 1 else self._find_smart_node(ref_node, 'attraction', exclude_names=used_names, strategy=strategy)
            evening = day_nodes[2] if len(day_nodes) > 2 else self._find_smart_node(afternoon['name'], 'attraction', exclude_names=used_names | {afternoon['name']}, strategy=strategy)

            itinerary.append({
                "day": d + 1,
                "morning": morning, "lunch": lunch,
                "afternoon": afternoon, "evening": evening,
                "dinner": dinner, "accommodation": hotel,
                "note": change_reason
            })
            # 更新已用
            for item in [lunch, dinner, hotel, morning, afternoon, evening]:
                if item: used_names.add(item['name'])

        # 计算总价
        total = 0
        for day in itinerary:
            for k in ["morning", "afternoon", "evening", "lunch", "dinner", "accommodation"]:
                if day[k]: total += day[k]['price']

        return itinerary, round(total, 2)

    def run(self):
        b, h, p = self.user_data['basic_info'], self.user_data['hard_constraints'], self.user_data['soft_constraints_persona']
        print(f"\n>>> [1. 任务解析]")
        print(f"  目标: {b['target_city']} {b['day_count']}天 | 预算偏好: {p['consumption_level']}")
        print(f"  必去景点: {h['must_visit_attractions']}")
        print(f"  必吃餐厅: {h['must_eat_restaurants']}")

        loy_iti, loy_total = self.solve_plan(strategy="loyalty")
        bud_iti, bud_total = self.solve_plan(strategy="budget")

        print("\n>>> [2. 结构化数据预览 (用于调试)]")
        debug_output = {"loyalty_total": loy_total, "budget_total": bud_total, "sample_day": loy_iti[0]}
        print(json.dumps(debug_output, ensure_ascii=False, indent=2))

        prompt = f"""你是一名资深导游。请根据以下 JSON 旅游数据，撰写两套深圳旅游方案。
        要求格式：
        #### 方案X：[名称]
        第N天：
        - 上午：[name] (费用: [price]元)。[type]特色。
        - 午餐：[name] (人均: [price]元)。招牌：[recommend]。
        - 下午：[name] (费用: [price]元)。
        - 晚上：[name] (费用: [price]元)。
        - 晚餐：[name] (人均: [price]元)。招牌：[recommend]。
        - 住宿：[name] (费用: [price]元)。
        
        注意：
        1. 每一个时间段必须有具体地点和价格，禁止出现“暂无”或“无”。
        2. 方案一忠实执行必吃必去。方案二若必吃太贵则进行平价替换并说明。
        3. 总价必须出现在方案末尾。

        [数据源-忠实版]: {json.dumps(loy_iti, ensure_ascii=False)} | 总价: {loy_total}
        [数据源-预算版]: {json.dumps(bud_iti, ensure_ascii=False)} | 总价: {bud_total}
        """

        print("\n>>> [3. Qwen 模型渲染中...]")
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=3000, temperature=0.1)
            response = self.tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
        
        # 存入文件
        report_path = "travel_itinerary_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"用户画像: {json.dumps(p, ensure_ascii=False)}\n")
            f.write(f"硬约束条件: {json.dumps(h, ensure_ascii=False)}\n")
            f.write("\n" + "="*50 + "\n")
            f.write(response)
        
        return f"\n✅ 规划完成！详细方案已保存至: {os.path.abspath(report_path)}"

if __name__ == "__main__":
    planner = TravelPlanner("/public/home/azsy/SynPT-main/test_json.json")
    print(planner.run())