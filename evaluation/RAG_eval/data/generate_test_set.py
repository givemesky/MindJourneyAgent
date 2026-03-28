# -*- coding: utf-8 -*-
import os
import pandas as pd
import json
import random
from itertools import product

# ================= 配置路径 =================
BASE_DATA_DIR = "/public/home/azsy/MindJourney_LoRA/MindJourneyAgent/RAG/kg_travel"
OUTPUT_PATH = "/public/home/azsy/MindJourney_LoRA/MindJourneyAgent/evaluation/RAG_eval/benchmark_test_set_60.json"

# ================= 维度定义 =================
CITIES = ["北京", "成都", "深圳", "上海", "苏州", "武汉", "南京", "广州", "重庆", "杭州"]
CITY_MAP = {
    "北京": "beijing", "成都": "chengdu", "深圳": "shenzhen", "上海": "shanghai",
    "苏州": "suzhou", "武汉": "wuhan", "南京": "nanjing", "广州": "guangzhou",
    "重庆": "chongqing", "杭州": "hangzhou"
}

CONSUMPTIONS = ["Budget", "Economy", "Luxury"]
INTERESTS = ["Culture", "Nature", "Entertainment", "Shopping"]
PACES = ["Fast", "Relaxed"]

INTEREST_KEYWORDS = {
    "Culture": ["博物馆", "寺", "古", "历史", "遗址", "园林", "建筑", "宫", "塔"],
    "Nature": ["公园", "湖", "山", "风景", "自然", "岛", "湿地", "林", "泉"],
    "Entertainment": ["游乐", "船", "剧", "演艺", "中心", "广场", "动物园", "海洋", "塔"],
    "Shopping": ["步行街", "商场", "广场", "购物", "奥特莱斯", "市集", "里", "街"]
}

class TestSetGenerator:
    def __init__(self):
        self.test_cases = []

    def load_city_data(self, city_zh):
        city_en = CITY_MAP[city_zh]
        attr_path = f"{BASE_DATA_DIR}/attractions/{city_en}/attractions.csv"
        rest_path = f"{BASE_DATA_DIR}/restaurants/{city_en}/restaurants_{city_en}.csv"
        
        try:
            df_attr = pd.read_csv(attr_path)
            df_rest = pd.read_csv(rest_path)
            # 清洗数据：去除名字为空的情况
            df_attr = df_attr.dropna(subset=['name'])
            df_rest = df_rest.dropna(subset=['name'])
            return df_attr, df_rest
        except Exception as e:
            print(f"⚠️ 警告: 无法加载 {city_zh} 的数据: {e}")
            return None, None

    def generate(self):
        # 1. 生成基础画像池 (24种组合)
        base_personas = list(product(CONSUMPTIONS, INTERESTS, PACES))
        
        # 2. 为了凑够 60 个，我们需要将画像池扩充并洗牌
        # 60 / 24 = 2.5 次，所以取 3 次循环后截断
        extended_personas = (base_personas * 3)
        random.shuffle(extended_personas)
        
        # 3. 难度任务队列：每个难度 20 个
        difficulty_task = ["Easy"] * 20 + ["Standard"] * 20 + ["Hard"] * 20
        random.shuffle(difficulty_task)

        # 4. 城市分配：每个城市 6 个 Case
        persona_idx = 0
        diff_idx = 0
        
        for city_zh in CITIES:
            df_attr, df_rest = self.load_city_data(city_zh)
            if df_attr is None: continue

            # 每个城市生成 6 个 Case (2 Easy, 2 Standard, 2 Hard)
            # 为了保证每个城市难度均衡，我们手动指定该城市的难度构成
            city_difficulties = ["Easy", "Easy", "Standard", "Standard", "Hard", "Hard"]
            random.shuffle(city_difficulties)

            for diff in city_difficulties:
                cons, inter, pace = extended_personas[persona_idx]
                
                case = self.create_case(city_zh, df_attr, df_rest, cons, inter, pace, diff)
                if case:
                    self.test_cases.append(case)
                
                persona_idx += 1
                diff_idx += 1

        # 5. 保存
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.test_cases, f, ensure_ascii=False, indent=4)
        
        # 打印统计信息
        print(f"\n" + "="*40)
        print(f"✅ 测试集构建成功！总计: {len(self.test_cases)} 个用例")
        print(f"📊 难度分布: Easy: 20 | Standard: 20 | Hard: 20")
        print(f"📂 城市分布: 10 城市 x 6 用例/城市")
        print(f"📍 路径: {OUTPUT_PATH}")
        print("="*40)

    def create_case(self, city, df_attr, df_rest, consumption, interest, pace, difficulty):
        # 参数配置
        if difficulty == "Easy":
            day_count = random.randint(1, 2)
            attr_num = random.randint(1, 2)
            rest_num = 1
        elif difficulty == "Standard":
            day_count = random.randint(3, 4)
            attr_num = random.randint(3, 4)
            rest_num = 2
        else: # Hard
            day_count = random.randint(5, 7)
            attr_num = random.randint(5, 7)
            rest_num = 3

        # 抽样景点
        keywords = INTEREST_KEYWORDS.get(interest, [])
        pattern = '|'.join(keywords)
        mask = df_attr['name'].str.contains(pattern, na=False) | df_attr['type'].str.contains(pattern, na=False)
        matched_attrs = df_attr[mask]
        
        if len(matched_attrs) < attr_num:
            pool = df_attr['name'].tolist()
        else:
            pool = matched_attrs['name'].tolist()
        
        selected_attrs = random.sample(pool, min(attr_num, len(pool)))

        # 抽样餐厅
        price_col = 'price' if 'price' in df_rest.columns else df_rest.columns[-1]
        
        if consumption == "Budget":
            df_res_filtered = df_rest[df_rest[price_col] < 60]
        elif consumption == "Luxury":
            df_res_filtered = df_rest[df_rest[price_col] > 150]
        else: # Economy
            df_res_filtered = df_rest[(df_rest[price_col] >= 60) & (df_rest[price_col] <= 150)]

        if df_res_filtered.empty: 
            pool_rest = df_rest['name'].tolist()
        else: 
            pool_rest = df_res_filtered['name'].tolist()
        
        selected_rests = random.sample(pool_rest, min(rest_num, len(pool_rest)))

        # 标签映射
        cons_map = {"Budget": "Budget (节省)", "Economy": "Economy (经济)", "Luxury": "Luxury (奢华)"}
        int_map = {"Culture": "Culture (人文历史)", "Nature": "Nature (自然风光)", "Entertainment": "Entertainment (娱乐休闲)", "Shopping": "Shopping (购物逛街)"}
        pace_map = {"Fast": "Fast (紧凑打卡)", "Relaxed": "Relaxed (悠闲随性)"}

        return {
            "case_id": len(self.test_cases) + 1,
            "difficulty": difficulty,
            "basic_info": {
                "target_city": city,
                "people_count": random.randint(1, 5),
                "day_count": day_count
            },
            "hard_constraints": {
                "must_visit_attractions": selected_attrs,
                "must_eat_restaurants": selected_rests,
                "min_activities_count": day_count * 2 + (1 if pace == "Fast" else 0)
            },
            "soft_constraints_persona": {
                "consumption_level": cons_map[consumption],
                "interest_preference": int_map[interest],
                "travel_pace": pace_map[pace]
            }
        }

if __name__ == "__main__":
    generator = TestSetGenerator()
    generator.generate()