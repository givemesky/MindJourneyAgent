# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import pandas as pd

# --- 1. 路径环境配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir)) 

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# --- 2. 导入组件 ---
from model_provider import SharedModelProvider
from RAG.rag_system.planner import TravelPlanner

# --- 3. 定义兴趣关键词与城市映射 ---
INTEREST_MAP = {
    "Culture": ["人文", "文化", "历史", "古", "艺术", "建筑", "博物馆", "遗址", "人文景观", "古迹", "园林"],
    "Nature": ["自然", "公园", "森林", "山", "湖", "岛", "湿地", "风景", "草", "郊外", "自然风光", "动物", "湾"],
    "Entertainment": ["乐园", "迪士尼", "娱乐", "游乐", "欢乐谷", "演出", "主题", "体育", "网红", "美食", "船", "夜景"],
    "Shopping": ["商业", "商圈", "购物", "步行街", "广场", "中心", "免税", "买买买", "奥特莱斯", "市集"]
}

CITY_EN_MAP = {
    "北京": "beijing", "成都": "chengdu", "深圳": "shenzhen", "上海": "shanghai",
    "苏州": "suzhou", "武汉": "wuhan", "南京": "nanjing", "广州": "guangzhou",
    "重庆": "chongqing", "杭州": "hangzhou"
}

def load_city_attraction_data(city_zh):
    """从 CSV 加载景点及其类型映射"""
    city_en = CITY_EN_MAP.get(city_zh)
    csv_path = os.path.join(root_dir, f"RAG/kg_travel/attractions/{city_en}/attractions.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️ 警告: 找不到城市数据文件 {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    return pd.Series(df.type.values, index=df.name).to_dict()

def test_one():
    json_path = os.path.join(current_dir, "data", "benchmark_test_set.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    case = test_cases[0] # 测试第一个用例
    city_zh = case['basic_info']['target_city']
    people = case['basic_info']['people_count']
    days = case['basic_info']['day_count']
    # 提取画像标签并去除括号说明
    target_interest = re.sub(r'\(.*?\)', '', case['soft_constraints_persona']['interest_preference']).strip()
    target_cons = re.sub(r'\(.*?\)', '', case['soft_constraints_persona']['consumption_level']).strip()
    
    must_attrs = [a.split('(')[0].strip() for a in case['hard_constraints']['must_visit_attractions']]
    must_rests = [r.split('(')[0].strip() for r in case['hard_constraints']['must_eat_restaurants']]
    all_must = must_attrs + must_rests

    # 加载该城市的景点库以供反查
    city_db = load_city_attraction_data(city_zh)

    print(f"📊 [测试配置] 城市: {city_zh} | 人数: {people} | 天数: {days}")
    print(f"🎯 [画像设定] 兴趣: {target_interest} | 消费水平: {target_cons}")
    print("-" * 60)

    # 运行规划器
    resource = SharedModelProvider.get_instance()
    planner = TravelPlanner(user_data=case, model=resource.model, tokenizer=resource.tokenizer)
    report_text = planner.run()

    # --- 文本切分 ---
    # 使用正则表达式拆分方案，防止空块
    plans = re.split(r'####', report_text)
    plan_contents = [p.strip() for p in plans if len(p.strip()) > 50]

    # --- 实验评估开始 ---
    print("\n" + "★"*20 + " 专项实验评估结果 " + "★"*20)

    # [实验 A] 全集匹配 (Global CSR)
    global_hits = [item for item in all_must if item in report_text]
    print(f"\n🧪 [实验 A] 全集约束满足率 (Global CSR): {len(global_hits)}/{len(all_must)}")
    for item in all_must:
        status = "✅" if item in report_text else "❌"
        print(f"   {status} {item}")

    # [实验 B] 方案一：硬约束满足率
    if len(plan_contents) >= 1:
        plan1 = plan_contents[0]
        p1_hits = sum(1 for item in all_must if item in plan1)
        print(f"\n🧪 [实验 B] 方案一：硬约束满足率: {p1_hits}/{len(all_must)} ({(p1_hits/len(all_must)):.1%})")

    # [实验 C] 方案二：兴趣偏好契合度
    if len(plan_contents) >= 2:
        print(f"\n🧪 [实验 C] 方案二：兴趣画像契合度 ({target_interest}):")
        plan2 = plan_contents[1]
        
        # 1. 寻找出现的非硬约束景点
        extra_recommendations = []
        for attr_name in city_db.keys():
            if attr_name in plan2 and attr_name not in must_attrs:
                extra_recommendations.append(attr_name)
        
        # 2. 反查类型匹配
        interest_keywords = INTEREST_MAP.get(target_interest, [])
        hit_recs = []
        for attr in extra_recommendations:
            attr_type = str(city_db.get(attr, ""))
            if any(kw in attr_type for kw in interest_keywords):
                hit_recs.append(f"{attr}[{attr_type}]")

        print(f"   - 系统主动推荐的景点: {extra_recommendations}")
        print(f"   - 符合画像类型的景点: {hit_recs}")
        if extra_recommendations:
            precision = len(hit_recs) / len(extra_recommendations)
            print(f"   ✅ 结论: 兴趣契合度 {precision:.1%}")
        else:
            print("   ⚠️ 结论: 方案二未发现额外的景点推荐")

    # [实验 D] 方案三：消费水平合理性
    if len(plan_contents) >= 3:
        plan3 = plan_contents[2]
        print(f"\n🧪 [实验 D] 方案三：消费定制测试 ({target_cons}):")
        
        # 【精准捕捉逻辑】：匹配“总预算预估：XXXX元”
        # 使用 \d+ 匹配数字，兼容中文全角和半角冒号
        price_match = re.search(r'总预算预估[：:](\d+)元', plan3)
        
        if price_match:
            total_price = float(price_match.group(1))
            # 这里的计算基于你设定的逻辑：人均每天 = 总价 / (人数 * 天数)
            daily_avg = total_price / (people * days)
            print(f"   - 方案总预估价: {total_price} 元")
            print(f"   - 换算人均每天: {daily_avg:.2f} 元")
            
            # 阶梯判定
            success = False
            evaluation = ""
            if target_cons == "Budget":
                success = daily_avg < 200
                evaluation = "符合 Budget (<200)" if success else "不符合 (需<200)"
            elif target_cons == "Economy":
                success = daily_avg < 500
                evaluation = "符合 Economy (<500)" if success else "不符合 (需<500)"
            elif target_cons == "Luxury":
                success = daily_avg >= 500
                evaluation = "符合 Luxury (>=500)" if success else "不符合 (需>=500)"
                
            print(f"   {'✅' if success else '❌'} 结论: {evaluation}")
        else:
            print("   ❌ 错误: 方案三中未捕捉到格式化的预算输出。")
            # 调试信息：打印方案三最后两行看看格式
            last_lines = "\n".join(plan3.split('\n')[-3:])
            print(f"     [调试末尾文本]: {last_lines}")

    print("\n" + "★"*60)

if __name__ == "__main__":
    test_one()