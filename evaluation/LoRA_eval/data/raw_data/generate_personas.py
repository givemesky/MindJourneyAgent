import json
import os
import random

# 配置路径
INPUT_FILE = r'/public/home/azsy/MindJourney_LoRA/MindJourneyAgent/evaluation/LoRA_eval/data/train.json'
OUTPUT_DIR = r'./data/processed'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'test_personas.json')

def map_consumption(goal_list):
    """映射消费水平"""
    price_val = 200 
    for item in goal_list:
        if any(x in item[2] for x in ["消费", "价格", "门票"]):
            try:
                nums = [int(s) for s in str(item[3]).replace('-', ' ').split() if s.isdigit()]
                if nums: price_val = sum(nums) / len(nums)
                break
            except: pass
    if price_val < 100: return "Budget"
    if price_val <= 250: return "Economy"
    return "Luxury"

def get_best_tag(goal_list, current_stats):
    """挖掘最合适的标签，兼顾稀缺度"""
    possible_tags = set()
    
    # 关键词库
    nature_ks = ['公园', '山', '湖', '风景', '植物园', '动物园', "湿地", "郊外", "树屋", "长城", "颐和园", "天坛", "潭", "岭", "八达岭", "园林"]
    shopping_ks = ['购物', '商场', '中心', '奥特莱斯', '市集', "步行街", "免税", "西单", "王府井", "三里屯", "大栅栏", "前门", "万达", "商圈", "百货"]
    culture_ks = ['故宫', '博物馆', '寺', '庙', '美术馆', '大剧院', '老字号', "文化", "历史", "古迹", "圆明园", "艺术", "恭王府", "雍和宫", "天安门", "纪念馆", "胡同", "四合院"]
    
    has_attraction = False
    for item in goal_list:
        entity, attr, val = item[1], item[2], str(item[3])
        if entity == "景点": has_attraction = True
        combined = entity + attr + val
        
        if any(k in combined for k in shopping_ks): possible_tags.add("Shopping")
        if any(k in combined for k in nature_ks): possible_tags.add("Nature")
        if any(k in combined for k in culture_ks): possible_tags.add("Culture")

    if has_attraction and not (possible_tags & {"Nature", "Shopping", "Culture"}):
        possible_tags.add("Culture")

    if not possible_tags:
        possible_tags.add("Entertainment")

    # 选择当前最缺的标签
    sorted_possibilities = sorted(list(possible_tags), key=lambda t: current_stats[t])
    return sorted_possibilities[0]

def process_and_balance():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = {"Shopping": 0, "Nature": 0, "Culture": 0, "Entertainment": 0}
    HARD_CAP = 125 
    balanced_personas = {}
    
    items = list(data.items())
    random.seed(42)
    random.shuffle(items)

    for d_id, content in items:
        goal_list = content['goal']
        interest_tag = get_best_tag(goal_list, stats)
        
        if stats[interest_tag] >= HARD_CAP:
            continue

        # --- 核心优化部分：提取具体景点和推荐菜 ---
        attractions = [i[3] for i in goal_list if i[1] == "景点" and i[2] == "名称" and str(i[3]) and "id=" not in str(i[3])]
        
        # 提取推荐菜 (CrossWOZ中推荐菜可能是列表也可能是字符串)
        dishes = []
        for i in goal_list:
            if i[1] == "餐馆" and i[2] == "推荐菜":
                val = i[3]
                if isinstance(val, list):
                    dishes.extend(val)
                elif isinstance(val, str) and val:
                    dishes.append(val)

        consumption = map_consumption(goal_list)
        pace = "Fast" if len(set([i[0] for i in goal_list])) >= 3 else "Relaxed"
        
        balanced_personas[d_id] = {
            "persona_id": d_id,
            "ground_truth": {
                "Destination": "北京",
                "Consumption": consumption,
                "Interest": interest_tag,
                "Pace": pace
            },
            "specific_attractions": attractions,
            "specific_dishes": dishes,        # 替换为具体菜品
            "original_goal": goal_list
        }
        stats[interest_tag] += 1

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(balanced_personas, f, indent=4, ensure_ascii=False)
    
    print("\n=== 最终实验数据集分布 (已硬均衡) ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"总计样本数: {sum(stats.values())}")

if __name__ == "__main__":
    process_and_balance()