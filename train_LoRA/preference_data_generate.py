import json
import random
import re

# 1. 定义偏好模式
PREFERENCE_SCHEMA = {
    "Consumption": ["Budget", "Economy", "Luxury"],
    "Interest": ["Culture", "Nature", "Entertainment", "Shopping", "其它"],
    "Pace": ["Fast", "Relaxed"]
}

# 2. 映射库配置
MAP_LIBRARY = {
    "Consumption": {
        "Luxury": ["商务座", "一等座", "软卧", "别墅", "五星级", "豪华型"],
        "Economy": ["二等座", "硬卧", "软座", "酒店", "公寓", "四星级"],
        "Budget": ["硬座", "无座", "普通列车", "客栈", "民宿", "青年旅舍"]
    },
    "Interest": {
        "Culture": ["摄影", "人文", "书店", "古迹", "博物馆", "特色集市", "食品茶酒", "历史", "艺术"],
        "Nature": ["徒步", "山水", "赏秋", "自驾", "森林", "登山", "海滨", "郊游"],
        "Entertainment": ["周末游", "邮轮", "海滨度假", "乐园", "亲子", "演出", "休闲", "综合商场"],
        "Shopping": ["综合商场", "特色集市", "服饰鞋包", "化妆品", "珠宝首饰", "钟表", "免税", "数码产品"]
    }
}

# 3. 工具函数
def extract_number(text):
    if not text or not isinstance(text, str) or text == "不限":
        return None
    nums = re.findall(r'\d+', text)
    return int(nums[0]) if nums else None

def get_people_multiplier(who_text):
    mapping = {"一个人": 1, "亲子": 2.5, "夫妻": 2, "和朋友": 2}
    return mapping.get(who_text, 1)

# 4. 偏好判定逻辑核心
def determine_consumption(item):
    """基于价格、席别、酒店类型综合判定"""
    # A. 基于硬件设施判定 (优先级最高)
    hardware_text = " ".join([item.get('seat_type', ''), item.get('hotel_type', ''), item.get('train_type', '')])
    for level, keywords in MAP_LIBRARY["Consumption"].items():
        if any(k in hardware_text for k in keywords):
            return level

    # B. 基于数学计算判定 (由cost/price计算)
    cost = extract_number(item.get('cost') or item.get('price'))
    days = extract_number(item.get('day', '1天'))
    people = get_people_multiplier(item.get('who', '不限'))
    
    if cost and days and people:
        val = cost / (days * people)
        if val < 250: return "Budget"
        if 250 <= val < 400: return "Economy"
        return "Luxury"

    # C. 兜底：随机
    return random.choice(PREFERENCE_SCHEMA["Consumption"])

def determine_interest(item):
    """基于玩法、商品类型、菜系判定"""
    text = " ".join([item.get('play', ''), item.get('type', ''), item.get('cuisine', ''), item.get('category', '')])
    for category, keywords in MAP_LIBRARY["Interest"].items():
        if any(k in text for k in keywords):
            return category
    return random.choice(PREFERENCE_SCHEMA["Interest"])

def determine_pace(item):
    """基于玩法判定节奏"""
    play_text = item.get('play', '')
    # 慢节奏关键词
    if any(k in play_text for k in ["摄影", "徒步", "赏秋", "海滨度假", "度假", "悠闲"]):
        return "Relaxed"
    # 快节奏关键词
    if any(k in play_text for k in ["自驾", "特种兵", "高铁", "直达"]):
        return "Fast"
    return random.choice(PREFERENCE_SCHEMA["Pace"])

# 5. 主转换函数
def preprocess_with_preferences(input_file, output_file):
    processed_count = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                if not line.strip(): continue
                item = json.loads(line)
                
                # 注入推断出的偏好
                item['pref_consumption'] = determine_consumption(item)
                item['pref_interest'] = determine_interest(item)
                item['pref_pace'] = determine_pace(item)
                
                # 构造 User Agent 的剧本（内心独白）
                # 这个字段将用于 user.py 里的 system_prompt
                # item['preference_script'] = (
                #     f"【内心偏好设定】\n"
                #     f"1. 消费观：你偏向于{item['pref_consumption']}级别的消费。\n"
                #     f"2. 兴趣点：你对{item['pref_interest']}方面最感兴趣。\n"
                #     f"3. 节奏感：你希望行程是{item['pref_pace']}的。\n"
                #     f"请在对话中根据助手的引导逐步表达这些偏好，不要一开始就全说出来。"
                # )
                
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
                processed_count += 1
    
    print(f"处理完成！共生成 {processed_count} 条带偏好的种子数据。")

if __name__ == "__main__":
    # 假设你的原始文件叫 seed_data.txt
    preprocess_with_preferences('data/seed_data.jsonl', 'data/seed_data_with_pref.jsonl')