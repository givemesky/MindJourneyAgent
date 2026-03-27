# -*- coding: utf-8 -*-
import json
import re
import os
from datasets import load_dataset
import pandas as pd

# ================= 配置与映射表 =================

# 1. 偏好标签映射 (根据 ChinaTravel 论文及常见偏好词汇)
# 你可以根据你的 Likelihood 函数定义进一步微调这些映射关系
PERSONA_MAPPING = {
    # 旅行节奏 (Pace)
    "more attractions": "Fast",
    "visit as many attractions as possible": "Fast",
    "maximize daily_attraction_count": "Fast",
    "leisurely": "Relaxed",
    "relaxed": "Relaxed",
    "fewer attractions": "Relaxed",
    
    # 消费水平 (Consumption)
    "expensive hotels": "Luxury",
    "high budget": "Luxury",
    "luxury": "Luxury",
    "cheap hotels": "Economy",
    "low cost": "Economy",
    "budget friendly": "Economy",
    "economy": "Economy",
    "moderate": "Standard",
    
    # 兴趣偏好 (Interest)
    "natural scenery": "Nature",
    "nature": "Nature",
    "cultural history": "Culture",
    "historical": "Culture",
    "museums": "Culture",
    "shopping": "Shopping",
    "modern city": "Shopping"
}

def map_preference(pref_str):
    """将原始偏好描述映射到 DPI-ME 标签"""
    pref_str = pref_str.lower()
    # 默认值
    result = {"Consumption": "Economy", "Interest": "Culture", "Pace": "Fast"}
    
    for key, value in PERSONA_MAPPING.items():
        if key in pref_str:
            if value in ["Fast", "Relaxed"]: result["Pace"] = value
            elif value in ["Budget", "Economy", "Luxury"]: result["Consumption"] = value
            elif value in ["Nature", "Culture", "Shopping", "Entertainment"]: result["Interest"] = value
    return result

def extract_budget(logic_list_str):
    """从 hard_logic_py 字符串列表中提取预算数字"""
    # 匹配类似 total_cost<=1300 或 cost<=2000 的数字
    match = re.search(r'cost<=\s*(\d+)', logic_list_str.replace(" ", ""))
    if match:
        return int(match.group(1))
    return 2000  # 默认兜底值

# ================= 主处理逻辑 =================

def preprocess_china_travel():
    print("🚀 正在从 HuggingFace 下载数据集...")
    # 加载数据集
    dataset = load_dataset("LAMDA-NeSy/ChinaTravel", name="preference")
    # 获取 preference0_base50 这个 split
    df = pd.DataFrame(dataset['preference0_base50'])

    processed_data = []

    print(f"📊 开始处理 {len(df)} 条原始数据...")

    for _, row in df.iterrows():
        # 1. 提取硬约束
        budget = extract_budget(str(row['hard_logic_py']))
        
        # 2. 标签对齐
        persona_gt = map_preference(str(row['preference_en']))
        
        # 3. 构造初始 Query (脱敏：删掉偏好描述，用于测试主动提问能力)
        # 假设我们只给系统第一句话的基础信息
        raw_text = row['nature_language']
        # 简单的切分逻辑：通常第一句是需求，后面是具体偏好。
        # 这里我们保留基础事实，去掉明显的偏好诱导词。
        base_query = f"我想从{row['start_city']}去{row['target_city']}玩{row['days']}天，人数{row['people_number']}人。"

        item = {
            "uid": row['uid'],
            "initial_query": base_query,
            "full_user_speech": raw_text,
            "ground_truth": {
                "target_city": row['target_city'],
                "days": int(row['days']),
                "people_count": int(row['people_number']),
                "budget_limit": budget,
                "persona": persona_gt
            },
            # 存储原始偏好描述供人工核对
            "raw_pref_desc": row['preference_en']
        }
        processed_data.append(item)

    # 保存为本地 JSON 文件
    output_path = "dpi_me_test_cases.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 预处理完成！共生成 {len(processed_data)} 个测试用例。")
    print(f"📂 文件已保存至: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    preprocess_china_travel()