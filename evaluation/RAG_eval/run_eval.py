# -*- coding: utf-8 -*-
import os
import sys
import json
import re
import pandas as pd
from tqdm import tqdm
import time
import traceback

# --- 1. 路径环境配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir)) 

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# --- 2. 导入项目组件 ---
from model_provider import SharedModelProvider
from RAG.rag_system.planner import TravelPlanner

# --- 3. 定义评测常量 ---
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

# --- 4. 辅助函数 ---
def load_city_attraction_data(city_zh):
    """
    [高精度反查] 从 attractions/{city}/attractions.csv 加载数据
    建立 { 景点名称: 景点类型 } 的映射
    """
    city_en = CITY_EN_MAP.get(city_zh)
    # 路径匹配：/public/home/azsy/MindJourney_LoRA/MindJourneyAgent/RAG/kg_travel/attractions/beijing/attractions.csv
    csv_path = os.path.join(root_dir, f"RAG/kg_travel/attractions/{city_en}/attractions.csv")
    
    if not os.path.exists(csv_path):
        return {}
    try:
        # 只读取 name 和 type 列提高速度
        df = pd.read_csv(csv_path, usecols=['name', 'type'])
        return pd.Series(df.type.values, index=df.name).to_dict()
    except Exception:
        return {}

def run_batch():
    json_path = os.path.join(current_dir, "data", "benchmark_test_set.json")
    log_base_dir = os.path.join(current_dir, "logs")
    os.makedirs(log_base_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    
    print(f"🚀 启动全量评测：共 {len(test_cases)} 个用例")
    resource = SharedModelProvider.get_instance()
    summary_results = []

    for index, case in enumerate(tqdm(test_cases, desc="MindJourney Eval")):
        case_id = case.get('case_id', index + 1)
        city_zh = case['basic_info']['target_city']
        difficulty = case.get('difficulty', 'Unknown')
        log_file_path = os.path.join(log_base_dir, f"case_{case_id}_{city_zh}_{difficulty}.txt")
        
        with open(log_file_path, "w", encoding="utf-8") as log_f:
            try:
                # 数据准备
                people = case['basic_info']['people_count']
                days = case['basic_info']['day_count']
                target_interest = re.sub(r'\(.*?\)', '', case['soft_constraints_persona']['interest_preference']).strip()
                target_cons = re.sub(r'\(.*?\)', '', case['soft_constraints_persona']['consumption_level']).strip()
                
                # 必去景点清洗（去掉括号内容以便匹配）
                must_attrs = [a.split('(')[0].strip() for a in case['hard_constraints']['must_visit_attractions']]
                must_rests = [r.split('(')[0].strip() for r in case['hard_constraints']['must_eat_restaurants']]
                all_must = must_attrs + must_rests
                
                # 加载该城市 CSV 反查字典
                city_db = load_city_attraction_data(city_zh)

                # 运行规划器
                planner = TravelPlanner(user_data=case, model=resource.model, tokenizer=resource.tokenizer)
                report_text = planner.run()
                
                # 方案切割
                plans = re.split(r'####', report_text)
                plan_contents = [p.strip() for p in plans if len(p.strip()) > 50]

                # --- 写入日志头 ---
                log_f.write(f"CASE ID: {case_id} | CITY: {city_zh} | DIFF: {difficulty}\n")
                log_f.write(f"IMAGE: Interest={target_interest}, Consumption={target_cons}\n")
                log_f.write("="*60 + "\n\n")
                log_f.write("[Report Body]\n" + report_text + "\n\n")
                log_f.write("="*20 + " 专项实验评估结果 " + "="*20 + "\n\n")

                # 🧪 实验 A: 全集满足率 (Global CSR)
                global_hits = [item for item in all_must if item in report_text]
                global_csr_val = len(global_hits) / len(all_must)
                log_f.write(f"🧪 [实验 A] 全集约束满足率 (Global CSR): {len(global_hits)}/{len(all_must)}\n")
                for item in all_must:
                    status = "✅" if item in report_text else "❌"
                    log_f.write(f"   {status} {item}\n")
                log_f.write("\n")

                # 🧪 实验 B: 方案一
                p1_csr = 0
                if len(plan_contents) >= 1:
                    p1_hits = sum(1 for item in all_must if item in plan_contents[0])
                    p1_csr = p1_hits / len(all_must)
                    log_f.write(f"🧪 [实验 B] 方案一：硬约束满足率: {p1_hits}/{len(all_must)} ({p1_csr:.1%})\n\n")

                # 🧪 实验 C: 方案二 (精确反向查表)
                interest_precision = -1.0 
                if len(plan_contents) >= 2:
                    log_f.write(f"🧪 [实验 C] 方案二：兴趣画像契合度 ({target_interest}):\n")
                    # 1. 仅提取“浏览”行中的内容
                    sightseeing_pattern = r'- (?:上午|下午|晚上)浏览[：:]\s*([^(\s（]+)'
                    extracted_raw = re.findall(sightseeing_pattern, plan_contents[1])
                    cleaned_names = [n.strip() for n in extracted_raw if "无" not in n and len(n.strip()) > 0]
                    
                    # 2. 识别系统主动推荐点（且存在于CSV中）
                    extra_recommendations = [n for n in cleaned_names if n not in must_attrs and n in city_db]
                    
                    if not extra_recommendations:
                        log_f.write(f"   - 方案二提取的游览点: {cleaned_names}\n")
                        log_f.write(f"   - 其中系统主动推荐的点: []\n")
                        log_f.write(f"   ✅ 结论: 兴趣契合度无体现 (未推荐额外景点)\n\n")
                    else:
                        # 3. 执行 CSV 类型反查
                        interest_keywords = INTEREST_MAP.get(target_interest, [])
                        hit_recs = []
                        for attr in extra_recommendations:
                            # 从 CSV 字典中取出的 type
                            csv_type = str(city_db.get(attr, ""))
                            # 关键词匹配
                            if any(kw in csv_type for kw in interest_keywords):
                                hit_recs.append(f"{attr}[{csv_type}]")
                        
                        interest_precision = len(hit_recs) / len(extra_recommendations)
                        log_f.write(f"   - 方案二提取的游览点: {cleaned_names}\n")
                        log_f.write(f"   - 其中系统主动推荐的点: {extra_recommendations}\n")
                        log_f.write(f"   - 符合画像类型的点(基于CSV反查): {hit_recs}\n")
                        log_f.write(f"   ✅ 结论: 兴趣契合度 {interest_precision:.1%}\n\n")

                # 🧪 实验 D: 方案三 (修复浮点数)
                cons_success = False
                daily_avg = 0
                if len(plan_contents) >= 3:
                    log_f.write(f"🧪 [实验 D] 方案三：消费定制测试 ({target_cons}):\n")
                    price_match = re.search(r'总预算预估[：:]([\d.]+)元', plan_contents[2])
                    if price_match:
                        total_p = float(price_match.group(1))
                        daily_avg = total_p / (people * days)
                        log_f.write(f"   - 方案总预估价: {total_p} 元\n")
                        log_f.write(f"   - 换算人均每天: {daily_avg:.2f} 元\n")
                        
                        if target_cons == "Budget": cons_success = daily_avg < 200
                        elif target_cons == "Economy": cons_success = daily_avg < 600
                        else: cons_success = daily_avg >= 600
                        log_f.write(f"   {'✅' if cons_success else '❌'} 结论: {'符合' if cons_success else '不符合'} {target_cons}\n")
                    else:
                        log_f.write("   ❌ 错误: 方案三中未捕捉到预算输出。\n")

                # 记录汇总
                summary_results.append({
                    "case_id": case_id, "city": city_zh, "difficulty": difficulty,
                    "global_csr": global_csr_val, "plan1_csr": p1_csr,
                    "interest_precision": interest_precision, "daily_avg": daily_avg,
                    "cons_level": target_cons, "cons_ok": int(cons_success)
                })

            except Exception as e:
                log_f.write(f"\n❌ CRITICAL ERROR: {str(e)}\n{traceback.format_exc()}")

    # 保存汇总结果
    pd.DataFrame(summary_results).to_csv(os.path.join(log_base_dir, "rag_eval_summary.csv"), index=False, encoding='utf-8-sig')
    print(f"\n✅ 完成！汇总报表见: {log_base_dir}/rag_eval_summary.csv")

if __name__ == "__main__":
    run_batch()