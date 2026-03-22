# -*- coding: utf-8 -*-
from .retriever import TravelRetriever

def run_rag_recommendation():
    """
    执行 RAG 检索并返回结构化的推荐数据列表
    返回格式: List[Dict] -> 每个 Dict 对应一个 RecommendationItem
    """
    try:
        engine = TravelRetriever()
        engine.load_logs()
        
        if not engine.load_kg():
            print("❌ 错误：未找到对应城市的图谱文件。")
            return []

        # 执行引擎检索逻辑
        result = engine.get_recommendations()
        
        if not result or len(result) != 3:
            print("❌ 错误：检索引擎未返回有效数据。")
            return []
            
        must_attrs, interest_attrs, rests = result
        
        # --- 核心修改：构造结构化数据列表 ---
        recommendations = []

        # 1. 处理必去景点 (Hard Match)
        for name in must_attrs:
            node_data = engine.kg.nodes[name]
            recommendations.append({
                "name": name,
                "type": f"{node_data.get('type', '景点')} (必去)",
                "price": float(node_data.get('price', 0.0)),
                "description": f"建议游玩时长: {node_data.get('max_t', 2)}h。这是根据您的必去要求匹配的。",
                "recommend_food": None
            })

        # 2. 处理兴趣匹配景点 (Soft Match)
        for name in interest_attrs[:6]: # 取前6个
            if any(r['name'] == name for r in recommendations): continue # 去重
            node_data = engine.kg.nodes[name]
            recommendations.append({
                "name": name,
                "type": node_data.get('type', '景点'),
                "price": float(node_data.get('price', 0.0)),
                "description": f"基于您的兴趣画像推荐。建议游玩: {node_data.get('max_t', 2)}h。",
                "recommend_food": None
            })

        # 3. 处理美食餐厅
        for name in rests[:8]: # 取前8个
            node_data = engine.kg.nodes[name]
            # 判断是否命中想吃的菜
            food_recommend = node_data.get('recommend', '特色美食')
            is_hit = any(fk in str(food_recommend) for fk in engine.hard_filters['food'])
            
            recommendations.append({
                "name": name,
                "type": f"{node_data.get('cuisine', '餐厅')}{' (符合口味)' if is_hit else ''}",
                "price": float(node_data.get('price', 0.0)),
                "description": f"位置: {engine.hard_filters['city']}。本地特色餐厅。",
                "recommend_food": str(food_recommend)
            })

        return recommendations

    except Exception as e:
        print(f"❌ RAG 引擎运行崩溃: {e}")
        import traceback
        traceback.print_exc()
        return []

def run_rag_report_text():
    """
    (可选) 保留原来的文本报告函数，用于控制台调试
    """
    # ... 你可以把原来的代码逻辑放在这里，如果还需要在终端看漂亮输出的话 ...
    pass


'''
# -*- coding: utf-8 -*-
from .retriever import TravelRetriever

def run_rag_recommendation():
    try:
        engine = TravelRetriever()
        engine.load_logs()
        
        if not engine.load_kg():
            return "❌ 错误：未找到对应城市的图谱文件。"

        # 打印详情
        print(f"\n" + "─"*50)
        print(f"📊 [RAG 检索输入详情]")
        print(f"  > 硬约束 (Hard Filters):")
        print(f"    - 目标城市: {engine.hard_filters['city']}")
        print(f"    - 必去关键词: {engine.hard_filters['must_visit']}")
        print(f"    - 计划预算: {engine.hard_filters['budget']} 元")
        print(f"    - 想吃的菜: {engine.hard_filters['food']}")
        print(f"─"*50)

        # 执行检索
        result = engine.get_recommendations()
        
        # 安全检查：防止返回 None 或 长度不对
        if not result or len(result) != 3:
            return "❌ 错误：检索引擎未返回有效数据。"
            
        must_attrs, interest_attrs, rests = result
        
        output = []
        output.append("\n" + "="*50)
        city_name = engine.hard_filters['city'] or "目的地"
        output.append(f"🚀 [RAG 增强生成] 为您定制的 {city_name} 方案：")
        output.append("="*50)

        output.append(f"\n📍 【核心打卡】(根据您的要求优先匹配)")
        if not must_attrs: output.append(" - 暂无精确匹配景点")
        for a in must_attrs:
            d = engine.kg.nodes[a]
            output.append(f" ✅ {a} [{d.get('type')}] | 建议:{d.get('max_t')}h")

        output.append(f"\n✨ 【同类推荐】(基于您的兴趣画像)")
        for a in interest_attrs[:6]:
            d = engine.kg.nodes[a]
            output.append(f" - {a} [{d.get('type')}] | 门票:{d.get('price')}元")

        output.append(f"\n🍽️ 【美食地图】")
        for r in rests[:8]:
            d = engine.kg.nodes[r]
            # 标识出命中菜品的餐厅
            is_hit = any(fk in str(d.get('recommend','')) for fk in engine.hard_filters['food'])
            prefix = "🔥 " if is_hit else " - "
            output.append(f" {prefix}{r} ({d.get('cuisine')}) | 招牌: {d.get('recommend')}")
        
        output.append("\n" + "="*50)
        return "\n".join(output)

    except Exception as e:
        import traceback
        return f"❌ RAG 引擎运行崩溃: {e}\n{traceback.format_exc()}"
'''