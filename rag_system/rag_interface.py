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
        output.append("🚀 [RAG 增强生成] 为您定制的南京方案：")
        output.append("="*50)

        output.append(f"\n📍 【核心打卡】(根据您的要求优先匹配)")
        if not must_attrs: output.append(" - 暂无精确匹配景点")
        for a in must_attrs:
            d = engine.kg.nodes[a]
            output.append(f" ✅ {a} [{d.get('type')}] | 建议:{d.get('max_t')}h")

        output.append(f"\n✨ 【同类推荐】(基于您的 Nature 画像)")
        for a in interest_attrs[:6]:
            d = engine.kg.nodes[a]
            output.append(f" - {a} [{d.get('type')}] | 门票:{d.get('price')}元")

        output.append(f"\n🍽️ 【美食地图】(已为您优先寻找鸭血粉丝汤)")
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