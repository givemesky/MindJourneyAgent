# 测试景点和餐厅RAG推荐文件：运行推荐逻辑并输出最终结果

from retriever import TravelRetriever
import os

def format_output():
    engine = TravelRetriever()
    engine.load_logs()
    
    # 1. 恢复用户画像识别结果显示
    print(f"\n=== 用户画像识别完成 ===")
    print(f"硬约束: {engine.hard_filters}")
    print(f"软偏好: {engine.soft_priors}\n")

    if not engine.load_kg():
        print("未找到对应城市的图谱文件。")
        return

    # 打印调试信息：节点属性键
    sample_node = list(engine.kg.nodes(data=True))[0]
    print(f"调试：节点示例属性列表: {list(sample_node[1].keys())}")

    must_attrs, interest_attrs, rests = engine.get_recommendations()

    print(f"开始图搜索... 关键词:{engine.hard_filters['must_visit']}, 兴趣:{engine.soft_priors['interest']}")
    print(f"检索完成：找到景点 {len(must_attrs) + len(interest_attrs)} 个，餐厅 {len(rests)} 个")

    # 2. 结构化输出
    print(f"\n★ 用户指定必去景点 ({len(must_attrs)}个):")
    print("=" * 40)
    for a in must_attrs:
        data = engine.kg.nodes[a]
        print(f"【必选】名称: {a} | 类型: {data.get('type')} | 门票: {data.get('price')}元 | 建议时长: {data.get('max_t')}h")

    print(f"\n☆ 为您推荐的匹配景点 ({len(interest_attrs)}个，展示前5):")
    print("-" * 40)
    for a in interest_attrs[:5]:
        data = engine.kg.nodes[a]
        print(f"名称: {a} | 类型: {data.get('type')} | 门票: {data.get('price')}元 | 建议时长: {data.get('max_t')}h")

    print(f"\n🍴 周边推荐餐厅 ({len(rests)}个，展示前10):")
    print("-" * 40)
    for r in rests[:10]:
        data = engine.kg.nodes[r]
        print(f"餐厅: {r} | 菜系: {data.get('cuisine')} | 招牌: {data.get('recommend')}")

if __name__ == "__main__":
    format_output()