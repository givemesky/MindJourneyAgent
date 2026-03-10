import pandas as pd
import networkx as nx
import os
import math

# ================= 路径配置 (相对于脚本所在位置) =================
# 脚本位于 kg_travel/attracions_restaurants_kg/
# 景点数据位于 kg_travel/attractions_kg/attractions/
ATTRACTION_BASE = "../attractions_kg/attractions"
# 餐厅数据位于 kg_travel/restaurants_kg/
RESTAURANT_BASE = "../restaurants_kg"
# 输出目录位于脚本同级目录下的 output 文件夹
OUTPUT_ROOT = "./output_attracions_restaurants_kg"

# 距离阈值设定 (单位: km)
ATTR_TO_ATTR_DIST = 10.0  # 景点与景点
RES_TO_RES_DIST = 5.0    # 餐厅与餐厅
ATTR_TO_RES_DIST = 1.0   # 景点周边餐厅 (1km内，方便游玩后步行)

# ================= 辅助工具函数 =================

def haversine(lat1, lon1, lat2, lon2):
    """计算地球两点间距离 (km)"""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def build_unified_kg(city_name, attr_csv, res_csv):
    """构建实体中心化的统一城市图谱"""
    print(f"\n>>> 正在构建统一图谱: {city_name}")
    G = nx.Graph()

    # 1. 加载并添加景点节点
    try:
        df_attr = pd.read_csv(attr_csv)
        attrs = df_attr.to_dict('records')
        for a in attrs:
            G.add_node(a['name'], 
                       label="attraction",
                       attr_id=a['id'],
                       type=a['type'],
                       lat=a['lat'], lon=a['lon'],
                       price=a['price'],
                       open=a['opentime'], end=a['endtime'],
                       min_t=a['recommendmintime'], max_t=a['recommendmaxtime'])
        print(f"  - 已添加 {len(attrs)} 个景点")
    except Exception as e:
        print(f"  [错误] 景点数据加载失败: {e}")
        return

    # 2. 加载并添加餐厅节点
    try:
        df_res = pd.read_csv(res_csv)
        # 处理可能的空值
        df_res['recommendedfood'] = df_res['recommendedfood'].fillna("暂无推荐")
        rests = df_res.to_dict('records')
        for r in rests:
            G.add_node(r['name'], 
                       label="restaurant",
                       res_id=r['id'],
                       cuisine=r['cuisine'],
                       price=r['price'],
                       lat=r['lat'], lon=r['lon'],
                       open=r['opentime'], end=r['endtime'],
                       recommend=str(r['recommendedfood']))
        print(f"  - 已添加 {len(rests)} 个餐厅")
    except Exception as e:
        print(f"  [错误] 餐厅数据加载失败: {e}")
        return

    # 3. 空间关联计算
    nodes_attr = [n for n, d in G.nodes(data=True) if d.get('label') == "attraction"]
    nodes_res = [n for n, d in G.nodes(data=True) if d.get('label') == "restaurant"]

    # A. 景点-景点 (10km)
    for i in range(len(nodes_attr)):
        for j in range(i + 1, len(nodes_attr)):
            n1, n2 = nodes_attr[i], nodes_attr[j]
            d = haversine(G.nodes[n1]['lat'], G.nodes[n1]['lon'], G.nodes[n2]['lat'], G.nodes[n2]['lon'])
            if d < ATTR_TO_ATTR_DIST:
                G.add_edge(n1, n2, relation="nearby_attr", distance=round(d, 2))

    # B. 餐厅-餐厅 (5km)
    for i in range(len(nodes_res)):
        for j in range(i + 1, len(nodes_res)):
            n1, n2 = nodes_res[i], nodes_res[j]
            d = haversine(G.nodes[n1]['lat'], G.nodes[n1]['lon'], G.nodes[n2]['lat'], G.nodes[n2]['lon'])
            if d < RES_TO_RES_DIST:
                G.add_edge(n1, n2, relation="nearby_res", distance=round(d, 2))

    # C. 景点-餐厅 (1km) —— 游玩就餐核心线索
    for na in nodes_attr:
        for nr in nodes_res:
            d = haversine(G.nodes[na]['lat'], G.nodes[na]['lon'], G.nodes[nr]['lat'], G.nodes[nr]['lon'])
            if d < ATTR_TO_RES_DIST:
                G.add_edge(na, nr, relation="attr_dining", distance=round(d, 2))

    # 保存图谱
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
    
    output_path = os.path.join(OUTPUT_ROOT, f"{city_name}_unified_kg.gml")
    nx.write_gml(G, output_path)
    print(f"  [成功] 节点: {G.number_of_nodes()}, 边: {G.number_of_edges()}")
    print(f"  [文件] {output_path}")

# ================= 主程序 =================

def main():
    # 以景点目录作为城市列表来源
    if not os.path.exists(ATTRACTION_BASE):
        print(f"错误: 找不到景点目录 {ATTRACTION_BASE}")
        return

    cities = [d for d in os.listdir(ATTRACTION_BASE) if os.path.isdir(os.path.join(ATTRACTION_BASE, d))]
    
    print(f"准备开始构建 {len(cities)} 个城市的统一图谱...")

    for city in cities:
        attr_csv = os.path.join(ATTRACTION_BASE, city, "attractions.csv")
        res_csv = os.path.join(RESTAURANT_BASE, city, f"restaurants_{city}.csv")
        
        if os.path.exists(attr_csv) and os.path.exists(res_csv):
            build_unified_kg(city, attr_csv, res_csv)
        else:
            if not os.path.exists(attr_csv):
                print(f"跳过 {city}: 缺少景点文件 {attr_csv}")
            if not os.path.exists(res_csv):
                print(f"跳过 {city}: 缺少餐厅文件 {res_csv}")

    print("\n所有任务执行完毕！")

if __name__ == "__main__":
    main()