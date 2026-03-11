import pandas as pd
import networkx as nx
import os
import math

# ================= 路径配置 (相对于 city_kg 目录) =================
# 原始数据都在父目录下
BASE_DIR = ".." 
ATTRACTION_BASE = os.path.join(BASE_DIR, "attractions")
RESTAURANT_BASE = os.path.join(BASE_DIR, "restaurants")
ACCOMMODATION_BASE = os.path.join(BASE_DIR, "accommodations")

# 输出目录
OUTPUT_ROOT = "./output_city_kg"

# 距离阈值设定 (单位: km)
ATTR_TO_ATTR_DIST = 10.0   # 景点-景点: 用于行程分天聚类
ATTR_TO_RES_DIST = 2.0     # 景点-餐厅: 决定游玩后就餐便利性
ATTR_TO_ACCOMM_DIST = 5.0  # 景点-住宿: 决定该区域是否适合作为下榻点
RES_TO_ACCOMM_DIST = 3.0   # 餐厅-住宿: 决定晚餐后回酒店的距离

# ================= 核心工具 =================

def haversine(lat1, lon1, lat2, lon2):
    """计算地球两点间距离 (km)"""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def build_unified_city_kg(city_name, attr_csv, res_csv, acc_csv):
    print(f"\n>>> 正在构建三位一体统一图谱: {city_name}")
    G = nx.Graph()

    # 1. 加载景点节点
    df_attr = pd.read_csv(attr_csv)
    attrs = df_attr.to_dict('records')
    for a in attrs:
        G.add_node(a['name'], 
                   label="attraction",
                   type=a['type'],
                   price=a['price'],
                   lat=a['lat'], lon=a['lon'],
                   max_t=a['recommendmaxtime'])

    # 2. 加载餐厅节点
    df_res = pd.read_csv(res_csv)
    df_res['recommendedfood'] = df_res['recommendedfood'].fillna("暂无")
    rests = df_res.to_dict('records')
    for r in rests:
        G.add_node(r['name'], 
                   label="restaurant",
                   cuisine=r['cuisine'],
                   price=r['price'],
                   lat=r['lat'], lon=r['lon'],
                   recommend=str(r['recommendedfood']))

    # 3. 加载住宿节点
    df_acc = pd.read_csv(acc_csv)
    accs = df_acc.to_dict('records')
    for h in accs:
        G.add_node(h['name'], 
                   label="accommodation",
                   type=h.get('type', '酒店'),
                   price=h['price'],
                   lat=h['lat'], lon=h['lon'])

    # 分类获取节点以便计算边
    nodes_attr = [n for n, d in G.nodes(data=True) if d.get('label') == "attraction"]
    nodes_res = [n for n, d in G.nodes(data=True) if d.get('label') == "restaurant"]
    nodes_acc = [n for n, d in G.nodes(data=True) if d.get('label') == "accommodation"]

    print(f"  节点统计: 景点({len(nodes_attr)}) 餐厅({len(nodes_res)}) 住宿({len(nodes_acc)})")

    # 4. 建立空间关系
    # A. 景点-景点 (A2A)
    for i in range(len(nodes_attr)):
        for j in range(i + 1, len(nodes_attr)):
            n1, n2 = nodes_attr[i], nodes_attr[j]
            d = haversine(G.nodes[n1]['lat'], G.nodes[n1]['lon'], G.nodes[n2]['lat'], G.nodes[n2]['lon'])
            if d < ATTR_TO_ATTR_DIST:
                G.add_edge(n1, n2, relation="nearby_attr", distance=round(d, 2))

    # B. 景点-餐厅 (A2R)
    for na in nodes_attr:
        for nr in nodes_res:
            d = haversine(G.nodes[na]['lat'], G.nodes[na]['lon'], G.nodes[nr]['lat'], G.nodes[nr]['lon'])
            if d < ATTR_TO_RES_DIST:
                G.add_edge(na, nr, relation="attr_dining", distance=round(d, 2))

    # C. 景点-住宿 (A2H)
    for na in nodes_attr:
        for nh in nodes_acc:
            d = haversine(G.nodes[na]['lat'], G.nodes[na]['lon'], G.nodes[nh]['lat'], G.nodes[nh]['lon'])
            if d < ATTR_TO_ACCOMM_DIST:
                G.add_edge(na, nh, relation="attr_stay", distance=round(d, 2))

    # D. 餐厅-住宿 (R2H)
    for nr in nodes_res:
        for nh in nodes_acc:
            d = haversine(G.nodes[nr]['lat'], G.nodes[nr]['lon'], G.nodes[nh]['lat'], G.nodes[nh]['lon'])
            if d < RES_TO_ACCOMM_DIST:
                G.add_edge(nr, nh, relation="res_stay", distance=round(d, 2))

    # 保存
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
    output_path = os.path.join(OUTPUT_ROOT, f"{city_name}_unified_kg.gml")
    nx.write_gml(G, output_path)
    print(f"  [完成] 统一图谱已保存，边总数: {G.number_of_edges()}")

# ================= 主程序 =================

def main():
    if not os.path.exists(ATTRACTION_BASE):
        print(f"错误: 找不到路径 {ATTRACTION_BASE}")
        return

    # 遍历城市目录
    cities = [d for d in os.listdir(ATTRACTION_BASE) if os.path.isdir(os.path.join(ATTRACTION_BASE, d))]
    
    for city in cities:
        attr_f = os.path.join(ATTRACTION_BASE, city, "attractions.csv")
        res_f = os.path.join(RESTAURANT_BASE, city, f"restaurants_{city}.csv")
        acc_f = os.path.join(ACCOMMODATION_BASE, city, "accommodations.csv")
        
        if all(os.path.exists(f) for f in [attr_f, res_f, acc_f]):
            build_unified_city_kg(city, attr_f, res_f, acc_f)
        else:
            print(f"跳过 {city}: 数据不完整 (需检查 attractions, restaurants, accommodations)")

if __name__ == "__main__":
    main()