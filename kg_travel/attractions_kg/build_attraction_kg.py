import pandas as pd
import networkx as nx
import os
import math

# ================= 路径配置 (根据你的目录结构已修正) =================
# 脚本在 attractions_kg 目录下，数据在同级目录的 attractions 文件夹下
DATA_ROOT = "./attractions" 
# 生成的知识图谱保存目录
OUTPUT_ROOT = "./output_attractions_kgs"

# 偏好模式定义
PREFERENCE_SCHEMA = {
    "Consumption": ["Budget", "Economy", "Luxury"],
    "Interest": ["Culture", "Nature", "Entertainment", "Shopping"],
    "Pace": ["Fast", "Relaxed"]
}

# 1. 兴趣维度映射规则
INTEREST_MAPPING = {
    "Culture": ["历史古迹", "博物馆/纪念馆", "红色景点", "美术馆/艺术馆", "文化旅游区"],
    "Nature": ["自然风光", "公园", "动物园"],
    "Entertainment": ["游乐园/体育娱乐", "人文景观"],
    "Shopping": ["商业街区"]
}

# ================= 辅助工具函数 =================

def haversine(lat1, lon1, lat2, lon2):
    """计算两点间的球面距离 (km)"""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def build_city_kg(city_name, csv_path):
    """构建单个城市的知识图谱"""
    print(f"正在处理城市: {city_name}...")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [错误] 无法读取 {csv_path}: {e}")
        return

    G = nx.Graph()

    # 1. 添加偏好标签节点
    for dimension, labels in PREFERENCE_SCHEMA.items():
        for label in labels:
            G.add_node(label, type="preference_label", dimension=dimension)

    # 2. 添加景点节点并建立偏好连接
    attractions = df.to_dict('records')

    for attr in attractions:
        name = attr['name']
        G.add_node(name, 
                   type="attraction",
                   attr_id=attr['id'],
                   lat=attr['lat'],
                   lon=attr['lon'],
                   price=attr['price'],
                   min_time=attr['recommendmintime'],
                   max_time=attr['recommendmaxtime'],
                   open_time=attr['opentime'],
                   end_time=attr['endtime'])

        # --- A. 兴趣映射 ---
        for label, types in INTEREST_MAPPING.items():
            if attr['type'] in types:
                G.add_edge(name, label, relation="has_interest")

        # --- B. 消费映射 ---
        price = attr['price']
        if price <= 30:
            G.add_edge(name, "Budget", relation="has_consumption")
        elif 30 < price < 150:
            G.add_edge(name, "Economy", relation="has_consumption")
        else:
            G.add_edge(name, "Luxury", relation="has_consumption")

        # --- C. 节奏映射 ---
        if attr['recommendmintime'] <= 3.0:
            G.add_edge(name, "Fast", relation="has_pace")
        else:
            G.add_edge(name, "Relaxed", relation="has_pace")

    # 3. 建立空间邻近关系 (NEARBY) - 10km
    for i in range(len(attractions)):
        for j in range(i + 1, len(attractions)):
            a1 = attractions[i]
            a2 = attractions[j]
            dist = haversine(a1['lat'], a1['lon'], a2['lat'], a2['lon'])
            if dist < 10.0:
                G.add_edge(a1['name'], a2['name'], relation="nearby", distance=round(dist, 2))

    # 保存
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
    
    output_path = os.path.join(OUTPUT_ROOT, f"{city_name}attracions_kg.gml")
    nx.write_gml(G, output_path)
    print(f"  [完成] {city_name} 节点: {G.number_of_nodes()}, 边: {G.number_of_edges()}")

# ================= 主程序 =================

def main():
    # 检查 DATA_ROOT 是否存在
    if not os.path.exists(DATA_ROOT):
        # 尝试使用绝对路径或更精确的相对路径
        current_dir = os.getcwd()
        print(f"错误: 找不到数据目录 '{DATA_ROOT}'")
        print(f"当前脚本运行路径: {current_dir}")
        print(f"请检查路径 {os.path.join(current_dir, DATA_ROOT)} 是否存在")
        return

    # 获取所有城市文件夹
    cities = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    print(f"发现城市列表: {cities}")

    for city in cities:
        csv_path = os.path.join(DATA_ROOT, city, "attractions.csv")
        if os.path.exists(csv_path):
            build_city_kg(city, csv_path)
        else:
            print(f"跳过 {city}: 未找到 {csv_path}")

    print("\n所有任务完成！结果保存在: ", OUTPUT_ROOT)

if __name__ == "__main__":
    main()