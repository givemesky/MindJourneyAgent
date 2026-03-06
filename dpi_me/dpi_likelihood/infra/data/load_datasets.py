import sys
import os
import json
import ast

# ===================================================
# 1. 路径修正逻辑
# ===================================================
# 当前路径: SynPT-main/dpi_me/dpi_likelihood/infra/data/load_datasets.py
# 向上跳 3 级到达 dpi_likelihood 目录
project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

# ===================================================
# 2. JSON 读写工具 (必须取消注释，供 nl2sl.py 调用)
# ===================================================

# 防御性引入 numpy，防止由于 Numpy 类型导致的序列化错误
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if HAS_NUMPY:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        return super(NpEncoder, self).default(obj)

def load_json_file(file_path):
    """通用 JSON 加载"""
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_file(json_data, file_path):
    """通用 JSON 保存"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf8") as dump_f:
        json.dump(json_data, dump_f, ensure_ascii=False, indent=4, cls=NpEncoder)

# ===================================================
# 3. 兼容性查询加载 (清理了 chinatravel 旧路径)
# ===================================================

def load_query_local(args, version="", verbose=False):
    """
    修改后的本地加载逻辑，指向 SynPT-main/data
    """
    query_data = {}
    # 向上回溯到 SynPT-main 根目录
    main_root = os.path.dirname(os.path.dirname(project_root_path))
    data_dir = os.path.join(main_root, "data")

    if not os.path.exists(data_dir):
        print(f"警告: 数据目录不存在: {data_dir}")
        return [], {}

    # 简化的加载逻辑
    files = [f for f in os.listdir(data_dir) if f.endswith('.json') or f.endswith('.jsonl')]
    for f in files:
        query_id = f.split(".")[0]
        # 这里仅作占位，实际逻辑可根据毕设需求扩展
        pass
    
    return [], {}

def load_query(args):
    """
    该函数原本从 HuggingFace 加载，毕设建议优先使用本地数据
    """
    print(">>> 正在调用 load_query 加载数据集...")
    # 如果你的毕设不需要在线加载 LAMDA 数据库，可以留空或指向本地
    return [], {}

if __name__ == "__main__":
    print(f"数据加载工具已就绪。Root: {project_root_path}")