# 配置文件：定义所有日志和图谱的绝对路径

import os

# 日志路径
LOG_DIR = "dpi_me/logs"
CONSTRAINTS_PATH = os.path.join(LOG_DIR, "UserProfile/constraints_log.json")
CONVERGENCE_PATH = os.path.join(LOG_DIR, "Convergence/persona_convergence.csv")

# 图谱数据路径
KG_DIR = "/public/home/azsy/MindJourney_LoRA/MindJourneyAgent/RAG/kg_travel/attracions_restaurants_kg/output_attracions_restaurants_kg"

# 输出设置
OUTPUT_FIELDS_ATTR = ["name", "type", "price", "max_time"]
OUTPUT_FIELDS_RES = ["name", "cuisine", "recommend"]