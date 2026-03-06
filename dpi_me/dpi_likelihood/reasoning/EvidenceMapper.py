# ===========================================================================================
# 获取似然向量的测试脚本 - 集成 EvidenceMapper.py
# mapper.py升级版本，使用了正则捕获组（Capture Groups），不仅能匹配关键词，还能识别数值。
# ===========================================================================================


import re
from .schema import PREFERENCE_SCHEMA

class EvidenceMapper:
    def __init__(self, schema=None):
        self.schema = schema if schema else PREFERENCE_SCHEMA
        
        # 扩展语义词典：直接映射关键词到标签
        self.keyword_map = {
            "Consumption": {
                "Budget": ["吃紧", "便宜", "省钱", "低价", "性价比", "不贵", "穷游", "200元", "三百元"],
                "Economy": ["中等", "经济", "普通", "适中", "标准"],
                "Luxury": ["高端", "豪华", "五星", "不差钱", "奢侈", "高档", "档次"]
            },
            "Interest": {
                "Culture": ["人文", "文化", "历史", "古", "艺术", "建筑", "博物馆", "遗址", "人文景观", "古迹"],
                "Nature": ["自然", "公园", "森林", "山", "湖", "岛", "湿地", "风景", "草", "郊外", "自然风光"],
                "Entertainment": ["乐园", "迪士尼", "娱乐", "游乐", "欢乐谷", "演出", "主题", "体育"],
                "Shopping": ["商业", "商圈", "购物", "步行街", "广场", "中心", "免税", "买买买"]
            },
            "Pace": {
                "Relaxed": ["不赶", "慢慢", "悠闲", "放松", "慢节奏", "一两个", "不要太累", "随性", "休闲", "慢一点"],
                "Fast": ["赶时间", "多去", "多", "效率", "紧凑", "打卡", "尽量多", "快节奏", "多逛", "多看几个"]
            }
        }

    def _handle_interest(self, match):
        captured_val = match.group(1)
        # 只要引号内的内容命中任何维度的关键词，就返回结果
        for label, keywords in self.keyword_map["Interest"].items():
            if any(kw in captured_val for kw in keywords):
                return "Interest", label, 0.95
        return None

    def _handle_consumption(self, match):
        kw = match.group(1).lower()
        op = match.group(2)
        val = float(match.group(3))

        # 针对不同场景的数值逻辑
        if "accommodation" in kw or "住" in kw:
            if val <= 300: return "Consumption", "Budget", 0.95
        if "budget" in kw or "cost" in kw:
            if val <= 1000: return "Consumption", "Budget", 0.95
            if val <= 3000: return "Consumption", "Economy", 0.90
        return "Consumption", "Luxury", 0.95


    # def get_likelihood_D(self, constraints, raw_text=""):
    #     """
    #     constraints: NL2SL提取的约束列表
    #     raw_text: 原始用户输入（可选，用于增加兜底命中率）
    #     """
    #     D = {dim: [0.5] * len(labels) for dim, labels in self.schema.items()}
        
    #     # 将所有约束文本合并，并加入原始文本
    #     all_text_to_scan = [str(c) for c in constraints]
    #     if raw_text:
    #         all_text_to_scan.append(raw_text)

    #     for text in all_text_to_scan:
    #         t_low = text.lower()
            
    #         # --- 1. 策略 A：关键词全扫描（解决软意图命中率低的问题） ---
    #         for dim, label_dict in self.keyword_map.items():
    #             for label, keywords in label_dict.items():
    #                 for kw in keywords:
    #                     if kw in t_low:
    #                         # 简单的否定词处理：比如“不”+“赶”
    #                         if kw == "赶" and "不" in t_low:
    #                             continue # 交给 Relaxed 逻辑处理
                            
    #                         idx = self.schema[dim].index(label)
    #                         D[dim][idx] = max(D[dim][idx], 0.92)
    #                         print(f"DEBUG: [语义命中] {dim}->{label} | 匹配词: {kw} | 来源: {text}")

    #         # --- 2. 策略 B：结构化正则捕获（保留对数值和符号的精准处理） ---
    #         # 消费正则
    #         price_m = re.search(r"(price|budget|cost|accommodation).*?([<>!=]=?)\s*(\d+)", t_low)
    #         if price_m:
    #             res = self._handle_consumption(price_m)
    #             if res:
    #                 dim, label, score = res
    #                 D[dim][self.schema[dim].index(label)] = max(D[dim][self.schema[dim].index(label)], score)
    #                 print(f"DEBUG: [数值命中] {dim}->{label} | 内容: {price_m.group(0)}")

    #         # 兴趣正则 (Position/Type)
    #         interest_matches = re.finditer(r"(?:position|type).*?\"(.+?)\"", t_low)
    #         for m in interest_matches:
    #             res = self._handle_interest(m)
    #             if res:
    #                 dim, label, score = res
    #                 D[dim][self.schema[dim].index(label)] = max(D[dim][self.schema[dim].index(label)], score)
    #                 print(f"DEBUG: [符号命中] {dim}->{label} | 内容: {m.group(0)}")

    #     return D

    def get_likelihood_D(self, constraints, raw_text=""):
        """
        获取似然向量 D
        constraints: NL2SL 提取的符号约束列表
        raw_text: 原始用户输入文本（强烈建议传入，用于兜底识别软偏好）
        """
        # 1. 初始化 D 向量，默认 0.5 (贝叶斯中性)
        D = {dim: [0.5] * len(labels) for dim, labels in self.schema.items()}
        
        # 2. 构造扫描池：逻辑约束 + 原始自然语言
        scan_pool = []
        for c in constraints:
            scan_pool.append({"text": str(c).lower(), "is_raw": False})
        if raw_text:
            scan_pool.append({"text": raw_text.lower(), "is_raw": True})

        for item in scan_pool:
            text = item["text"]
            is_raw = item["is_raw"]
            
            # --- 策略 A: 噪音过滤 ---
            # 如果一条逻辑约束里包含过多的引号或中括号，通常是模型在“背诵”Schema 候选列表，而非真实意图
            # 例如: Type in ["自然风光", "历史古迹", "人文景观", ...]
            if not is_raw:
                quote_count = len(re.findall(r'["\']', text))
                if quote_count > 6 or len(text) > 150:
                    print(f"DEBUG: [噪音过滤] 忽略疑似列表倾倒的约束: {text[:50]}...")
                    continue

            # --- 策略 B: 语义关键词扫描 ---
            for dim, label_dict in self.keyword_map.items():
                if dim not in self.schema:
                    continue
                
                # 记录本条文本在该维度下命中的标签，用于冲突检测
                hits_in_this_line = set()
                
                for label, keywords in label_dict.items():
                    if label not in self.schema[dim]:
                        continue
                    
                    for kw in keywords:
                        if kw in text:
                            # 特殊否定逻辑处理：如“不”+“赶”
                            if kw == "赶" and "不" in text:
                                if "Relaxed" in self.schema["Pace"]:
                                    idx = self.schema["Pace"].index("Relaxed")
                                    D["Pace"][idx] = max(D["Pace"][idx], 0.95)
                                    print(f"DEBUG: [语义命中] Pace->Relaxed | 匹配词: 不赶")
                                continue
                            
                            hits_in_this_line.add(label)
                            break # 一个标签命中一个词即可
                
                # 冲突过滤：如果同一行文本触发了一个维度下的 3 个以上标签，通常是无效证据（如背诵列表）
                if len(hits_in_this_line) >= 3 and not is_raw:
                    print(f"DEBUG: [冲突过滤] 维度 {dim} 命中过多标签 {hits_in_this_line}，已忽略。")
                    continue
                
                # 正式更新 D 向量
                for label in hits_in_this_line:
                    idx = self.schema[dim].index(label)
                    # 设定似然概率：设定为 0.95 以上可让贝叶斯置信度跳变更明显
                    D[dim][idx] = max(D[dim][idx], 0.95)
                    source_type = "原始文本" if is_raw else "逻辑约束"
                    print(f"DEBUG: [语义命中] {dim}->{label} | 匹配词来源: {source_type}")

            # --- 策略 C: 结构化正则捕获 (保持对数值的精准处理) ---
            # 1. 消费数字正则
            price_m = re.search(r"(price|budget|cost|accommodation).*?([<>!=]=?)\s*(\d+)", text)
            if price_m:
                res = self._handle_consumption(price_m)
                if res:
                    dim, label, score = res
                    if dim in self.schema and label in self.schema[dim]:
                        idx = self.schema[dim].index(label)
                        D[dim][idx] = max(D[dim][idx], score)
                        print(f"DEBUG: [数值命中] {dim}->{label} | 内容: {price_m.group(0)}")

            # 2. 兴趣符号正则 (Position/Type)
            interest_matches = re.finditer(r"(?:position|type).*?\"(.+?)\"", text)
            for m in interest_matches:
                res = self._handle_interest(m)
                if res:
                    dim, label, score = res
                    if dim in self.schema and label in self.schema[dim]:
                        idx = self.schema[dim].index(label)
                        D[dim][idx] = max(D[dim][idx], score)
                        print(f"DEBUG: [符号命中] {dim}->{label} | 内容: {m.group(0)}")

        return D

#################v2#####################################
# # EvidenceMapper.py 基本判定可以实现，节奏和价格识别的不是很好
# import re
# from .schema import PREFERENCE_SCHEMA

# class EvidenceMapper:
#     def __init__(self, schema=None):
#         self.schema = schema if schema else PREFERENCE_SCHEMA
        
#         # 语义特征词典：保持核心词根
#         self.interest_features = {
#             "Culture": ["博物馆", "古迹", "建筑", "寺", "文化", "历史", "人文", "艺术", "古镇", "景观", "展览"],
#             "Nature": ["自然", "公园", "森林", "山", "湖", "岛", "湿地", "风景", "草"],
#             "Entertainment": ["乐园", "迪士尼", "娱乐", "游乐", "欢乐谷", "演出", "主题"],
#             "Shopping": ["商业", "商圈", "购物", "步行街", "广场", "中心", "免税"]
#         }

#     def _handle_interest(self, match):
#         """
#         优化后的处理函数：不仅提取引号内容，还支持模糊词典匹配
#         """
#         # 此时 match.group(1) 是引号内的文本内容
#         captured_val = match.group(1)
        
#         for label, keywords in self.interest_features.items():
#             if any(kw in captured_val for kw in keywords):
#                 return "Interest", label, 0.95
#         return None

#     def _handle_consumption(self, match):
#         """
#         智能消费识别：区分单体价格和总预算
#         """
#         kw = match.group(1).lower() # 捕获到的关键词 (price/budget/cost)
#         op = match.group(2)         # 操作符
#         val = float(match.group(3)) # 数值

#         # 逻辑：如果是餐饮单价
#         if "restaurant" in kw or "price" in kw:
#             if val <= 200: return "Consumption", "Budget", 0.95
#             if val <= 500: return "Consumption", "Economy", 0.90
#             return "Consumption", "Luxury", 0.90
        
#         # 逻辑：如果是总预算 (Budget/Cost)
#         # 3天4000元，平均每天 > 1000，通常属于 Luxury 或高端 Economy
#         if "budget" in kw or "cost" in kw:
#             if val <= 1000: return "Consumption", "Budget", 0.95
#             if val <= 3000: return "Consumption", "Economy", 0.90
#             return "Consumption", "Luxury", 0.95 # 4000元属于 Luxury 范畴
            
#         return None

#     def get_likelihood_D(self, constraints):
#         D = {dim: [0.5] * len(labels) for dim, labels in self.schema.items()}
        
#         # 改进的正则规则
#         rules = [
#             # 1. 消费：捕获 关键词 + 操作符 + 数字
#             {"pattern": r"(price|budget|cost|activities\s+cost).*?([<>!=]=?)\s*(\d+)", "handler": self._handle_consumption},
            
#             # 2. 兴趣：【重大改进】只要引号里的内容出现在 Position 或 Type 所在的行，就提取出来
#             # 它能匹配 Type in ["A", "B"] 里的每个引号
#             {"pattern": r"(?:Position|Type).*?\"(.+?)\"", "handler": self._handle_interest},
            
#             # 3. 节奏：
#             # {"pattern": r"(?:time|duration).*?([<>!=]=?)\s*(\d+)", "handler": lambda m: ("Pace", "Relaxed" if float(m.group(2)) >= 100 else "Fast", 0.9)},
#             # {"pattern": r"(慢|悠闲|放松|不赶|slow)", "dim": "Pace", "label": "Relaxed", "score": 0.90},
#             # {"pattern": r"(快|赶|效率|紧凑|fast|多去)", "dim": "Pace", "label": "Fast", "score": 0.85}

#             # 识别“慢”、“不赶”等积极信号
#             {"pattern": r"(慢|悠闲|放松|不赶|slow|慢慢逛)", "dim": "Pace", "label": "Relaxed", "score": 0.95},
#             # 识别数量信号：例如“一两个地方”暗示节奏慢
#             {"pattern": r"([12一两]).*?(景点|地方|place)", "dim": "Pace", "label": "Relaxed", "score": 0.90},
#             # 识别“赶”、“快”信号
#             {"pattern": r"(快|赶|效率|紧凑|fast|多|打卡)", "dim": "Pace", "label": "Fast", "score": 0.90}
#         ]

#         for c in constraints:
#             c_str = str(c)
#             # 处理“不要太赶”：如果句子中包含“不”+“赶”，应视为 Relaxed 而不是 Fast
#             if "不" in c_str and "赶" in c_str:
#                 D["Pace"][self.schema["Pace"].index("Relaxed")] = max(D["Pace"][self.schema["Pace"].index("Relaxed")], 0.95)
#                 continue # 跳过后续正则，防止误触“赶”这个 Fast 关键词

#             for rule in rules:
#                 # 使用 finditer 处理一行中的多个匹配（如 Type in ["A", "B"]）
#                 matches = re.finditer(rule["pattern"], c_str, re.IGNORECASE)
#                 for m in matches:
#                     if "handler" in rule:
#                         res = rule["handler"](m)
#                         if not res: continue
#                         dim, label, score = res
#                     else:
#                         dim, label, score = rule["dim"], rule["label"], rule["score"]

#                     if dim in self.schema and label in self.schema[dim]:
#                         idx = self.schema[dim].index(label)
#                         D[dim][idx] = max(D[dim][idx], score)
#                         print(f"DEBUG: 命中! [{dim}] -> {label} (Score: {score}) | 内容: {m.group(0)}")

        
        
#         return D





#################v1#####################################
# import re
# from .schema import PREFERENCE_SCHEMA

# class EvidenceMapper:
#     def __init__(self):
#         # 定义规则库：{正则表达式: 处理逻辑}
#         # 这里的正则设计原则：抓取核心动词和名词，忽略修饰语
#         self.rules = [
#             # --- 1. 消费水平 (Consumption) ---
#             {
#                 "pattern": r"(budget|price|cost).*<=\s*(\d+)", 
#                 "handler": self._handle_consumption
#             },
#             {
#                 "pattern": r"(luxury|五星|高端|豪华)", 
#                 "dim": "Consumption", "label": "Luxury", "score": 0.9
#             },

#             # --- 2. 兴趣倾向 (Interest) ---
#             {
#                 "pattern": r"position\s*==\s*\"(外滩|博物馆|城隍庙|东方明珠)\"", 
#                 "dim": "Interest", "label": "Culture", "score": 0.85
#             },
#             {
#                 "pattern": r"position\s*==\s*\"(自然公园|森林|共青|植物园)\"", 
#                 "dim": "Interest", "label": "Nature", "score": 0.9
#             },

#             # --- 3. 出行节奏 (Pace) ---
#             {
#                 "pattern": r"(time|节奏).*(慢|悠闲|放松|慢悠悠)", 
#                 "dim": "Pace", "label": "Relaxed", "score": 0.95
#             },
#             {
#                 "pattern": r"(time|节奏).*(快|赶|紧|效率)", 
#                 "dim": "Pace", "label": "Fast", "score": 0.85
#             }
#         ]

#     def _handle_consumption(self, match):
#         """处理带数值的消费约束"""
#         # match.group(1) 是关键词 (budget/price/cost)
#         # match.group(2) 是提取出来的数字
#         value = float(match.group(2))
#         if value <= 300:
#             return "Consumption", "Budget", 0.95
#         elif value <= 800:
#             return "Consumption", "Economy", 0.80
#         else:
#             return "Consumption", "Luxury", 0.5 # 似然度保持中立或微调

#     def get_likelihood_D(self, constraints):
#         # 初始化 D 向量：全 0.5 (中性)
#         D = {dim: [0.5] * len(labels) for dim, labels in PREFERENCE_SCHEMA.items()}

#         for c in constraints:
#             c_str = str(c).lower()
#             for rule in self.rules:
#                 match = re.search(rule["pattern"], c_str)
#                 if match:
#                     # 分支处理：如果有 handler 走函数，否则取固定分值
#                     if "handler" in rule:
#                         res = rule["handler"](match)
#                         if not res: continue
#                         dim, label, score = res
#                     else:
#                         dim, label, score = rule["dim"], rule["label"], rule["score"]

#                     # 映射到 D 向量对应的索引
#                     labels = PREFERENCE_SCHEMA[dim]
#                     if label in labels:
#                         idx = labels.index(label)
#                         # 更新似然度：Max-Pooling 策略
#                         D[dim][idx] = max(D[dim][idx], score)
#         return D