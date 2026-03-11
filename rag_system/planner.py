import json
import os
import networkx as nx
import numpy as np
from scipy.cluster.vq import kmeans2
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 配置路径 =================
MODEL_PATH = "/public/home/azsy/SynPT-main/model/Qwen2.5-7B-Instruct"

class TravelPlanner:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.user_data = json.load(f)
        
        city_mapping = {"北京": "beijing", "成都": "chengdu", "深圳": "shenzhen", "上海": "shanghai", "苏州": "suzhou"}
        target_city_name = self.user_data['basic_info']['target_city']
        self.city_key = city_mapping.get(target_city_name, "beijing")
        
        # 加载图谱
        self.kg = self._load_kg()
        
        # 直接在此处初始化原生模型 (不加载 LoRA)
        print(f"正在直接加载基座模型: {MODEL_PATH}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16, # 使用 bfloat16 节省显存
            device_map="auto",          # 自动平衡显存
            trust_remote_code=True
        )
        self.model.eval()

    def _load_kg(self):
        kg_path = f"/public/home/azsy/SynPT-main/kg_travel/city_kg/output_city_kg/{self.city_key}_unified_kg.gml"
        return nx.read_gml(kg_path)

    # ... (中间的 _clean_name, _get_node, _get_nearby_candidates, solve_plan 函数逻辑保持不变) ...

    def run(self):
        print(f"正在分析 {self.user_data['basic_info']['target_city']} 统一知识图谱数据...")
        loyalty_plan = self.solve_plan(strategy="loyalty")
        budget_plan = self.solve_plan(strategy="budget")

        prompt = f"""
        你是一名专业的资深导游。请根据以下两个结构化排期方案，为用户写一段生动的对比推荐方案。
        
        用户偏好：消费偏好({self.user_data['soft_constraints_persona']['consumption_level']})，兴趣倾向({self.user_data['soft_constraints_persona']['interest_preference']})。
        
        [方案一：{loyalty_plan['strategy_name']}]
        {json.dumps(loyalty_plan, ensure_ascii=False, indent=2)}
        
        [方案二：{budget_plan['strategy_name']}]
        {json.dumps(budget_plan, ensure_ascii=False, indent=2)}
        
        要求：
        1. 语气热情且专业。
        2. 方案一保留了用户选的所有店，但需提醒某些店开销较高，可能超出 Budget 预算。
        3. 方案二推荐了平价美食，指出这更符合用户经济实惠的需求。
        4. 描述每天的亮点。
        5. 最后给出各方案预计总花销。
        6. 不要提及任何交通工具。
        """
        
        # 直接使用原生模型的推理逻辑
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        print("原生 Qwen 大模型正在生成最终对比报告...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            new_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            return self.tokenizer.decode(new_ids, skip_special_tokens=True)

if __name__ == "__main__":
    planner = TravelPlanner("/public/home/azsy/SynPT-main/test_json.json")
    print("\n" + "="*60)
    print(planner.run())
    print("="*60)