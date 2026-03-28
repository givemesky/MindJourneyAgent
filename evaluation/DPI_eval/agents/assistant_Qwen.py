# -*- coding: utf-8 -*-
import sys
import io
import os
import json
import time
import re
import torch

# 自动寻找根目录并加入 path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if root not in sys.path:
    sys.path.insert(0, root)

# 导入模型提供者
from model_provider import SharedModelProvider

# 强制设置标准输入输出为 UTF-8
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class AssistantQwen:
    def __init__(self, log_dir="evaluation/DPI_eval/logs_Qwen"):
        print("🤖 [Assistant Qwen] 正在初始化纯大模型引擎 (无贝叶斯优化)...")
        self.log_dir = log_dir
        # 获取共享模型实例
        self.model_res = SharedModelProvider.get_instance()
        self.model = self.model_res.model
        self.tokenizer = self.model_res.tokenizer
        
        # 内部状态记录，用于生成最终日志
        self.history_evolution = {}
        self.captured_constraints = []

    def _get_system_prompt(self):
        return """你是一名专业的旅游规划助手。你的目标是通过对话了解用户的旅游偏好，以便最后提供定制方案。
你必须关注以下三个维度的标签：
1. 消费偏好 (Consumption): [Budget, Economy, Luxury]
2. 兴趣特征 (Interest): [Culture, Nature, Entertainment, Shopping]
3. 行程节奏 (Pace): [Fast, Relaxed]

【重要要求】：
在你的每一轮回复之后，你必须另起一行，输出一个固定的 JSON 格式块，用于记录你当前对用户画像的推断。
格式如下（必须包含在 <ANALYSIS> 标签内）：
<ANALYSIS>
{
  "persona": {
    "Consumption": "标签名或Unknown",
    "Interest": "标签名或Unknown",
    "Pace": "标签名或Unknown"
  },
  "constraints": ["提取到的硬约束1", "提取到的硬约束2"],
  "terminate": true/false (如果你觉得已经完全掌握了以上三个维度以及目的地、人数、天数，请设为 true，否则为 false)
}
</ANALYSIS>

请记住：你的回复文本要亲切自然，不要在回复文本中提及具体的标签名，只需在 <ANALYSIS> 块中记录。"""

    def process_turn(self, user_input, turn, chat_history_all):
        """
        处理一轮对话
        :param chat_history_all: 包含系统提示词和之前所有对话的列表
        """
        # 1. 构造发送给 Qwen 的消息流
        messages = [{"role": "system", "content": self._get_system_prompt()}]
        messages.extend(chat_history_all)

        # 2. 编码输入并获取 attention_mask
        model_inputs = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt",
            return_dict=True 
        ).to(self.model.device)

        # 3. 调用模型生成
        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
                top_p=0.8,
                top_k=20,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
            
            # 提取生成的回复（切除 Prompt 部分）
            response_text = self.tokenizer.decode(
                output_ids[0][len(model_inputs['input_ids'][0]):], 
                skip_special_tokens=True
            )

        # 4. 解析回复与分析块
        # 提取回复文本部分（去掉 <ANALYSIS> 块）
        reply_part = re.sub(r'<ANALYSIS>.*?</ANALYSIS>', '', response_text, flags=re.DOTALL).strip()
        # 提取 JSON 分析块
        analysis_match = re.search(r'<ANALYSIS>(.*?)</ANALYSIS>', response_text, flags=re.DOTALL)
        
        analysis_data = {
            "persona": {"Consumption": "Unknown", "Interest": "Unknown", "Pace": "Unknown"},
            "constraints": [],
            "terminate": False
        }
        
        if analysis_match:
            try:
                # 尝试解析 JSON
                analysis_data = json.loads(analysis_match.group(1).strip())
            except:
                print(f"⚠️ [Assistant Qwen] JSON 解析失败，原始输出: {analysis_match.group(1)}")

        # 5. 封装画像结果
        persona_result = {}
        for dim in ["Consumption", "Interest", "Pace"]:
            label = analysis_data.get("persona", {}).get(dim, "Unknown")
            persona_result[dim] = {
                "label": label,
                "confidence": 1.0 if label != "Unknown" else 0.33,
                "is_stable": True if label != "Unknown" else False
            }

        # 记录演化轨迹
        self.history_evolution[str(turn)] = persona_result
        self.captured_constraints = analysis_data.get("constraints", [])

        # 【核心修复】：确保函数返回字典
        return {
            "reply": reply_part if reply_part else "好的，我明白了。还有什么需要补充的吗？",
            "status": "TERMINATE" if analysis_data.get("terminate") else "ASK",
            "persona": persona_result,
            "metrics": {}, 
            "reason": "LLM Self-Decision"
        }

    def terminate_session(self, case_uid):
        """
        保存对照组日志
        """
        case_log_path = os.path.join(self.log_dir, case_uid)
        os.makedirs(case_log_path, exist_ok=True)

        # 1. 保存画像演化记录
        with open(os.path.join(case_log_path, "persona_evolution.json"), "w", encoding="utf-8") as f:
            json.dump(self.history_evolution, f, ensure_ascii=False, indent=2)

        # 2. 保存约束记录
        dpi_constraints = []
        for i, c in enumerate(self.captured_constraints):
            dpi_constraints.append(f"{i+1}. {c} == \"Captured\"")
        
        with open(os.path.join(case_log_path, "constraints_log.json"), "w", encoding="utf-8") as f:
            json.dump([{"turn": 99, "constraints": dpi_constraints}], f, ensure_ascii=False, indent=2)

        # 3. 模拟空的收敛 CSV
        with open(os.path.join(case_log_path, "persona_convergence.csv"), "w", encoding="utf-8") as f:
            f.write("turn,Consumption_entropy,Interest_entropy,Pace_entropy\n")
            for t in self.history_evolution.keys():
                f.write(f"{t},0,0,0\n")

        print(f"✅ [Assistant Qwen] 对照组日志已保存。")