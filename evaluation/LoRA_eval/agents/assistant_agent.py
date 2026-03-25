import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class MindJourneyAssistant:
    def __init__(
        self, 
        mode="lora",  # "base", "few-shot", "lora"
        base_model_path="/public/home/azsy/MindJourney_LoRA/MindJourneyAgent/model/Qwen2.5-7B-Instruct",
        lora_path="/public/home/azsy/MindJourney_LoRA/MindJourneyAgent/lora/mindjourney_v1",
        device="cuda"
    ):
        self.mode = mode
        self.device = device
        
        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
        # 2. 加载基座模型
        print(f"[{mode.upper()} 模式] 正在加载基座模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 3. 如果是 LoRA 模式，挂载权重
        if mode == "lora":
            print(f"正在加载 LoRA 权重: {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        
        self.model.eval()

        # 4. 定义系统提示词 (通用)
        self.system_prompt = (
            "你是一个具备“主动意图挖掘”能力的文旅对话智能体。你的任务是通过主动追问，"
            "收集并识别用户的 4 个核心维度信息，并将其映射到以下标准标签中：\n\n"
            "### 1. 标准标签定义 (PREFERENCE_SCHEMA):\n"
            "- **目的地 (Destination)**: 用户想去的具体城市（如：北京）。\n"
            "- **消费偏好 (Consumption)**: [Budget (普通), Economy (经济型/穷游), Luxury (豪华型/高端)]\n"
            "- **兴趣特征 (Interest)**: [Culture (历史文化), Nature (自然风光), Entertainment (现代娱乐/酒店), Shopping (购物中心)]\n"
            "- **行程节奏 (Pace)**: [Fast (特种兵/紧凑), Relaxed (慢节奏/休闲)]\n\n"
            "### 2. 输出要求:\n"
            "每一轮回复必须严格遵守以下三段式格式：\n"
            "<think>分析当前已掌握信息，判定还缺失哪些维度，决定下一步追问策略</think>\n"
            "<answer>对用户的自然语言回复或追问</answer>\n"
            "<summary>{\"Destination\": \"标签或null\", \"Consumption\": \"标签或null\", \"Interest\": \"标签或null\", \"Pace\": \"标签或null\"}</summary>\n\n"
            "请注意：在未确定某个维度前，该字段必须填 null。一旦识别，请填写上述方括号内的标准英文标签。"
        )

        # 5. 定义 Few-shot 示例 (仅用于 Group B)
        self.few_shot_examples = [
            {"role": "user", "content": "你好，我想去北京旅游。"},
            {"role": "assistant", "content": "<think>用户明确了目的地是北京。接下来需要挖掘消费偏好、兴趣和节奏。我先从消费偏好开始问起。</think><answer>北京欢迎您！为了能给您更精准的建议，我想先了解一下您的预算情况，您是倾向于经济实惠的穷游，还是舒适高端的品质游呢？</answer><summary>{\"Destination\": \"北京\", \"Consumption\": null, \"Interest\": null, \"Pace\": null}</summary>"},
            {"role": "user", "content": "我想省点钱，吃点地道的就行。"},
            {"role": "assistant", "content": "<think>用户选择了经济型消费(Economy)。现在还差兴趣和节奏。用户提到了吃地道的，可能对文化美食感兴趣。</think><answer>收到！北京有很多物美价廉的老字号。那您平时喜欢什么样的景点？是喜欢故宫这种人文古迹，还是喜欢去郊外看自然风光？</answer><summary>{\"Destination\": \"北京\", \"Consumption\": \"Economy\", \"Interest\": \"Culture\", \"Pace\": null}</summary>"}
        ]

    def generate_response(self, conversation_history):
        # 构造输入消息序列
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 如果是 Few-shot 模式，在历史之前插入示例
        if self.mode == "few-shot":
            # 将示例插入 system 之后，当前对话之前
            messages.extend(self.few_shot_examples)
            
        messages.extend(conversation_history)
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response_ids = output_ids[0][len(input_ids[0]):]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)