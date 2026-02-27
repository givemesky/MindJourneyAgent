import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 配置路径 ---
base_model_path = "/public/home/azsy/SynPT-main/model/Qwen2.5-7B-Instruct"
lora_path = "/public/home/azsy/SynPT-main/lora" 

# --- 加载 ---
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

# --- 初始化对话历史 ---
# System Prompt 很重要，它能维持模型的“专家导游”人设
messages = [
    {"role": "system", "content": "你是一个专业的旅游助手，擅长通过提问挖掘用户潜在的旅游意图。"}
]

print("\n🌟 Qwen-PT 已就绪！输入 'quit' 退出对话。")

while True:
    user_input = input("\n用户: ")
    if user_input.lower() in ['quit', 'exit', '退出']:
        break
    
    # 1. 将用户输入加入历史
    messages.append({"role": "user", "content": user_input})
    
    # 2. 转换格式并生成
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
        # 只取新生成的内容
        new_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = tokenizer.decode(new_ids, skip_special_tokens=True)
    
    # 3. 将模型回复加入历史，实现“记忆”
    print(f"\n助手: {response}")
    messages.append({"role": "assistant", "content": response})