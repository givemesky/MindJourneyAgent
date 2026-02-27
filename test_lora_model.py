import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 基础模型路径（你刚才测试成功的路径）
base_model_path = "/public/home/azsy/SynPT-main/model/Qwen2.5-7B-Instruct"
# 2. LoRA 权重路径（替换为你文件夹的实际名称）
lora_path = "/public/home/azsy/SynPT-main/lora" 

print("正在加载分词器和基础模型...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("正在挂载 LoRA 权重插件...")
# 关键一步：使用 PeftModel 将插件合体
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

print("✅ Qwen-PT 加载完成！正在测试主动挖掘意图的能力...")

# 3. 测试对话（使用论文中的场景）
prompt = "我想去北京玩。"
messages = [
    {"role": "system", "content": "你是一个专业的旅游助手，擅长通过提问挖掘用户潜在的旅游意图。"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**model_inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
    response = tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
    
    print("\n" + "="*30)
    print(f"用户输入: {prompt}")
    print(f"模型回复: \n{response}")
    print("="*30)