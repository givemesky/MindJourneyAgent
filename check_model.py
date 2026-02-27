import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 注意：这里填写你下载后的实际路径
# 如果你 cache_dir='./models'，路径通常如下：
model_path = "/public/home/azsy/SynPT-main/model/Qwen2.5-7B-Instruct"

print("正在检查路径是否存在...")
import os
if not os.path.exists(model_path):
    print(f"错误：路径 {model_path} 不存在！请检查下载位置。")
else:
    try:
        print("正在加载分词器 (Tokenizer)...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        print("正在加载模型 (Model) 到 GPU...")
        # device_map="auto" 会自动把模型加载到显卡上
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("\n✅ 恭喜！模型加载成功！")
        
        # 简单测试一下对话
        prompt = "你好，请介绍一下你自己。"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        print("正在生成测试回复...")
        generated_ids = model.generate(**model_inputs, max_new_tokens=50)
        response = tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
        
        print(f"\n模型回复：{response}")
        
    except Exception as e:
        print(f"\n❌ 加载失败，错误原因：\n{e}")