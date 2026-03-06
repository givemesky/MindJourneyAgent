import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 设置路径
base_model_path = "/public/home/azsy/SynPT-main/model/Qwen2.5-7B-Instruct"
lora_path = "/public/home/azsy/SynPT-main/lora" 

# 2. 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 3. 加载 LoRA 权重
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

# 4. 进入循环对话模式
history = []  # 用于存储对话上下文
print("系统：Qwen-PT 已就绪！输入 'exit' 退出对话，输入 'clear' 清空上下文。")

while True:
    user_input = input("\n👤 用户：")
    
    if user_input.lower() == 'exit':
        break
    if user_input.lower() == 'clear':
        history = []
        print("系统：对话历史已清空。")
        continue

    # 构建对话模板 (Qwen2.5 格式)
    messages = history + [{"role": "user", "content": user_input}]
    
    # 使用 tokenizer 的 apply_chat_template 处理格式
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    # 生成回复
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        # 仅截取新生成的 tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"\n🤖 助手：{response}")

    # 将本次对话记录到历史中
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})