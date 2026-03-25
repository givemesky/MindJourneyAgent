import openai # 假设你使用的是兼容 OpenAI 格式的 API 转发

def call_gemma_api(model, system_prompt, messages):
    client = openai.OpenAI(api_key="AIzaSyCfU8v-9LTXMT0-xWtGf503GaudDh55FDs", base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    
    # Gemma-3 兼容性处理：将 System Prompt 合并到第一条 User 消息中，或者放在 messages 首位
    # 根据你之前的经验，这里采用合并方式
    formatted_messages = []
    
    # 注入系统提示词
    # 如果模型不支持 system role，将其转为 user
    combined_first_msg = f"【System Instructions】\n{system_prompt}\n\n【Conversation Start】\n"
    
    if messages:
        # 如果已有历史，将系统提示词拼在第一条消息前
        first_msg = messages[0]
        formatted_messages = [{"role": "user", "content": combined_first_msg + first_msg['content']}]
        formatted_messages.extend(messages[1:])
    else:
        # 如果是首轮
        formatted_messages = [{"role": "user", "content": combined_first_msg + "你好"}]

    completion = client.chat.completions.create(
        model=model,
        messages=formatted_messages,
        temperature=0.7
    )
    return completion.choices[0].message.content