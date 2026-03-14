### 清洗对话数据，生成可以直接用于LoRA训练格式的数据

import json
import os

# 配置输入输出路径
input_file = 'data/agent_dialogues_output.jsonl'
output_file = 'train_data/mindjourney_train_dialogues.json'

# 确保输出目录存在
os.makedirs('train_data', exist_ok=True)

if not os.path.exists(input_file):
    print(f"错误：找不到文件 {input_file}")
    exit()

with open(input_file, 'r', encoding='utf-8') as f:
    input_list = [json.loads(line) for line in f]

output_list = []

for item in input_list:
    # 仅清洗状态为成功的对话
    if item.get('state') is not True:
        continue
    
    dialogue_messages = []
    for message in item['messages']:
        train_msg = dict()
        
        if message['role'] == 'user':
            train_msg['role'] = 'user'
            # 移除内容前后的多余换行
            train_msg['content'] = message['content'].strip()
            
        elif message['role'] == 'assistant':
            train_msg['role'] = 'assistant'
            
            # 1. 组装思考过程 (Chain of Thought)
            thought_segments = []
            if 'intention_thought' in message and message['intention_thought']:
                thought_segments.append(f"【意图识别与状态更新】\n{message['intention_thought'].strip()}")
            if 'emotion_thought' in message and message['emotion_thought']:
                thought_segments.append(f"【用户情绪分析】\n{message['emotion_thought'].strip()}")
            if 'options_thought' in message and message['options_thought']:
                thought_segments.append(f"【策略与选项思考】\n{message['options_thought'].strip()}")
            
            # 拼接思考块
            thought_content = "\n\n".join(thought_segments)
            combined_content = f"<think>\n{thought_content}\n</think>"

            # 2. 组装最终输出 (普通回复 或 总结)
            # 如果是中间轮次，有 content
            if 'content' in message and message['content']:
                combined_content += f"\n\n<answer>{message['content'].strip()}</answer>"
            
            # 如果是最后一轮，包含意图总结 summary
            if 'summary' in message and message['summary']:
                # 将总结部分放入特定标签，便于模型学习规范化输出
                combined_content += f"\n\n<summary>{message['summary'].strip()}</summary>"
            
            train_msg['content'] = combined_content
        else:
            continue
            
        dialogue_messages.append(train_msg)
    
    # 只有当对话消息不为空时才添加
    if dialogue_messages:
        output_list.append({'messages': dialogue_messages})

print(f"数据清洗完成！")
print(f"原始数据: {len(input_list)} 条")
print(f"成功转换: {len(output_list)} 条")
print(f"训练文件已保存至: {output_file}")

# 保存为格式化好的 JSON
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(output_list, json_file, ensure_ascii=False, indent=4)