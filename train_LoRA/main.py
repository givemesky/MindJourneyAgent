###通过调用api合成对话数据，用于LoRA训练

import argparse
import asyncio
import json
import random
import os
from typing import Dict, Any

import yaml
from tqdm import tqdm
from openai import OpenAI

# 导入 Prompts
from prompts import first_inquiry_system_prompt, user_prompt, user_prompt_anxiety
# 导入 Utils 工具函数
from utils import (
    get_simple_messages,
    from_message_to_text,
    call_openai_api_token_usage_async,
    count_tokens,
    get_info_from_config,              
    get_random_intention_from_config_prob  
)
from user import User
from assistant_agent import AssistantAgent

# 配置参数解析
parser = argparse.ArgumentParser(description="Run dialogue simulation with LLMs.")
parser.add_argument('--input_file', type=str, default='data/seed_data_with_pref.jsonl', help='Seed data file (jsonl)')
parser.add_argument('--config_file', type=str, default='config.yaml', help='Path to config.yaml')
parser.add_argument('--output_file', type=str, help='Output file path (jsonl)')
parser.add_argument('--batch_size', type=int,  default=1, help='Number of items per batch')

# 模型相关参数
parser.add_argument('--user_model_id', type=str, default='gemma-3-27b-it', help='User model ID')
parser.add_argument('--assistant_model_id', type=str,  default='gemma-3-27b-it', help='Assistant model ID')

# API Key 与 URL 参数
parser.add_argument('--user_api_key', type=str, default='AIzaSyDgpc-WpDzv24ZEPINcLdZtgyEZeuZPb4E', help='API key for user model')
parser.add_argument('--assistant_api_key', type=str, default='AIzaSyCfkpYBCb_EkZK9ufX3wvg3wI3mWiSnP8Q', help='API key for assistant model')
parser.add_argument('--user_base_url', type=str, default="https://generativelanguage.googleapis.com/v1beta/openai/", help='Base URL for user model')
parser.add_argument('--assistant_base_url', type=str, default="https://generativelanguage.googleapis.com/v1beta/openai/", help='Base URL for assistant model')


async def process_dialogue(item: Dict[str, Any], output_file: str, config: Dict[str, Any],
                           user_client, assistant_client,
                           user_model_id: str, assistant_model_id: str) -> None:
    """处理单条对话生成的协程"""
    
    # 1. 种子和 Token 统计初始化
    random.seed(int(item['id']))
    token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

    # 2. 从 Config 获取全量任务信息
    task_name, intention_types, full_intention_info, max_turns_limit = get_info_from_config(config['task'], item)

    # 3. 概率采样：生成首轮对话的“初始背景”
    sampled_intent_info, _ = get_random_intention_from_config_prob(config['task'], item)

    # 4. 生成首轮用户咨询 (First Inquiry)
    try:
        sys_prompt_first = first_inquiry_system_prompt.format(task_name=task_name)
    except KeyError as e:
        print(f"\n[错误] prompts.py 中的 first_inquiry_system_prompt 包含了多余的占位符 {e}")
        return

    user_first_content = f"以下是你的旅行背景信息，请据此向助手发起首轮咨询：\n{sampled_intent_info}"
    messages = get_simple_messages(sys_prompt_first, user_first_content)
    first_inquiry, usage = await call_openai_api_token_usage_async(user_client, user_model_id, messages)
    count_tokens(token_usage, usage)

    if not first_inquiry or "Error" in first_inquiry:
        print(f"ID {item['id']} 首轮咨询生成失败: {first_inquiry}")
        return

    # 5. 初始化两个 Agent 的配置
    # 无论是否焦虑，初始化时【统一使用冷静版 user_prompt】
    # 这样可以确保用户在前几轮表现得像个正常人
    anxiety_switch = random.random() < 0.33
    
    try:
        # 初始初始化强制使用 user_prompt (冷静版)
        user_setup_prompt = user_prompt.format(
            task_name=task_name,
            intention_info=full_intention_info
        )
    except KeyError as e:
        print(f"\n[错误] prompts.py 中的 user_prompt 模板占位符错误: {e}")
        return

    # 初始化 User Agent
    user_agent = User('user', user_setup_prompt, user_model_id, 0.01, user_client)
    # 初始化 Assistant Agent
    assistant_agent = AssistantAgent(assistant_client, assistant_model_id, task_name, intention_types)

    # --- 6. 进入对话循环 ---
    last_user_response = first_inquiry
    current_intention_state = None 

    for i in range(max_turns_limit):
        # 助手处理用户的回答
        response_dict, current_intention_state, usage = await assistant_agent.chat(last_user_response, current_intention_state)
        count_tokens(token_usage, usage)

        if response_dict is None:
            break

        # 检查是否满足结束条件 (收集全或由于焦虑触发 summary)
        if 'summary' in response_dict or '<summary>' in str(response_dict):
            save_result(output_file, item, assistant_agent.dialog_history_details, token_usage, completed=True, anxiety=anxiety_switch)
            break

        if i == max_turns_limit - 1:
            save_result(output_file, item, assistant_agent.dialog_history_details, token_usage, completed=False, anxiety=anxiety_switch)
            break

        # --- 核心修改：动态性格转变逻辑 ---
        # i=0 代表助手的第1次追问，i=1 代表助手的第2次追问，i=2 代表助手的第3次追问
        # 我们设定在助手第3次追问（i >= 2）之后，用户开始变焦急
        anxiety_instruction = ""
        if anxiety_switch and i >= 2:
            anxiety_instruction = (
                "\n\n【系统提醒：你现在已经失去了耐心，感到非常焦急和烦躁！"
                "请在回复中表现出明显的不耐烦，催促助手直接给方案，拒绝再回答任何琐碎的问题。】"
            )

        # 用户 Agent 根据历史对话回复助手
        history_text = from_message_to_text(assistant_agent.dialog_history)
        user_reply_input = (
            f'```历史对话：\n{history_text}```\n'
            f'根据历史对话、你的意图背景以及角色要求，请回答助手的最后提问。{anxiety_instruction}'
        )
        
        last_user_response, usage = await user_agent.chat(user_reply_input)
        count_tokens(token_usage, usage)

def save_result(output_file, item, messages, token_usage, completed, anxiety):
    """保存生成的结果到 JSONL"""
    result = {
        'id': item['id'],
        'category': item.get('category'),
        'state': completed,
        'anxiety': anxiety,
        'destination': item.get('destination'),
        'messages': messages,  # 这里保存的是带有思考过程的详细历史
        'usage': token_usage
    }
    with open(output_file, 'a+', encoding='utf-8') as file:
        file.write(json.dumps(result, ensure_ascii=False) + '\n')


async def main():
    args = parser.parse_args()
    if not args.output_file:
        args.output_file = f"data/agent_dialogues_output.jsonl"

    # 初始化 OpenAI 客户端
    user_client = OpenAI(api_key=args.user_api_key, base_url=args.user_base_url)
    assistant_client = OpenAI(api_key=args.assistant_api_key, base_url=args.assistant_base_url)

    # 加载种子数据
    if not os.path.exists(args.input_file):
        print(f"Error: 找不到输入文件 {args.input_file}")
        return
        
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_list = [json.loads(line) for line in f]

    # 加载任务配置 (YAML)
    if not os.path.exists(args.config_file):
        print(f"Error: 找不到配置文件 {args.config_file}")
        return

    with open(args.config_file, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    print(f"开始生成对话，共 {len(input_list)} 条种子记录...")

    # 按照 Batch Size 批量处理
    for i in tqdm(range(0, len(input_list), args.batch_size)):
        batch = input_list[i:i + args.batch_size]
        tasks = [
            process_dialogue(
                item, args.output_file, config_data, 
                user_client, assistant_client, 
                args.user_model_id, args.assistant_model_id
            ) for item in batch
        ]
        # 并发执行 Batch 中的任务
        await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())