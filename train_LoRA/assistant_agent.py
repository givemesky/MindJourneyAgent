# --- START OF FILE assistant_agent.py ---
import re
from prompts import (
    assistant_prompt_intention, assistant_prompt_summary, 
    assistant_prompt_emotion, assistant_prompt_options, 
    assistant_prompt_response
)
from utils import get_simple_messages, call_openai_api_token_usage_async, from_message_to_text_current_turn, count_tokens

class AssistantAgent:
    def __init__(self, client, model, task_name, intention_types):
        self.client = client
        self.model = model
        self.task_name = task_name
        self.intention_types = intention_types # 这里的列表应该是 ["目的地", "消费水平", "兴趣特征", "行程节奏"]
        self.dialog_history = []         # 简化的对话历史（用于传给 LLM）
        self.dialog_history_details = [] # 详细的历史记录（包含思考过程，用于保存结果）

    def update_history(self, user_message, model_response):
        if user_message:
            self.dialog_history.append({"role": "user", "content": user_message})
        if model_response:
            self.dialog_history.append({"role": "assistant", "content": model_response})
    
    def update_history_details(self, user_message, assistant_step_dict):
        if user_message:
            self.dialog_history_details.append({"role": "user", "content": user_message})
        if assistant_step_dict:
            # assistant_step_dict 包含了各步思考和最终 content
            detail_record = {"role": "assistant"}
            detail_record.update(assistant_step_dict)
            self.dialog_history_details.append(detail_record)
            
    async def chat(self, user_response, last_intention_state):
        current_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        self.update_history(user_response, None)
        
        # 获取当前对话上下文（最近2轮）
        current_context_text = from_message_to_text_current_turn(self.dialog_history)
        assistant_process = {}

        # --- 步骤 1: 意图识别与槽位填充 ---
        thought_content = f"本轮对话：\n{current_context_text}\n上一状态：{last_intention_state}"
        # 格式化系统提示词
        sys_prompt = assistant_prompt_intention.format(
            task_name=self.task_name, 
            intention_types=str(self.intention_types)
        )
        
        messages = get_simple_messages(sys_prompt, thought_content)
        raw_intent_res, usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
        count_tokens(current_usage, usage)

        # 解析标签
        user_intention = self._extract_tag(raw_intent_res, 'user_intention')
        inquiry_type = self._extract_tag(raw_intent_res, 'inquiry_type')
        assistant_process['intention_thought'] = f"{user_intention}\n下一项建议：{inquiry_type}"

        # --- 步骤 2: 检查是否结束 (槽位填满标记 <summary>) ---
        if '<summary>' in raw_intent_res:
            return await self._finalize_summary(user_response, user_intention, assistant_process, current_usage)

        # --- 步骤 3: 情绪分析 (检查焦急标记 <anxiety>) ---
        emotion_content = f"分析以下对话中用户的意图和情绪：\n{current_context_text}"
        messages = get_simple_messages(assistant_prompt_emotion, emotion_content)
        emotion_res, usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
        count_tokens(current_usage, usage)
        
        assistant_process['emotion_thought'] = emotion_res.replace('<anxiety>', '').strip()

        # 如果用户焦急，强行结束并总结已有信息
        if '<anxiety>' in emotion_res:
            return await self._finalize_summary(user_response, user_intention, assistant_process, current_usage)

        # --- 步骤 4: 生成选项 (为缺失的那个维度提供参考) ---
        options_input = f"当前意图状态：{user_intention}\n当前询问类型：{inquiry_type}"
        messages = get_simple_messages(assistant_prompt_options, options_input)
        options_res, usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
        count_tokens(current_usage, usage)
        
        ref_options = self._extract_tag(options_res, 'ref_option')
        assistant_process['options_thought'] = ref_options if ref_options else options_res

        # --- 步骤 5: 生成回复话术 ---
        response_input = f"历史：{current_context_text}\n已掌握意图：{user_intention}\n待问：{inquiry_type}\n参考选项：{ref_options}"
        messages = get_simple_messages(assistant_prompt_response, response_input)
        final_text, usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
        count_tokens(current_usage, usage)
        
        assistant_process['content'] = final_text
        self.update_history(None, final_text)
        self.update_history_details(user_response, assistant_process)
        
        return assistant_process, user_intention, current_usage

    async def _finalize_summary(self, user_msg, intention_state, process_dict, current_usage):
        """内部方法：执行最后的总结步骤并结束对话"""
        summary_input = f"请根据最终意图状态进行润色总结：\n{intention_state}"
        messages = get_simple_messages(assistant_prompt_summary, summary_input)
        summary_res, usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
        count_tokens(current_usage, usage)
        
        process_dict['summary'] = summary_res
        process_dict['content'] = "已完成所有信息收集。"
        self.update_history_details(user_msg, process_dict)
        return process_dict, intention_state, current_usage

    def _extract_tag(self, text, tag):
        pattern = f'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""