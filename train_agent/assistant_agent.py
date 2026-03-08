from prompts import assistant_prompt_intention, assistant_prompt_summary, assistant_prompt_emotion, assistant_prompt_options, assistant_prompt_response
from utils import get_simple_messages, call_openai_api_token_usage_async, from_message_to_text_current_turn,count_tokens
import re

#定义了一个名为 AssistantAgent 的 Python 类，它是一个基于异步架构的对话助手核心逻辑组件。它通过多个步骤（意图识别、情感分析、选项生成、最终回复）来处理用户的输入。

# 状态与历史管理
# __init__: 初始化模型配置、任务名称、意图类型，并维护两个列表：
# dialog_history: 存储标准的对话上下文（role/content），用于后续模型调用的背景。
# dialog_history_details: 存储更详尽的中间状态（如意图思考、情感判断、选项等），方便系统回溯或 UI 展示。
# update_history & update_history_details: 负责将每一轮的新内容追加到对应的历史记录中。
class AssistantAgent:
    def __init__(self, client, model, task_name, intention_types):
        self.model = model
        self.task_name = task_name
        self.intention_types = intention_types
        self.client = client
        self.dialog_history = []
        self.dialog_history_details = []
    def update_history(self, user_message, model_response):
        """
        Update the dialogue history.
        :param user_message: User input
        :param model_response: Model response
        """
        if user_message is not None:
            self.dialog_history.append({"role": "user", "content": user_message})
        if model_response is not None:
            self.dialog_history.append({"role": "assistant", "content": model_response})
    
    def update_history_details(self, user_message, model_response):
        """
        Update the detailed dialogue history.
        :param user_message: User input
        :param model_response: Model response
        """
        if user_message is not None:
            self.dialog_history_details.append({"role": "user", "content": user_message})
        if model_response is not None:
            temp_dict = dict()
            temp_dict['role'] = 'assistant'
            for k in model_response.keys():
                temp_dict[k] = model_response[k]
            self.dialog_history_details.append(temp_dict)

    # 多步推理工作流 (chat 方法)这是该类的核心方法，采用了一种串行推理链的设计。它不是简单地向 AI 问答，而是将任务拆解为五个阶段：阶段任务逻辑要点
    # 1. 意图思考识别用户想做什么解析 <user_intention> 和 <inquiry_type> 标签。
    # 2. 提前拦截意图摘要触发如果模型输出包含 <summary>，则直接生成摘要并提前结束流程。
    # 3. 情感检测判断用户焦虑感识别用户情绪，若检测到 <anxiety>（焦虑），同样会触发摘要逻辑并结束。
    # 4. 备选方案生成参考选项根据意图生成一系列参考建议（<ref_option>），为最终回复做参考。
    # 5. 最终回复整合生成回答结合历史、意图、查询类型和参考选项，生成给用户的最终文本。       
    async def chat(self, user_response, user_intention):
        # Count tokens
        current_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        self.update_history(user_response, None)
        current_history = from_message_to_text_current_turn(self.dialog_history)
        assistant_response = dict()

        # 1. User intention thinking
        thought_content = f"```本轮对话：\n{current_history}```\n```上一状态的用户意图：{user_intention}```\n根据以上信息，完成你的任务。"
        messages = get_simple_messages(assistant_prompt_intention.format(task_name=self.task_name, intention_types=self.intention_types), thought_content)
        intention_response, step_usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
        count_tokens(current_usage, step_usage)
        # print(intention_response)

        # Parse intention_response
        user_intention_pattern = r'<user_intention>(.*?)</user_intention>'
        user_intention_match = re.search(user_intention_pattern, intention_response, re.DOTALL)
        if user_intention_match:
            user_intention = user_intention_match.group(1)

        inquiry_type = ''
        inquiry_type_pattern = r'<inquiry_type>(.*?)</inquiry_type>'
        inquiry_type_match = re.search(inquiry_type_pattern, intention_response, re.DOTALL)
        if inquiry_type_match:
            inquiry_type = inquiry_type_match.group(1)

        assistant_response['intention_thought'] = user_intention.replace('<summary>', '') + '\n' + inquiry_type.replace('<summary>', '')

        # 2. Intention summarization
        if '<summary>' in intention_response:
            summary_content = f"用户意图：\n{user_intention}```根据以上信息，完成你的任务，总结用户意图。"
            messages = get_simple_messages(assistant_prompt_summary, summary_content)
            summary_response, step_usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
            count_tokens(current_usage, step_usage)
            # print(summary_response)
            assistant_response['summary'] = summary_response
            self.update_history_details(user_response, assistant_response)
            return assistant_response, user_intention, current_usage

        # 3. Emotion
        emotion_content = f"```本轮对话：\n{current_history}```根据以上信息，完成你的任务，输出你的判断。"
        messages = get_simple_messages(assistant_prompt_emotion, emotion_content)
        emotion_response, step_usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
        count_tokens(current_usage, step_usage)
        # print(emotion_response)
        assistant_response['emotion_thought'] = emotion_response.replace('<anxiety>', '')

        # Trigger 3. Intention summarization
        if '<anxiety>' in emotion_response:
            summary_content = f"用户意图：\n{user_intention}```根据以上信息，完成你的任务，总结用户意图。"
            messages = get_simple_messages(assistant_prompt_summary, summary_content)
            summary_response, step_usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
            count_tokens(current_usage, step_usage)
            # print(summary_response)
            assistant_response['summary'] = summary_response
            self.update_history_details(user_response, assistant_response)
            return assistant_response, user_intention, current_usage

        # 4. Reference options
        options_content = f"```用户意图：\n{user_intention}```\n```{inquiry_type}```\n根据以上信息，完成你的任务。"
        messages = get_simple_messages(assistant_prompt_options, options_content)
        options_response, step_usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
        count_tokens(current_usage, step_usage)
        # print(options_response)
        options_pattern = r'<ref_option>(.*?)</ref_option>'
        options_match = re.search(options_pattern, options_response, re.DOTALL)
        if options_match:
            options = options_match.group(1)
            assistant_response['options_thought'] = options_response[:options_response.find('<')] + '参考选项：' + options
        else:
            options = options_response
            assistant_response['options_thought'] = options_response

        # 5. Respond to user
        response_content = f"```本轮对话：\n{current_history}```\n```用户意图：\n{user_intention}```\n```{inquiry_type}```\n```参考选项：\n{options}```\n根据以上信息和你的任务，回复用户。"
        messages = get_simple_messages(assistant_prompt_response, response_content)
        response, step_usage = await call_openai_api_token_usage_async(self.client, self.model, messages)
        count_tokens(current_usage, step_usage)
        # print(response)
        assistant_response['content'] = response
        
        # 工具集成
        # Token 统计: 使用 count_tokens 实时追踪 prompt_tokens 和 completion_tokens，这对于成本控制和性能监控至关重要。
        # 正则解析: 大量使用 re.search 来提取模型返回的结构化 XML 标签内容，体现了“结构化 Prompt”的设计思路。
        # Update dialogue history
        self.update_history(None, response)
        self.update_history_details(user_response, assistant_response)
        return assistant_response, user_intention, current_usage
