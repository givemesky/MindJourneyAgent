from utils import call_openai_api_token_usage_async

class User:
    def __init__(self, name, system_prompt, model,temperature,client):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.client = client

    # async def chat(self, input_text):
    #     messages=[{'role': 'system', 'content': self.system_prompt},
    #                             {'role': 'user', 'content': input_text}]
    #     model_response, current_usages = await call_openai_api_token_usage_async(self.client, self.model, messages, self.temperature)
    #     return model_response, current_usages

    # user.py 修改 chat 方法
    async def chat(self, input_text):
        # 将 'system' 改为 'user'
        messages = [
            {'role': 'user', 'content': f"Instructions: {self.system_prompt}"},
            {'role': 'user', 'content': input_text}
        ]
        model_response, current_usages = await call_openai_api_token_usage_async(
            self.client, self.model, messages, self.temperature
        )
        return model_response, current_usages
