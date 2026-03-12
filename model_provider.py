# -*- coding: utf-8 -*-
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class SharedModelProvider:
    _instance = None
    model = None
    tokenizer = None

    BASE_MODEL_PATH = "/public/home/azsy/MindJourney/MindJourneyAgent/model/Qwen2.5-7B-Instruct"
    LORA_PATH = "/public/home/azsy/SynPT-main/lora"

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._initialize_model()
        return cls._instance

    @classmethod
    def _initialize_model(cls):
        print(f"📦 [SharedModel] 正在初始化全局共享模型资源...")
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.BASE_MODEL_PATH, trust_remote_code=True)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            cls.BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if os.path.exists(cls.LORA_PATH):
            print(f"   --> 加载 LoRA 权重: {cls.LORA_PATH}")
            cls.model = PeftModel.from_pretrained(base_model, cls.LORA_PATH)
        else:
            cls.model = base_model
        
        cls.model.eval()
        print("✅ [SharedModel] 模型加载完成。")

    def chat_generate(self, messages, max_new_tokens=512):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            new_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            return self.tokenizer.decode(new_ids, skip_special_tokens=True)