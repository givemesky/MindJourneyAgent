# dpi_me/model_provider.py [核心显存管理]：
# 职责：Qwen2.5+LoRA 共享单例，提供全局 generate 接口
# 功能：提供一个全局唯一的生成接口，供所有人（DPI算法和对话逻辑）调用。

# -*- coding: utf-8 -*-
"""
Global Shared Model Provider (全局共享模型加载器)
功能：单例模式加载 Qwen2.5 + LoRA，确保显存只被占用一次。
      为对话助手 (SynPT) 和 逻辑提取 (DPI-ME) 提供统一接口。
"""

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class SharedModelProvider:
    _instance = None
    model = None
    tokenizer = None

    # 配置路径 (基于 chat_proactive.py)
    BASE_MODEL_PATH = "/public/home/azsy/SynPT-main/model/Qwen2.5-7B-Instruct"
    LORA_PATH = "/public/home/azsy/SynPT-main/lora"

    @classmethod
    def get_instance(cls):
        """获取或初始化共享模型实例"""
        if cls._instance is None:
            cls._instance = cls()
            cls._initialize_model()
        return cls._instance

    @classmethod
    def _initialize_model(cls):
        """模型加载核心逻辑"""
        print(f"📦 [SharedModel] 正在初始化全局共享模型资源...")
        
        # 1. 加载 Tokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.BASE_MODEL_PATH, 
            trust_remote_code=True
        )
        
        # 2. 加载基础模型 (使用 bfloat16 和 auto 映射以优化显存)
        print(f"   --> 加载基座模型: {os.path.basename(cls.BASE_MODEL_PATH)}")
        base_model = AutoModelForCausalLM.from_pretrained(
            cls.BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 3. 加载 LoRA Adapter (Peft)
        if os.path.exists(cls.LORA_PATH):
            print(f"   --> 加载 LoRA 权重: {os.path.basename(cls.LORA_PATH)}")
            cls.model = PeftModel.from_pretrained(base_model, cls.LORA_PATH)
        else:
            print(f"   ⚠️ [SharedModel] 未发现 LoRA 目录，将使用基座模型推理。")
            cls.model = base_model
        
        cls.model.eval()
        print("✅ [SharedModel] 全局模型与 LoRA 加载完成，显存共享已开启。")

    @classmethod
    def chat_generate(cls, messages, max_new_tokens=512):
        """
        供 SynPT 对话助手使用：支持上下文记忆和聊天模板
        :param messages: List of dicts, e.g., [{"role": "user", "content": "..."}]
        """
        if cls.model is None: cls.get_instance()

        # 使用 Qwen 官方聊天模板转换格式
        text = cls.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = cls.tokenizer([text], return_tensors="pt").to(cls.model.device)
        
        with torch.no_grad():
            generated_ids = cls.model.generate(
                **model_inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=cls.tokenizer.eos_token_id
            )
            # 只取新生成的内容
            new_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            response = cls.tokenizer.decode(new_ids, skip_special_tokens=True)
            
        return response

    @classmethod
    def extract_generate(cls, prompt, max_new_tokens=256):
        """
        供 DPI-ME 感知层 (NL2SL) 使用：执行指令式的逻辑提取
        :param prompt: 构造好的 NL2SL Prompt 字符串
        """
        if cls.model is None: cls.get_instance()

        # 感知层通常是单次推理，不需要维护 messages 历史，直接生成
        inputs = cls.tokenizer(prompt, return_tensors="pt").to(cls.model.device)
        
        with torch.no_grad():
            outputs = cls.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False, # 逻辑提取建议关闭随机性，使用 Greedy Search
                pad_token_id=cls.tokenizer.eos_token_id
            )
            response = cls.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):], 
                skip_special_tokens=True
            )
            
        return response