# ===================================================
#  dpi_likelihood接口文件: 获取似然向量D接口文件 (集成 EvidenceMapper)
# ===================================================

import os
import sys
import types
import pprint

# ===================================================
# 1. 核心接口类
# ===================================================
class LikelihoodExtractor:
    def __init__(self, model_name="Qwen2.5-7B-Instruct"):
        """
        初始化：加载模型和映射器（此过程耗时，建议只执行一次）
        """
        self._setup_env()
        self._setup_ghost_modules()
        
        # 延迟导入业务模块，防止路径未初始化
        # 注意：这里导入的是你的新分区路径和新文件
        from perception.llms import Qwen
        from reasoning.schema import PREFERENCE_SCHEMA
        from reasoning.EvidenceMapper import EvidenceMapper
        
        print(f">>> [DPI 接口] 正在初始化模型: {model_name}...")
        self.llm_engine = Qwen(model_name)
        
        # 修复点：传入 PREFERENCE_SCHEMA 避免 TypeError
        self.mapper = EvidenceMapper(PREFERENCE_SCHEMA)
        
        print("✅ [DPI 接口] 系统初始化完成。")

    def _setup_env(self):
        """环境变量与路径初始化"""
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
        
        # 获取当前 interface.py 所在的目录 (SynPT-main/dpi_me/dpi_likelihood)
        self.logic_root = os.path.dirname(os.path.abspath(__file__))
        if self.logic_root not in sys.path:
            sys.path.insert(0, self.logic_root)

    def _setup_ghost_modules(self):
        """虚拟模块映射逻辑（确保 nl2sl.py 能正常工作）"""
        try:
            # 物理导入
            from perception import nesy_agent
            from infra.data import load_datasets as i_data
            
            # 创建虚拟包路径
            ct = types.ModuleType('chinatravel')
            ct.agent = types.ModuleType('agent')
            ct.data = types.ModuleType('data')
            sys.modules['chinatravel'] = ct
            sys.modules['chinatravel.agent'] = ct.agent
            sys.modules['chinatravel.data'] = ct.data
            
            # 指向物理位置
            sys.modules['chinatravel.agent.nesy_agent'] = nesy_agent
            sys.modules['chinatravel.data.load_datasets'] = i_data
            ct.agent.nesy_agent = nesy_agent
            ct.data.load_datasets = i_data
            
            print("✅ [DPI 接口] 虚拟路径 chinatravel 映射成功")
        except Exception as e:
            print(f"⚠️ [DPI 接口] 模块映射警告: {e}")

    def get_likelihood_D(self, user_text, uid="api_user"):
        """
        外部调用主函数：输入一句话，输出似然向量 D
        """
        from perception.nesy_agent.nl2sl import nl2sl_v2
        
        # 1. 构造查询结构
        query = {"nature_language": user_text, "uid": uid}
        
        # 2. 执行 NL2SL 感知层 (神经网络 -> 符号)
        # 注意：这里会通过 vLLM 进行一次模型推理
        result = nl2sl_v2(query, self.llm_engine)
        constraints = result.get("nature_language_constraints", [])
        
        # 3. 执行 EvidenceMapper 推理层 (符号 -> 似然概率)
        #obs_D = self.mapper.get_likelihood_D(constraints)
        obs_D = self.mapper.get_likelihood_D(constraints, raw_text=user_text)
        
        return obs_D, constraints

# ===================================================
# 2. 调用示例
# ===================================================
if __name__ == "__main__":
    # 在主程序中只需要实例化一次，加载 7B 模型权重
    dpi_service = LikelihoodExtractor()
    
    # 模拟真实用户的多轮偏好暴露
    test_texts = [
        "我想去上海玩，每天餐饮预算200元。",
        "想去外滩看夜景，打算步行游览，不赶时间。"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n>>> 模拟对话 Turn {i+1}: {text}")
        D, sl = dpi_service.get_likelihood_D(text)
        
        print(f"识别到的逻辑约束: {sl}")
        print("产生的似然向量 D:")
        pprint.pprint(D)

# # ===================================================
# #  dpi_likelihood接口文件:获取似然向量D接口文件，旧版本用mapper
# # ===================================================



# import os
# import sys
# import types
# import pprint

# # ===================================================
# # 1. 核心接口类
# # ===================================================
# class LikelihoodExtractor:
#     def __init__(self, model_name="Qwen2.5-7B-Instruct"):
#         """
#         初始化：加载模型和映射器（此过程耗时，建议只执行一次）
#         """
#         self._setup_env()
#         self._setup_ghost_modules()
        
#         # 延迟导入业务模块，防止路径未初始化
#         from perception.llms import Qwen
#         from reasoning.mapper import EvidenceMapper
        
#         print(f">>> [DPI 接口] 正在初始化模型: {model_name}...")
#         self.llm_engine = Qwen(model_name)
#         self.mapper = EvidenceMapper()
#         print("✅ [DPI 接口] 系统初始化完成。")

#     def _setup_env(self):
#         """环境变量与路径初始化"""
#         os.environ["VLLM_USE_V1"] = "0"
#         os.environ["NCCL_P2P_DISABLE"] = "1"
        
#         # 获取当前 interface.py 所在的目录
#         self.logic_root = os.path.dirname(os.path.abspath(__file__))
#         if self.logic_root not in sys.path:
#             sys.path.insert(0, self.logic_root)

#     def _setup_ghost_modules(self):
#         """虚拟模块映射逻辑（确保 nl2sl.py 能正常工作）"""
#         try:
#             import perception
#             import infra.data.load_datasets as i_data
            
#             ct = types.ModuleType('chinatravel')
#             sys.modules['chinatravel'] = ct
#             sys.modules['chinatravel.agent'] = perception
#             sys.modules['chinatravel.data'] = i_data
#             ct.agent = perception
#             ct.data = i_data
#         except Exception as e:
#             print(f"⚠️ [DPI 接口] 模块映射警告: {e}")

#     def get_likelihood_D(self, user_text, uid="api_user"):
#         """
#         外部调用主函数：输入一句话，输出似然向量 D
#         """
#         from perception.nesy_agent.nl2sl import nl2sl_v2
        
#         # 1. 构造查询结构
#         query = {"nature_language": user_text, "uid": uid}
        
#         # 2. 执行 NL2SL 感知层
#         result = nl2sl_v2(query, self.llm_engine)
#         constraints = result.get("nature_language_constraints", [])
        
#         # 3. 执行 Mapper 推理层
#         obs_D = self.mapper.get_likelihood_D(constraints)
        
#         return obs_D, constraints

# # ===================================================
# # 2. 调用示例 (用于展示如何集成)
# # ===================================================
# if __name__ == "__main__":
#     # 在主程序中只需要实例化一次
#     dpi_service = LikelihoodExtractor()
    
#     # 模拟后续的实时调用
#     text = "我想去北京玩，想住五星级酒店，预算不封顶。"
#     D, sl = dpi_service.get_likelihood_D(text)
    
#     print("\n--- 调用结果 ---")
#     print(f"逻辑约束: {sl}")
#     pprint.pprint(D)