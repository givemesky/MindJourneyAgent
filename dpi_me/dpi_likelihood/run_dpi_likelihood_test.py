# ===================================================
# 获取似然向量的测试脚本
# ===================================================

import os
import sys
import types
import pprint
import traceback

# ===================================================
# 1. 环境自救补丁 (防止 numpy/vLLM 崩溃)
# ===================================================
os.environ["OPENBLAS_CORETYPE"] = "Haswell"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["VLLM_USE_V1"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

try:
    import torch
    import numpy as np
    print(f">>> [环境] 基础库加载正常 (Numpy: {np.__version__})")
except Exception as e:
    print(f"⚠️ [环境] 基础库警告: {e}")

# ===================================================
# 2. 路径与“核心映射”逻辑 (解决 ModuleNotFoundError)
# ===================================================
current_file_path = os.path.abspath(__file__)
logic_root = os.path.dirname(current_file_path)

if logic_root not in sys.path:
    sys.path.insert(0, logic_root)

def setup_ghost_modules():
    """
    终极补丁：将旧的 chinatravel 路径在内存中全部映射到新路径
    """
    try:
        # 先加载新路径下的模块
        import perception
        import infra
        import infra.data
        
        # 创建虚拟的 chinatravel 顶层包
        ct = types.ModuleType('chinatravel')
        sys.modules['chinatravel'] = ct
        
        # 核心映射：把 chinatravel.agent 映射到 perception
        # 这样无论在哪个 __init__.py 里执行 from chinatravel.agent... 都会成功
        sys.modules['chinatravel.agent'] = perception
        ct.agent = perception
        
        # 核心映射：把 chinatravel.data 映射到 infra.data
        sys.modules['chinatravel.data'] = infra.data
        ct.data = infra.data
        
        print("✅ [映射] 成功建立了虚拟路径映射 (chinatravel -> perception/infra)")
    except Exception as e:
        print(f"⚠️ [映射] 建立失败: {e}")

setup_ghost_modules()

# ===================================================
# 3. 业务模块导入
# ===================================================
try:
    from perception.llms import Qwen
    from perception.nesy_agent.nl2sl import nl2sl_v2
    from reasoning.mapper import EvidenceMapper
    from reasoning.schema import PREFERENCE_SCHEMA
    print("✅ [加载] 业务模块全部就绪")
except Exception as e:
    print(f"❌ [加载] 业务模块导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# ===================================================
# 4. 核心测试逻辑
# ===================================================
def test_dpi_full_pipeline():
    model_name = "Qwen2.5-7B-Instruct"
    
    print("\n" + "="*30 + " 系统初始化 " + "="*30)
    try:
        # 初始化模型
        llm_engine = Qwen(model_name) 
        mapper = EvidenceMapper()
        print("🚀 [初始化] vLLM 加载成功，算法引擎就绪")
    except Exception as e:
        print(f"❌ [初始化] 崩溃: {e}")
        traceback.print_exc()
        return

    test_query = {
        "uid": "dpi_test_001",
        "nature_language": "我想去上海玩3天，餐饮预算每天不超过200元，必须包含外滩。我不喜欢赶时间，喜欢慢节奏。",
    }

    print("\n" + "="*30 + " 执行推理 " + "="*30)
    try:     
        print("[Step 1] 正在通过 NL2SL 提取逻辑约束...")
        result = nl2sl_v2(test_query, llm_engine) 
        constraints = result.get("nature_language_constraints", [])
        
        print(f">>> 识别到的约束: {constraints}")

        print("\n[Step 2] 正在将约束映射为贝叶斯似然向量 D...")
        observation_D = mapper.get_likelihood_D(constraints)
        pprint.pprint(observation_D, width=50)

        print("\n" + "-"*20 + " 实时画像识别结果 " + "-"*20)
        for dim, scores in observation_D.items():
            max_score = max(scores)
            if max_score > 0.5:
                idx = scores.index(max_score)
                label = PREFERENCE_SCHEMA[dim][idx]
                print(f" ✅ {dim:12}: {label:10} (似然概率: {max_score})")

    except Exception as e:
        print(f"❌ [运行] 过程中发生错误: {e}")
        traceback.print_exc() 

if __name__ == "__main__":
    test_dpi_full_pipeline()