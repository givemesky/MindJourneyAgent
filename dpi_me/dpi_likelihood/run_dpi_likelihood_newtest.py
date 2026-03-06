# ===================================================================
#  run_dpi_likelihood_test.py 获取似然向量的测试脚本(集成新 Mapper 版)
# ===================================================================


import os
import sys
import types
import pprint
import traceback

# ===================================================
# 1. 环境自救补丁 (防止 numpy/vLLM 崩溃)
# ===================================================
os.environ["VLLM_USE_V1"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

# ===================================================
# 2. 路径与虚拟模块映射
# ===================================================
current_file_path = os.path.abspath(__file__)
logic_root = os.path.dirname(current_file_path)

if logic_root not in sys.path:
    sys.path.insert(0, logic_root)

def setup_ghost_modules():
    try:
        import perception
        import infra.data
        ct = types.ModuleType('chinatravel')
        sys.modules['chinatravel'] = ct
        sys.modules['chinatravel.agent'] = perception
        sys.modules['chinatravel.data'] = infra.data
        ct.agent = perception
        ct.data = infra.data
        print("✅ [映射] 虚拟路径映射成功")
    except Exception as e:
        print(f"⚠️ [映射] 建立失败: {e}")

setup_ghost_modules()

# ===================================================
# 3. 业务模块导入 (关键修改处)
# ===================================================
try:
    from perception.llms import Qwen
    from perception.nesy_agent.nl2sl import nl2sl_v2
    from reasoning.schema import PREFERENCE_SCHEMA
    
    # --- 修改点：从新文件 EvidenceMapper.py 中导入 EvidenceMapper 类 ---
    from reasoning.EvidenceMapper import EvidenceMapper 
    
    print("✅ [加载] 业务模块及 EvidenceMapper 加载成功")
except Exception as e:
    print(f"❌ [加载] 导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# ===================================================
# 4. 核心测试逻辑
# ===================================================
def test_dpi_full_pipeline():
    model_name = "Qwen2.5-7B-Instruct"
    
    print("\n" + "="*30 + " 系统初始化 " + "="*30)
    try:
        # 1. 初始化模型
        llm_engine = Qwen(model_name) 
        
        # 2. 初始化你的新 Mapper (传入 schema)
        # 假设你的 EvidenceMapper.__init__ 接收 schema 参数
        mapper = EvidenceMapper() 
        
        print("🚀 [初始化] vLLM 加载成功，新算法引擎 EvidenceMapper 就绪")
    except Exception as e:
        print(f"❌ [初始化] 崩溃: {e}")
        traceback.print_exc()
        return

    test_query = {
        "uid": "dpi_test_001",
        #"nature_language": "我想去上海玩3天，餐饮预算每天不超过200元，必须包含外滩。我不喜欢赶时间，喜欢慢节奏。",
        #"nature_language": "我想从武汉去重庆玩3天，希望总开销不超过4000元，必须包含洪崖洞。我想多去几个景点。",
        "nature_language": "我想从北京去武汉玩1天，希望吃饭不超过500元，必须包含中山公园。"
    }

    print("\n" + "="*30 + " 执行推理 " + "="*30)
    try:     
        print("[Step 1] NL2SL 正在提取符号逻辑约束...")
        result = nl2sl_v2(test_query, llm_engine) 
        constraints = result.get("nature_language_constraints", [])
        print(f">>> 识别到的约束: {constraints}")

        print("\n[Step 2] EvidenceMapper 正在映射似然向量 D...")
        # 调用你新写的基于正则的映射函数
        observation_D = mapper.get_likelihood_D(constraints)
        
        print("\n>>> 贝叶斯似然向量 D:")
        pprint.pprint(observation_D, width=50)

        print("\n" + "-"*20 + " 实时画像识别结果 " + "-"*20)
        for dim, scores in observation_D.items():
            max_score = max(scores)
            if max_score > 0.5:
                idx = scores.index(max_score)
                label = PREFERENCE_SCHEMA[dim][idx]
                print(f" ✅ {dim:12}: {label:10} (似然概率: {max_score})")
            else:
                print(f" ⏳ {dim:12}: 尚未捕获有效证据")

    except Exception as e:
        print(f"❌ [运行] 过程中发生错误: {e}")
        traceback.print_exc() 

if __name__ == "__main__":
    test_dpi_full_pipeline()