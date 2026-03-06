import os
import sys
import pprint

# ===================================================
# 1. 路径自动适配：确保能找到父目录下的 dpi_likelihood
# ===================================================
current_file_path = os.path.abspath(__file__)
# dpi_reason 目录
reason_dir = os.path.dirname(current_file_path)
# dpi_me 根目录
dpi_me_root = os.path.dirname(reason_dir)

# 将 dpi_me 根目录加入 sys.path，这样才能 import dpi_likelihood
if dpi_me_root not in sys.path:
    sys.path.insert(0, dpi_me_root)

# ===================================================
# 2. 模块导入
# ===================================================
# 导入当前目录下的推理组件
from .persona import UserPersona
from .engine import BayesianInferenceEngine

# 导入隔壁目录下的感知组件
try:
    from dpi_likelihood.dpi_likelihood_interface import LikelihoodExtractor
    from dpi_likelihood.reasoning.schema import PREFERENCE_SCHEMA
except ImportError as e:
    print(f"❌ 跨模块导入失败: {e}")
    print(f"请确保 {dpi_me_root}/dpi_likelihood 目录下有 __init__.py")
    sys.exit(1)

class DPISystemManager:
    def __init__(self, model_name="Qwen2.5-7B-Instruct"):
        print(">>> [系统中枢] 正在初始化旅游偏好推理系统...")
        
        # 加载感知层 (传感器)
        self.extractor = LikelihoodExtractor(model_name)
        
        # 加载推理层 (算法)
        self.engine = BayesianInferenceEngine(alpha=0.92)
        
        # 初始化状态层 (画像)
        self.persona = UserPersona(PREFERENCE_SCHEMA)
        
        print("✅ [系统中枢] 系统初始化完成，随时可以开始对话。")

    def process_turn(self, user_text, turn_id):
        """
        全链路处理：文本 -> 逻辑约束 -> 似然向量 -> 贝叶斯更新 -> 画像演化
        """
        print(f"\n" + "="*20 + f" Turn {turn_id} " + "="*20)
        
        # 1. 感知阶段：获取本轮似然观测 D
        obs_D, constraints = self.extractor.get_likelihood_D(user_text, uid=f"t_{turn_id}")
        print(f"[感知层] 识别到约束: {constraints}")

        # ✨ 新增：打印似然向量 D，看看算法到底“看”到了什么
        print(f"[推理层] 产生的似然向量 D (Likelihood):")
        pprint.pprint(obs_D)

        # 2. 推理阶段：递归更新贝叶斯信念
        self.engine.update(self.persona, obs_D)
        
        # 3. 记录阶段：持久化演化轨迹
        self.persona.record_step(turn_id)
        
        # 4. 获取当前结论
        return self.persona.get_top_persona()

    def generate_final_report(self):
        """保存实验日志并生成最终报告"""
        log_dir = os.path.join(dpi_me_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        report_path = os.path.join(log_dir, "persona_evolution.json")
        self.persona.save_logs(report_path)
        print(f"\n📊 偏好演化日志已保存至: {report_path}")
        return self.persona.get_top_persona()

# ===================================================
# 3. 集成测试脚本
# ===================================================
if __name__ == "__main__":
    # 只需要实例化一次，加载模型
    manager = DPISystemManager()
    
    # 模拟一个典型的三轮偏好暴露场景
    test_dialogue = [
        "我想去上海玩，我是学生，想住便宜点的酒店。", 
        "必须包含外滩这个景点，我打算全程走路看风景。",
        "我不喜欢太赶的行程，希望能慢悠悠地逛。"
    ]

    for i, text in enumerate(test_dialogue):
        current_persona = manager.process_turn(text, turn_id=i+1)
        
        print(f"\n[Turn {i+1} 推理画像摘要]:")
        for dim, info in current_persona.items():
            status = "★" if info['is_stable'] else " "
            print(f" {status} {dim:12}: {info['label']:10} (置信度: {info['confidence']})")

    # 结束任务
    manager.generate_final_report()