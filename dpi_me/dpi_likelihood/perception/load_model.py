# 设定method参数，加载智能体逻辑控制器
def init_agent(kwargs):
    from .nesy_agent.rule_driven_rec import RuleDrivenAgent
    from .nesy_agent.llm_driven_rec import LLMDrivenAgent
    from .pure_neuro_agent.pure_neuro_agent import ActAgent, ReActAgent
    from .pure_neuro_agent.prompts import (
        ZEROSHOT_ACT_INSTRUCTION,
        ZEROSHOT_REACT_INSTRUCTION,
        ZEROSHOT_REACT_INSTRUCTION_GLM4,
        ONESHOT_REACT_INSTRUCTION,
        ONESHOT_REACT_INSTRUCTION_GLM4,
    )

    from .nesy_verifier import LLMModuloAgent

    from .tpc_agent.tpc_agent import TPCAgent

# 神经符号 (NeSy) RuleNeSy, LLMNeSy结合了规则驱动和大模型驱动的推荐/推理逻辑。
# 纯神经/推导	Act, ReAct, ReAct0	经典的思维链动作模型。ReAct 区分了零样本（0）和单样本（One-shot），并对 GLM4 模型做了适配。
# 验证与修正LLM-modulo包含 refine_steps 参数，侧重于模型的自我修正（Verifier）。
# 特定任务TPCAgent针对特定框架（可能是 TPC 任务）设计的专用智能体。
    if kwargs["method"] == "RuleNeSy":
        agent = RuleDrivenAgent(
            env=kwargs["env"],
            backbone_llm=kwargs["backbone_llm"],
            cache_dir=kwargs["cache_dir"],
            debug=kwargs["debug"],
        )
    elif kwargs["method"] == "LLMNeSy":
        agent = LLMDrivenAgent(
            **kwargs
        )
    elif kwargs["method"] == "Act":
        agent = ActAgent(
            env=kwargs["env"],
            backbone_llm=kwargs["backbone_llm"],
            prompt=ZEROSHOT_ACT_INSTRUCTION,
        )
    elif kwargs["method"] == "ReAct":
        agent = ReActAgent(
            env=kwargs["env"],
            backbone_llm=kwargs["backbone_llm"],
            prompt=(
                ONESHOT_REACT_INSTRUCTION
                if "glm4" not in kwargs["backbone_llm"].name.lower()
                else ONESHOT_REACT_INSTRUCTION_GLM4
            ),
        )
    elif kwargs["method"] == "ReAct0":
        agent = ReActAgent(
            env=kwargs["env"],
            backbone_llm=kwargs["backbone_llm"],
            prompt=(
                ZEROSHOT_REACT_INSTRUCTION
                if "glm4" not in kwargs["backbone_llm"].name.lower()
                else ZEROSHOT_REACT_INSTRUCTION_GLM4
            ),
        )
    elif kwargs["method"] == "LLM-modulo":
        kwargs["model"] = kwargs["backbone_llm"]
        kwargs["max_steps"] = kwargs["refine_steps"]
        agent = LLMModuloAgent(
            **kwargs
        )
    elif kwargs["method"] == "TPCAgent":
        agent = TPCAgent(
            **kwargs
        )
    else:
        raise Exception("Not Implemented")
    return agent


# 设定调用大模型类型
def init_llm(llm_name, max_model_len=None):
    from .llms import Deepseek, GPT4o, GLM4Plus, Qwen, Mistral, Llama, EmptyLLM

    from .tpc_agent.tpc_llm import TPCLLM

    if llm_name == "deepseek":
        llm = Deepseek()
    elif llm_name == "gpt-4o":
        llm = GPT4o()
    elif llm_name == "glm4-plus":
        llm = GLM4Plus()
    elif "Qwen" in llm_name:
        llm = Qwen(llm_name, max_model_len=max_model_len)
    elif llm_name == "mistral":
        llm = Mistral(max_model_len=max_model_len)
    elif "Llama" in llm_name:
        llm = Llama(llm_name)
    elif llm_name == "rule":
        return EmptyLLM()
    elif llm_name == "TPCLLM":
        llm = TPCLLM()
    else:
        raise Exception("Not Implemented")

    return llm
