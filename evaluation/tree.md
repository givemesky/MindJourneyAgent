MindJourney_Eval/
├── data/
│   ├── raw/
│   │   └── test.json               # 你解压后的 CrossWOZ 原始测试集
│   └── processed/
│       └── test_personas.json      # 【脚本生成】提取并映射后的用户画像（Ground Truth）
├── agents/
│   ├── __init__.py
│   ├── assistant_agent.py          # 加载 LoRA 模型或 Base 模型的封装类
│   └── user_simulator.py           # 基于 GPT-4o 或 Qwen-Max 的用户模拟器逻辑
├── core/
│   ├── __init__.py
│   ├── evaluator.py                # 核心逻辑：控制对话轮次、统计 Recall/TTC 指标
│   └── parser.py                   # 解析器：提取 <think>, <answer>, <summary> 标签内容
├── prompts/
│   ├── user_system_prompt.py       # 驱动 User Simulator 的提示词（要求其“挤牙膏”式回答）
│   ├── base_system_prompt.py       # 给 Base 模型的 System Prompt
│   └── few_shot_examples.py        # Few-shot 组使用的示例数据
├── results/                        # 【自动生成】存放实验结果
│   ├── logs/                       # 存放每一场对话的完整对话历史（.txt 或 .json）
│   └── metrics_summary.json        # 最终的对比统计报表（Recall, TTC, Accuracy 等）
├── utils/
│   ├── api_client.py               # 封装大模型 API 调用（支持本地 LoRA 和 远程 API）
│   └── data_utils.py               # 格式转换工具
├── config.py                       # 配置文件：模型路径、API Key、测试轮数等
├── run_eval.py                     # 【主入口】运行自动化评估实验
└── analyze_results.py              # 【可视化】读取 results 绘制对比图表的脚本

这是一个非常专业的实验设计思路。既然你决定使用 **CrossWOZ** 作为基础，你的实验设计核心逻辑应该是：**“将 CrossWOZ 的目标（Goal）作为标准答案（Ground Truth），测试不同模型从‘挤牙膏式’的用户口中挖掘出这些信息的效率。”**


# 1. 运行基座对比组
python evaluator.py  --mode base --limit 5

# 2. 运行 Few-shot 对比组
python evaluator.py  --mode few_shot --limit 5

# 3. 运行你的微调模型组
python evaluator.py  --mode lora --limit 5

以下是为您设计的完整实验方案：

---

### 1. 实验组设置 (Experimental Setup)

为了证明 LoRA 微调的有效性，你需要设置以下对比组（消融实验）：

*   **Group A (Base):** 原始 Qwen2.5-7B-Instruct（仅通过 System Prompt 告知任务，不进行微调）。
*   **Group B (Few-shot):** 原始 Qwen2.5-7B-Instruct + 3-5 个带 `<think>` 标签的示例（测试纯 Prompt Engineering 的极限）。
*   **Group C (MindJourney-LoRA):** 你微调后的模型（本项目核心）。

---

### 2. 测试集构建 (Test Set Construction)

从 CrossWOZ 的 `test.json` 中抽取 100-200 个对话目标（Goal）。

1.  **特征提取**：将 CrossWOZ 的复杂目标映射到你的 4 个维度。
    *   *例如：* 提取 `人均消费: 50-100元` -> 对应 `Consumption: Economy`。
2.  **构建 User Simulator (用户模拟器)**：
    *   你需要一个专门的 **User Agent**（可以用 GPT-4o 或原版 Qwen-Max）。
    *   **核心指令**：给它 CrossWOZ 的全量 Goal，但要求它：“除非助手主动询问，否则不要一次性说出所有需求。每轮只回答助手提出的问题。”

---

### 3. 核心评估指标 (Metrics)

你需要编写一个评估脚本，在对话结束后自动计算以下指标：

#### ① 信息完整度 (Recall / Information Completeness)
*   **定义**：在对话结束（或达到最大轮数 6 轮）时，模型在 `<summary>` 标签中成功识别出的维度比例。
*   **公式**：`Recall = (正确识别的维度数) / 4`。
*   **预期**：微调后的模型应显著高于 Base 组，因为 Base 组往往会忘记在末尾输出结构化总结。

#### ② 收敛步数 (Turns to Completion, TTC)
*   **定义**：从对话开始，到模型集齐所有 4 个维度信息并给出初步建议所经历的**有效对话轮次**。
*   **意义**：步数越少，证明模型的主动挖掘策略（Strategy）越精准，没有废话。
*   **对比点**：观察微调模型是否比 Base 组更早地问出关键信息（如 Consumption）。

#### ③ 诱导成功率 (Proactivity Rate)
*   **定义**：在用户仅提供模糊需求（如“我想去玩”）的情况下，模型在第一轮就主动发起追问（而不是直接盲目推荐）的比例。
*   **判定标准**：解析 `<think>` 标签中是否包含“挖掘”、“询问”等关键词，或回复中是否包含问号。

#### ④ 标签准确率 (Schema Accuracy)
*   **定义**：模型在 `<summary>` 中提取的标签是否与 CrossWOZ Goal 的原始数据一致。
*   **意义**：证明模型不仅在追问，而且听懂了用户的回答并正确映射到了 Schema。

---

### 4. 实验流程设计 (Workflow)

1.  **自动化对弈**：
    *   让 **User Agent (基于 CrossWOZ Goal)** 与 **Assistant Agent (Group A/B/C)** 进行对话。
    *   设置最大对话轮数（建议 6-8 轮）。
2.  **日志记录**：
    *   保存每一轮的对话文本、`<think>` 内容以及 `<summary>` 的 JSON 状态。
3.  **统计分析**：
    *   使用 Python 脚本解析日志，计算上述四个指标的平均值。

---

### 5. 消融实验的可视化建议

在毕业论文或项目报告中，你可以使用以下图表来展示 LoRA 的威力：

*   **柱状图**：对比 A/B/C 三组在 **Recall** 和 **Accuracy** 上的表现。
*   **折线图**：展示 **TTC (收敛步数)** 的分布，微调模型应该在 3-4 轮处出现峰值，而 Base 模型可能分布在 6 轮以上或根本无法收敛。
*   **案例对比表**：展示同一个 CrossWOZ Goal 下，Base 模型和 LoRA 模型的对话轨迹对比。
    *   *Base 组*：用户说去北京，它直接推荐了故宫（未询问预算）。
    *   *LoRA 组*：用户说去北京，`<think>` 显示“目的地已明确，缺失消费偏好，需追问”，随后询问了预算。

---

### 6. 一个具体的评估脚本 Demo (伪代码)

```python
results = []
test_goals = load_crosswoz_goals("test.json")[:100] # 取100个样本

for goal in test_goals:
    # 模拟一次对话
    history = run_dialogue_simulation(assistant_model, user_simulator, goal)
    
    # 评估
    last_summary = parse_json(history[-1]['output']) # 解析最后的标签
    gt_tags = map_goal_to_schema(goal) # 标准答案
    
    metrics = {
        "recall": calculate_recall(last_summary, gt_tags),
        "turns": len(history),
        "proactive": check_if_asked_questions(history[1]) # 第一轮是否追问
    }
    results.append(metrics)

# 计算平均分
avg_recall = sum([m['recall'] for m in results]) / len(results)
print(f"微调模型平均召回率: {avg_recall}")
```

### 总结建议
利用 CrossWOZ 里的 **"Goal"** 做为 **User Agent** 的驱动引擎，是证明你 Agent “聪明程度”的最硬核方式。如果你能证明 **LoRA 模型在更短的轮次内（TTC更低）获得了更全的信息（Recall更高）**，你的毕业设计深度将远超普通的对话机器人项目。