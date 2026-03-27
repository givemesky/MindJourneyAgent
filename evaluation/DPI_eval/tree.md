有了预处理好的 JSON 数据，我们可以设计一套**自动化模拟实验**。

这套实验的核心逻辑是：**用一个“机器人（用户模拟器）”去调你的“API（旅游助手）”**。模拟器手握“全量需求”（`full_user_speech`），但一开始只给助手一点点信息（`initial_query`），看看助手能不能通过主动提问把剩下的信息挖出来。

以下是详细的实验设计方案：

---

### 一、 实验核心流程：多轮交互模拟 (Iterative Simulation)

对于每一条测试数据（如你提供的这组杭州数据），实验按以下步骤运行：

1.  **第一轮 (T1)**：实验脚本向 `/v1/plan` 发送 `initial_query`（“我想从上海去杭州玩2天，人数2人”）。
2.  **后端响应**：
    *   如果返回 `status: ASK`（例如问：“请问您的预算是多少？”）。
    *   **模拟器动作**：脚本从 `full_user_speech` 中提取缺失信息（或直接把整句发过去）进行回答。
    *   如果返回 `status: TERMINATE`（直接出推荐）。
3.  **循环**：重复上述过程，直到后端返回 `TERMINATE`。
4.  **记录**：保存这一路所有的 `persona` 置信度变化曲线、熵值变化、以及最终识别出的标签。

---

### 二、 核心评估指标 (Key Metrics)

你需要从三个维度量化系统表现：

#### 1. 偏好推理准确率 (Preference Inference Accuracy)
*   **计算方法**：对比后端收敛时的标签与 JSON 中的 `ground_truth.persona`。
*   **示例**：
    *   GT 是 `Pace: Relaxed`。如果后端最终识别为 `Relaxed`，得分；识别为 `Fast`，失分。
    *   计算三个维度的 **Macro-F1 Score**。

#### 2. 收敛效率 (Convergence Efficiency)
*   **TTC (Turns to Convergence)**：记录从第一句话到 `TERMINATE` 经历了多少轮。
*   **预期**：好的 DPI-ME 系统应该在 3 轮左右精准收敛。如果轮次过多，说明提问策略太低效；轮次过少（如 1 轮），说明系统在信息不足时盲目自信。

#### 3. 约束满足率 (Constraint Satisfaction Rate, CSR)
*   **计算方法**：检查生成的最终方案。
    *   是否有“江户烧肉(高德店)”？
    *   总价是否 $\le 2000$ 元？
*   这是验证 **RAG + Planner** 阶段是否真正听从了 **DPI-ME** 的推断结果。

---

### 三、 对比实验设计 (Baseline Design)

为了证明你的 DPI-ME 算法有价值，建议设置以下三个对比组：

1.  **Full-Info Group (上限组)**：直接输入 `full_user_speech`。看系统在信息完美时是否能一次性识别准确。
2.  **Random-Ask Group (对照组)**：修改后端策略，不使用熵增益（Entropy Gain）提问，而是随机在 Consumption, Interest, Pace 中抽一个维度提问。
3.  **DPI-ME Group (实验组)**：你现在的系统，使用熵值最小化策略进行提问。

**预期结论**：DPI-ME 组在 TTC（轮次）上应显著优于随机提问组，且在准确率上接近全量信息组。

---

### 四、 实验代码框架思路 (Python)

你可以写一个 `run_benchmark.py` 脚本：

```python
import json
import requests

# 加载测试集
with open("dpi_me_test_cases.json", "r") as f:
    test_cases = json.load(f)

results = []

for case in test_cases:
    session_id = f"test_{case['uid']}"
    current_input = case['initial_query']
    history = []
    
    # 开始模拟交互
    for turn in range(1, 6): # 最多测5轮
        resp = requests.post("http://api/v1/plan", json={
            "session_id": session_id,
            "user_input": current_input
        }).json()
        
        # 记录每轮的置信度，用于画图
        history.append({
            "turn": turn,
            "persona": resp['persona'],
            "status": resp['status']
        })
        
        if resp['status'] == "TERMINATE":
            break
            
        # 如果是 ASK，模拟用户提供更多信息
        # 实际实验中，可以设定：第二轮给“吃江户烧肉”，第三轮给“2000预算”
        if turn == 1:
            current_input = case['full_user_speech'] # 模拟提供全量信息
            
    # 评估
    final_persona = resp['persona']
    # 检查 final_persona 是否等于 case['ground_truth']['persona']
    # ... 统计逻辑 ...
```

---

### 五、 可视化：让实验结果会说话

1.  **熵减轨迹图 (Entropy Trajectory)**：
    X 轴是轮次，Y 轴是三个维度的熵和。展示随着对话，系统不确定性是如何陡峭下降的。
2.  **置信度热力图**：
    对比不同用户画像下，系统识别不同标签的速度。
3.  **消融实验柱状图**：
    展示有无“主动提问（ASK）”逻辑对准确率的提升。

### 针对你提供的杭州数据的特殊看点：
*   **硬约束识别**：看系统能不能把“江户烧肉(高德店)”提取出来，并在最终卡片里展示。
*   **隐性转化**：看系统能不能把“休闲为主”准确地转化为 `Pace: Relaxed`，而不是因为杭州旅游景点多就误判为 `Fast`。

**你需要我帮你写一个能自动运行这种“模拟对话”并输出准确率报告的脚本吗？**


通过你目前收集到的这五类日志（画像演化 JSON、收敛指标 CSV、约束日志 JSON、对话原文 JSON 和 评估总结 TXT），你可以进行深度的数据挖掘，支撑起一篇高质量论文或实验报告的**结果分析（Results）**部分。

以下是你可以获得的实验数据以及建议绘制的图表：

---

### 1. 核心性能指标（System Performance）
这部分数据主要来自 `eval_summary.txt` 和多个 UID 文件夹的汇总。

*   **实验数据：**
    *   **平均收敛轮次 (Average Turns to Convergence, TTC)**：系统平均聊几句能看透用户？
    *   **维度准确率 (Dimensional Accuracy)**：Consumption、Interest、Pace 各个维度的推断准确率。
    *   **总体 F1-Score**：综合评估推断的精确度。
*   **建议图表：**
    *   **柱状图 (Bar Chart)**：展示三个维度各自的准确率。
    *   **饼图 (Pie Chart)**：展示“完全匹配”、“部分匹配（2/3个维度）”和“不匹配”的比例。
    *   **直方图 (Histogram)**：X轴为轮次（1, 2, 3...），Y轴为 Case 数量。展示收敛轮次的分布情况（例如：大部分 Case 是否集中在 3-4 轮收敛）。

---

### 2. 信息论指标分析（Information Theoretic Analysis）
这部分数据来自 `persona_convergence.csv`。这是证明 **DPI-ME（动态偏好推理）** 算法有效性的核心证据。

*   **实验数据：**
    *   **熵减速度 (Entropy Decay Rate)**：每一轮对话平均消减了多少不确定性。
    *   **置信度增益 (Confidence Gain)**：每一轮提问后，系统确定性的提升幅度。
*   **建议图表：**
    *   **熵减轨迹图 (Entropy Trajectory Line Chart)**：
        *   X轴为对话轮次，Y轴为熵值（Entropy）。
        *   画出三条线（三个维度）或一条总和线。
        *   **亮点**：你应该能看到曲线在某轮（系统精准提问后）有剧烈的下降，这证明了“主动提问”的威力。
    *   **确定度增长曲线 (Certainty Growth Curve)**：
        *   展示 Certainty 指标随轮次平滑上升的过程。

---

### 3. 推理稳健性分析（Inference Robustness）
这部分数据来自 `persona_evolution.json`。

*   **实验数据：**
    *   **标签震荡率 (Label Flipping Rate)**：推断标签在对话过程中发生改变的次数。
    *   **收敛稳定性**：达到收敛后，标签是否还发生过抖动？
*   **建议图表：**
    *   **置信度堆叠面积图 (Stacked Area Chart)**：展示某一 Case 中，不同标签（如 Budget/Economy/Luxury）的概率分布随轮次的变化。
    *   **混淆矩阵 (Confusion Matrix)**：
        *   横轴为系统预测标签，纵轴为人工标注的 Ground Truth。
        *   分析系统最容易把哪个标签看走眼（比如：是否容易把 Standard 误判为 Luxury）。

---

### 4. 语义感知能力分析（Semantic Perception）
这部分数据来自 `constraints_log.json` 和 `chat_history.json`。

*   **实验数据：**
    *   **有效约束提取数**：平均每轮对话能挖掘出多少个符号约束（`Target City`, `Budget` 等）。
    *   **NL2SL 成功率**：用户说的话里包含的信息点 vs 系统提取出来的符号点。
*   **建议图表：**
    *   **约束累积曲线 (Cumulative Constraints Curve)**：随着轮次增加，系统掌握的硬约束数量。
    *   **对话长度与准确率的相关性散点图**：看是不是聊得越久提取越准。

---

### 5. 综合案例研究（Case Study）
利用 `chat_history.json` 选出两个典型的案例。

*   **图表形式：**
    *   **对话时序可视化图**：左侧放对话原文气泡，右侧对应画出该轮次后的画像雷达图或置信度条。
    *   **成功 vs 失败对比分析**：对比一个 3 轮完美收敛的案例和一个 8 轮未收敛的案例，分析失败原因（如用户表达模糊、LLM API 拦截等）。

---

### 💡 给你的后续操作建议（自动化制图）：

既然你已经有了日志，你可以写一个 **`visualizer.py`** 脚本来全自动产出这些图。步骤如下：
1.  **遍历 `logs/` 目录**下所有的 UID 文件夹。
2.  **提取 CSV 数据**：用 `pandas` 读取 `persona_convergence.csv`，计算所有 Case 在第 1, 2, 3... 轮的**平均熵值**。
3.  **提取 TXT 数据**：统计总的准确率百分比。
4.  **绘图**：用 `matplotlib` 或 `seaborn` 绘制上述图表并保存为 PDF 或 PNG。

**你需要我帮你写一段基础的“汇总统计并画出平均熵减曲线”的 Python 脚本框架吗？**