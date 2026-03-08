# dpi_me/engine.py (递归贝叶斯算法)
# 实现具体的数学公式。

# dpi_me/engine.py
import numpy as np

class BayesianInferenceEngine:
    def __init__(self, alpha=0.6, beta=1.2, transition_noise=0.01):
        """
        alpha: 历史保留系数 (0.9-0.98)，数值越大对历史记忆越强
        beta: 证据增强因子 (建议 1.0 - 1.5)。大于1时会放大证据的显著性，加速收敛。
        transition_noise: 状态转移噪声，允许用户在长对话中发生微小的偏好偏移
        """
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-9 # 防止除以0
        self.noise = transition_noise

    def update(self, persona, observation_D):
        """
        执行优化后的递归贝叶斯更新
        """
        for dim, likelihoods in observation_D.items():
            if dim not in persona.beliefs:
                continue
            
            # 1. 获取先验分布 (Prior)
            prior = persona.beliefs[dim]
            
            # 2. 证据强度增强 (Likelihood Sharpening)
            # 通过对似然值进行幂运算，拉大标签之间的差距，加速熵减
            likelihood_arr = np.array(likelihoods)
            if self.beta != 1.0:
                # 归一化处理以应用幂运算
                likelihood_arr = np.power(likelihood_arr, self.beta)
            
            # 3. 检查是否有有效证据
            # 如果本轮证据全是0.5左右，说明无信息增益，跳过更新防止引入噪声
            if np.all(np.isclose(likelihood_arr, likelihood_arr[0], atol=1e-2)):
                continue
                
            # 4. 贝叶斯核心更新: Posterior ∝ Prior * Likelihood
            raw_posterior = likelihood_arr * prior
            
            # 5. 归一化
            if np.sum(raw_posterior) > self.epsilon:
                posterior = raw_posterior / np.sum(raw_posterior)
            else:
                continue # 证据冲突极其严重时放弃本次更新
            
            # 6. 自适应平滑 (Adaptive Smoothing)
            # 修改点：不再简单地混合均匀分布，而是根据 alpha 平滑
            # 防止概率彻底变成 0 或 1，保留“反悔”空间
            n_labels = len(posterior)
            # 混合当前的后验分布与一个极小的均匀扰动
            # 这样既能保证收敛速度，又不会让计算陷入数值死角
            smooth_posterior = (self.alpha * posterior) + ((1 - self.alpha) * (1.0 / n_labels))
            
            # 7. 更新信念状态
            persona.beliefs[dim] = smooth_posterior

        return persona

    # def update(self, persona, observation_D):
    #     """
    #     执行递归贝叶斯更新
    #     persona: UserPersona 实例
    #     observation_D: 由感知层提供的似然向量 {dim: [p1, p2, ...]}
    #     """
    #     for dim, likelihoods in observation_D.items():
    #         if dim not in persona.beliefs:
    #             continue
            
    #         # 1. 获取先验分布 (Prior)
    #         prior = persona.beliefs[dim]
            
    #         # 2. 似然加权 (Likelihood * Prior)
    #         # 根据贝叶斯定理，后验概率正比于先验与似然的乘积
    #         likelihood_arr = np.array(likelihoods)
            
    #         # 💡 算法优化：如果本轮证据全是0.5（即无有效证据），减弱更新强度
    #         if np.all(np.isclose(likelihood_arr, 0.5)):
    #             continue
                
    #         raw_posterior = likelihood_arr * prior
            
    #         # 3. 归一化 (Normalization)
    #         # 确保概率之和等于 1
    #         if np.sum(raw_posterior) > self.epsilon:
    #             posterior = raw_posterior / np.sum(raw_posterior)
    #         else:
    #             posterior = prior
            
    #         # 4. 引入平滑与漂移 (Transition / Smoothing)
    #         # 防止某个概率变成绝对的 0 或 1，从而允许用户“反悔”或修正偏好
    #         n_labels = len(posterior)
    #         smooth_posterior = (self.alpha * posterior) + ((1 - self.alpha) * (1.0 / n_labels))
            
    #         # 5. 更新信念状态
    #         persona.beliefs[dim] = smooth_posterior

    #     return persona