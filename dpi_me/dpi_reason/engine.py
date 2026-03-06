# dpi_me/engine.py (递归贝叶斯算法)
# 实现具体的数学公式。

# dpi_me/engine.py
import numpy as np

class BayesianInferenceEngine:
    def __init__(self, alpha=0.92, transition_noise=0.01):
        """
        alpha: 历史保留系数 (0.9-0.98)，数值越大对历史记忆越强
        transition_noise: 状态转移噪声，允许用户在长对话中发生微小的偏好偏移
        """
        self.alpha = alpha
        self.epsilon = 1e-9 # 防止除以0
        self.noise = transition_noise

    def update(self, persona, observation_D):
        """
        执行递归贝叶斯更新
        persona: UserPersona 实例
        observation_D: 由感知层提供的似然向量 {dim: [p1, p2, ...]}
        """
        for dim, likelihoods in observation_D.items():
            if dim not in persona.beliefs:
                continue
            
            # 1. 获取先验分布 (Prior)
            prior = persona.beliefs[dim]
            
            # 2. 似然加权 (Likelihood * Prior)
            # 根据贝叶斯定理，后验概率正比于先验与似然的乘积
            likelihood_arr = np.array(likelihoods)
            
            # 💡 算法优化：如果本轮证据全是0.5（即无有效证据），减弱更新强度
            if np.all(np.isclose(likelihood_arr, 0.5)):
                continue
                
            raw_posterior = likelihood_arr * prior
            
            # 3. 归一化 (Normalization)
            # 确保概率之和等于 1
            if np.sum(raw_posterior) > self.epsilon:
                posterior = raw_posterior / np.sum(raw_posterior)
            else:
                posterior = prior
            
            # 4. 引入平滑与漂移 (Transition / Smoothing)
            # 防止某个概率变成绝对的 0 或 1，从而允许用户“反悔”或修正偏好
            n_labels = len(posterior)
            smooth_posterior = (self.alpha * posterior) + ((1 - self.alpha) * (1.0 / n_labels))
            
            # 5. 更新信念状态
            persona.beliefs[dim] = smooth_posterior

        return persona