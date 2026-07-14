"""
RECAP: RL with Experience and Corrections via Advantage-conditioned Policies
基于 openpi (π0.5) 的复现实现

论文: π*0.6: a VLA That Learns From Experience (arXiv:2511.14759)
"""

import dataclasses
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from openpi.recap.conditioning import ConditioningState
from openpi.recap.conditioning import apply_condition_dropout
from openpi.recap.conditioning import combine_cfg


# ============================================================
# 1. 分布式价值函数 (Distributional Value Function)
# ============================================================

@dataclasses.dataclass
class ValueFunctionConfig:
    """价值函数配置"""
    # VLM 主干参数
    vlm_backbone: str = "gemma_2b"  # 使用比策略模型更小的VLM
    num_value_bins: int = 100        # 价值离散化 bin 数
    value_min: float = -1.0          # 价值范围下界 (失败 episode return=-1)
    value_max: float = 1.0           # 价值范围上界 (成功 episode return=+1, 含时间奖励时可能>0)
    
    # 训练参数
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 100000
    batch_size: int = 256
    ema_decay: float = 0.99
    
    # 数据混合
    web_data_ratio: float = 0.05     # 5% 多模态网页数据防止过拟合


class DistributionalValueFunction:
    """
    分布式价值函数 p_φ(V | o_t, ℓ) ∈ Δ_B
    
    将状态价值建模为离散分布而非单点估计，提供更鲁棒的价值预测。
    架构与 VLA 策略相同，但使用更小的 VLM 主干。
    
    输入: 观测 o_t (图像 + 本体感觉) + 语言指令 ℓ
    输出: B 个 bin 上的概率分布
    """
    
    def __init__(self, config: ValueFunctionConfig):
        self.config = config
        self.bin_edges = jnp.linspace(
            config.value_min, config.value_max, config.num_value_bins + 1
        )
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
    
    def compute_loss(
        self,
        value_logits: jnp.ndarray,    # (B, num_bins)
        empirical_returns: jnp.ndarray, # (B,)
    ) -> jnp.ndarray:
        """
        计算交叉熵损失
        
        L = -log p_φ(V = R_t(τ) | o_t, ℓ)
        
        其中 R_t(τ) = Σ_{t'=t}^{T} r_{t'} 是经验回报
        """
        # 将经验回报映射到 bin 索引
        bin_indices = self._returns_to_bin_indices(empirical_returns)
        
        # 标准交叉熵损失
        log_probs = jax.nn.log_softmax(value_logits, axis=-1)
        loss = -jnp.take_along_axis(
            log_probs, bin_indices[:, None], axis=-1
        ).squeeze(-1)
        
        return loss
    
    def predict_value(self, value_logits: jnp.ndarray) -> jnp.ndarray:
        """
        从分布中提取期望价值
        
        V(o_t, ℓ) = E[p_φ(V | o_t, ℓ)] = Σ_b p_b · v_b
        """
        probs = jax.nn.softmax(value_logits, axis=-1)
        expected_value = jnp.sum(probs * self.bin_centers, axis=-1)
        return expected_value
    
    def _returns_to_bin_indices(self, returns: jnp.ndarray) -> jnp.ndarray:
        """将连续回报映射到离散 bin 索引"""
        range_span = self.config.value_max - self.config.value_min
        normalized = (returns - self.config.value_min) / range_span
        # 使用 floor 映射，并 clip 到 [0, B-1] 防止越界
        # 当 returns == value_max 时，normalized=1.0 → idx=B，clip 到 B-1
        bin_idx = jnp.clip(
            jnp.floor(normalized * self.config.num_value_bins).astype(jnp.int32),
            0, self.config.num_value_bins - 1
        )
        return bin_idx


# ============================================================
# 2. 优势计算 (Advantage Computation)
# ============================================================

@dataclasses.dataclass
class AdvantageConfig:
    """优势计算配置"""
    n_step: int = 50                  # N-step lookahead (post-training)
    advantage_threshold_pct: float = 0.40  # 正优势比例 (40% 分位)
    use_mc_estimate: bool = False     # True=Monte Carlo (pre-training), False=N-step
    gamma: float = 1.0                # 折扣因子 (论文不用折扣)


class AdvantageComputer:
    """
    优势值计算与二值化
    
    Post-training (N-step, 更精确):
        A^π(o_t, a_t) = Σ_{t'=t}^{t+N-1} r_{t'} + V^π(o_{t+N}) - V^π(o_t)
    
    Pre-training (Monte Carlo, 更高效):
        A^π(o_t, a_t) = Σ_{t'=0}^{T} r_{t'} - V^π(o_t)
    """
    
    def __init__(self, config: AdvantageConfig, value_fn: DistributionalValueFunction):
        self.config = config
        self.value_fn = value_fn
    
    def compute_advantages(
        self,
        rewards: np.ndarray,           # (T,) 每步奖励
        value_predictions: np.ndarray,  # (T,) 每步价值预测
        episode_length: int,
    ) -> np.ndarray:
        """
        计算每步的 advantage 值
        
        Args:
            rewards: episode 中每步的奖励
            value_predictions: 价值函数对每步的预测值
            episode_length: episode 长度
            
        Returns:
            advantages: (T,) 每步的优势值
        """
        T = episode_length
        advantages = np.zeros(T, dtype=np.float32)
        
        gamma = self.config.gamma

        if self.config.use_mc_estimate:
            # Monte Carlo: A = Σ γ^k r_{t+k} - V(o_t)
            for t in range(T):
                discounted_return = 0.0
                for k in range(T - t):
                    discounted_return += (gamma ** k) * rewards[t + k]
                advantages[t] = discounted_return - value_predictions[t]
        else:
            # N-step: A = Σ_{k=0}^{N-1} γ^k r_{t+k} + γ^N V(o_{t+N}) - V(o_t)
            N = self.config.n_step
            for t in range(T):
                n_step_return = 0.0
                for k in range(min(N, T - t)):
                    n_step_return += (gamma ** k) * rewards[t + k]
                if t + N < T:
                    bootstrap_value = (gamma ** N) * value_predictions[t + N]
                else:
                    bootstrap_value = 0.0  # episode 结束后价值为 0
                advantages[t] = n_step_return + bootstrap_value - value_predictions[t]
        
        return advantages
    
    def binarize_advantages(
        self,
        advantages: np.ndarray,
        task_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        将连续优势值二值化为 positive/negative 标签
        
        阈值设定:
        - 预训练: 约 30% 数据为 positive
        - 微调: 约 40% 数据为 positive
        - 严格标准任务: 约 10% 为 positive
        """
        threshold_pct = self.config.advantage_threshold_pct
        threshold = np.percentile(advantages, (1 - threshold_pct) * 100)
        
        labels = np.where(
            advantages > threshold,
            AdvantageLabel.POSITIVE,
            AdvantageLabel.NEGATIVE,
        )
        return labels
    
    def compute_advantages_for_dataset(
        self,
        episodes: list,
        value_fn_params: dict,
    ) -> list:
        """
        对整个数据集计算优势标签
        
        Args:
            episodes: episode 列表，每个包含 observations, actions, rewards, language
            value_fn_params: 价值函数参数
            
        Returns:
            带优势标签的 episode 列表
        """
        labeled_episodes = []
        all_advantages = []
        
        for ep in episodes:
            # 1. 用价值函数预测每步价值
            value_preds = self._predict_values(ep, value_fn_params)
            
            # 2. 计算优势
            advantages = self.compute_advantages(
                ep['rewards'], value_preds, len(ep['rewards'])
            )
            all_advantages.extend(advantages.tolist())
            
            # 3. 存储优势值（二值化在全局计算后进行）
            ep_with_adv = {**ep, 'advantages': advantages}
            labeled_episodes.append(ep_with_adv)
        
        # 4. 全局二值化
        all_advantages = np.array(all_advantages)
        threshold = np.percentile(all_advantages, (1 - self.config.advantage_threshold_pct) * 100)
        
        for ep in labeled_episodes:
            ep['advantage_labels'] = np.where(
                ep['advantages'] > threshold,
                AdvantageLabel.POSITIVE,
                AdvantageLabel.NEGATIVE,
            )
        
        return labeled_episodes
    
    def _predict_values(self, episode: dict, params: dict) -> np.ndarray:
        """Require a learned value inference implementation before labeling."""
        raise NotImplementedError(
            "Value inference is not implemented in recap_core; attach learned "
            "value_predictions before computing RECAP advantages"
        )


class AdvantageLabel:
    """优势标签常量"""
    POSITIVE = 1
    NEGATIVE = 0
    UNCONDITIONAL = 2


# ============================================================
# 3. 优势条件策略 (Advantage-Conditioned Policy)
# ============================================================

class AdvantageConditioner:
    """
    将优势标签注入 VLA 的 prefix，实现 advantage conditioning

    核心修改:
    1. 在 VLA prefix (图像+语言 token 序列) 前添加优势条件 token
    2. 训练时 30% 概率随机丢弃优势条件 (dropout)，支持 CFG 推理
    3. 推理时使用 Classifier-Free Guidance (CFG) 增强高优势行为

    数学基础:
    π^(a|o) ∝ π_ref(a|o) · p(I | A^π_ref(o,a))^β

    其中 I 是二值化优势标签, β 控制 guidance 强度

    实际集成方式 (openpi/π0.5):
    - 在 PaliGemma VLM 的 input token 序列最前面插入 1 个 advantage token
    - advantage embedding 使用 negative / positive / null 三个独立状态
    - 训练时在 forward pass 内部做 dropout (非数据预处理阶段)
    - 推理时分别用 advantage=1 和 advantage=0 做两次前向，用 CFG 组合
    """

    def __init__(self, dropout_rate: float = 0.3, rng: Optional[jax.Array] = None):
        self.dropout_rate = dropout_rate
        # 使用外部传入的 RNG 或创建默认 key；训练循环中应在每个 step 传入新的 key
        self._rng = rng if rng is not None else jax.random.PRNGKey(42)

    def step_rng(self) -> jax.Array:
        """Split out a fresh RNG key for each call — avoids fixed-seed dropout."""
        self._rng, subkey = jax.random.split(self._rng)
        return subkey

    def add_advantage_tokens(
        self,
        prefix_tokens: jnp.ndarray,    # (B, L, D) 原始 prefix tokens
        advantage_labels: jnp.ndarray,  # (B,) 优势标签
        train: bool = True,
        rng: Optional[jax.Array] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        在 prefix tokens 前添加优势条件 token

        Args:
            prefix_tokens: 原始 VLA prefix tokens (图像+语言编码)
            advantage_labels: 每个样本的优势标签 (positive/negative)
            train: 是否训练模式（影响 dropout）
            rng: 外部传入的 RNG key (推荐); 不提供时使用内部自增 key

        Returns:
            conditioned_tokens: (B, L+1, D) 添加优势条件后的 tokens
            advantage_mask: (B,) 优势条件是否有效（未 dropout）
        """
        B, L, D = prefix_tokens.shape

        if train:
            # 30% 概率 dropout 优势条件 — 使用非固定 seed
            dropout_key = rng if rng is not None else self.step_rng()
            dropout_mask = jax.random.bernoulli(
                dropout_key,
                p=1 - self.dropout_rate,
                shape=(B,),
            ).astype(jnp.int32)
            effective_labels = apply_condition_dropout(advantage_labels, dropout_mask)
        else:
            effective_labels = advantage_labels
            dropout_mask = jnp.ones(B, dtype=jnp.int32)

        # 将优势标签编码为 embedding
        advantage_emb = self._encode_advantage_label(effective_labels, D)

        # 在 prefix 前插入优势条件 token
        conditioned_tokens = jnp.concatenate(
            [advantage_emb[:, None, :], prefix_tokens], axis=1
        )

        return conditioned_tokens, dropout_mask

    def _encode_advantage_label(
        self, labels: jnp.ndarray, dim: int
    ) -> jnp.ndarray:
        """
        将二值化标签编码为向量 embedding

        注意: 真正的可学习 embedding 在 Flax 模型内部定义 (见 openpi_patches.py
        Pi0WithAdvantage.advantage_embedding)。此方法用于独立测试/原型验证，
        使用正交初始化使 positive/negative 两个 embedding 尽可能不同。
        实际训练时应由模型内部的 nn.Embed 层替代。
        """
        B = labels.shape[0]
        # 使用确定性但不同的初始化 (非学习参数，仅用于原型验证)
        pos_vec = jnp.ones(dim) * 0.5
        neg_vec = jnp.ones(dim) * (-0.5)
        emb = jnp.where(labels[:, None] == 1,
                        jnp.broadcast_to(pos_vec, (B, dim)),
                        jnp.broadcast_to(neg_vec, (B, dim)))
        return emb


# ============================================================
# 4. CFG 策略推理 (Classifier-Free Guidance at Inference)
# ============================================================

class CFGPolicy:
    """
    基于 Classifier-Free Guidance 的策略推理
    
    推理时增强高优势行为:
    π^(a|o,ℓ) ∝ π(a|o,ℓ) · (π(a|o,I=positive,ℓ) / π(a|o,ℓ))^β
    
    在 flow matching 中的实现:
    ∇_a log π^(a|o,ℓ) = ∇_a log π(a|o,ℓ) 
                         + β · (∇_a log π(a|o,I=positive,ℓ) - ∇_a log π(a|o,ℓ))
    """
    
    def __init__(self, beta: float = 1.5):
        self.beta = beta
    
    def guided_flow_step(
        self,
        conditional_velocity: jnp.ndarray,   # ∇_a log π(a|o,I=pos,ℓ)
        unconditional_velocity: jnp.ndarray,  # ∇_a log π(a|o,ℓ)
    ) -> jnp.ndarray:
        """
        CFG 引导的 flow matching 单步
        
        guided_v = unconditional_v + β · (conditional_v - unconditional_v)
        """
        return combine_cfg(unconditional_velocity, conditional_velocity, self.beta)
    
    def sample_with_cfg(
        self,
        policy_model,
        observation: dict,
        num_flow_steps: int = 10,
    ) -> jnp.ndarray:
        """
        使用 CFG 的完整推理流程
        
        对于 flow matching 的每个积分步骤:
        1. 分别计算有条件和无条件的速度场
        2. 用 CFG 公式组合
        3. 更新噪声动作
        """
        # 初始噪声
        noise = jax.random.normal(
            jax.random.PRNGKey(42),
            shape=(1, 10, 7),  # (batch, action_horizon, action_dim) for LIBERO
        )
        
        x = noise
        for step in range(num_flow_steps):
            t = step / num_flow_steps
            
            # 有条件推理 (advantage = positive)
            v_cond = policy_model.predict_velocity(x, t, observation, advantage=1)
            
            # 无条件推理使用独立 null 状态，而不是 negative 标签
            v_uncond = policy_model.predict_velocity(
                x, t, observation, advantage=int(ConditioningState.UNCONDITIONAL)
            )
            
            # CFG 组合
            v_guided = self.guided_flow_step(v_cond, v_uncond)
            
            # Euler 积分步骤
            dt = 1.0 / num_flow_steps
            x = x + v_guided * dt
        
        return x


# ============================================================
# 5. Rollout 采集器 (Rollout Collector)
# ============================================================

@dataclasses.dataclass
class RolloutConfig:
    """Rollout 采集配置"""
    num_episodes: int = 300          # 每轮采集 episode 数
    max_steps_per_episode: int = 400 # 每个 episode 最大步数
    save_every: int = 50             # 每 N 个 episode 保存一次
    num_workers: int = 1             # 并行环境数


class RolloutCollector:
    """
    LIBERO 仿真环境中的策略 rollout 采集
    
    采集内容:
    - 观测 (图像 + 本体感觉状态)
    - 动作
    - 奖励 (稀疏: 成功=+1, 失败=-1, 进行中=0)
    - 语言指令
    - Episode 结果 (成功/失败)
    """
    
    def __init__(self, config: RolloutConfig):
        self.config = config
    
    def collect_rollouts(
        self,
        policy_fn,                      # 策略函数: obs → action
        env,                            # LIBERO 环境
        task_name: str,
    ) -> list:
        """
        用给定策略在环境中采集 rollout 数据
        
        Returns:
            episodes: list of dict, 每个 dict 包含:
                - observations: list of obs
                - actions: list of action
                - rewards: list of reward
                - language: task description
                - success: bool
        """
        episodes = []
        
        for ep_idx in range(self.config.num_episodes):
            obs = env.reset()
            done = False
            step = 0
            
            episode = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'language': task_name,
                'success': False,
            }
            
            while not done and step < self.config.max_steps_per_episode:
                # 策略推理
                action = policy_fn(obs)
                
                # 环境交互
                next_obs, reward, done, info = env.step(action)
                
                episode['observations'].append(obs)
                episode['actions'].append(action)
                episode['rewards'].append(reward)
                
                obs = next_obs
                step += 1
            
            # 标记成功/失败
            episode['success'] = info.get('success', False)

            # 转换奖励为论文格式 (稀疏奖励):
            # 先清零所有步骤的原始奖励，再设终端奖励
            # r_t = 0 for t < T, r_T = +1 (成功) or -1 (失败)
            T = len(episode['rewards'])
            episode['rewards'] = [0.0] * T
            if episode['success']:
                episode['rewards'][-1] = 1.0
            else:
                episode['rewards'][-1] = -1.0
            
            episodes.append(episode)
            
            if (ep_idx + 1) % self.config.save_every == 0:
                print(f"Collected {ep_idx + 1}/{self.config.num_episodes} episodes")
        
        return episodes


# ============================================================
# 6. RECAP 训练循环 (Main Training Loop)
# ============================================================

class RECAPTrainer:
    """
    RECAP 完整训练循环
    
    对应论文 Algorithm 1:
    1. 在演示数据上 SFT 训练 base policy
    2. 用 base policy 采集 rollout
    3. 训练分布式价值函数
    4. 计算优势标签
    5. 用优势条件训练改进策略
    6. 可选: 迭代多轮
    """
    
    def __init__(
        self,
        value_fn_config: ValueFunctionConfig,
        advantage_config: AdvantageConfig,
        cfg_beta: float = 1.5,
        advantage_dropout_rate: float = 0.3,
    ):
        self.value_fn = DistributionalValueFunction(value_fn_config)
        self.advantage_computer = AdvantageComputer(advantage_config, self.value_fn)
        self.advantage_conditioner = AdvantageConditioner(advantage_dropout_rate)
        self.cfg_policy = CFGPolicy(cfg_beta)
    
    def run_iteration(
        self,
        demo_data: list,
        policy_model,
        env,
        task_names: list,
        iteration: int = 0,
    ) -> dict:
        """
        执行一轮 RECAP 迭代
        
        Args:
            demo_data: 人类演示数据
            policy_model: 当前策略模型
            env: 仿真环境
            task_names: 任务名称列表
            iteration: 当前迭代编号
            
        Returns:
            results: 包含改进策略和评测结果的字典
        """
        print(f"\n{'='*60}")
        print(f"RECAP Iteration {iteration}")
        print(f"{'='*60}")
        
        # Step 1: 采集自主 rollout
        print("[1/4] Collecting autonomous rollouts...")
        rollout_config = RolloutConfig(num_episodes=300)
        collector = RolloutCollector(rollout_config)
        
        all_rollouts = []
        for task in task_names:
            rollouts = collector.collect_rollouts(
                policy_fn=policy_model.infer,
                env=env,
                task_name=task,
            )
            all_rollouts.extend(rollouts)
        
        # Step 2: 训练价值函数
        print("[2/4] Training value function...")
        combined_data = demo_data + all_rollouts
        # value_fn_params = self._train_value_function(combined_data)
        
        # Step 3: 计算优势标签
        print("[3/4] Computing advantage labels...")
        labeled_data = self.advantage_computer.compute_advantages_for_dataset(
            combined_data, value_fn_params=None  # placeholder
        )
        
        # Step 4: 训练优势条件策略
        print("[4/4] Training advantage-conditioned policy...")
        # improved_policy = self._train_advantage_conditioned_policy(
        #     policy_model, labeled_data
        # )
        
        # 评测
        print("Evaluating improved policy...")
        # eval_results = self._evaluate(improved_policy, env, task_names)
        
        return {
            'rollouts': all_rollouts,
            'labeled_data': labeled_data,
            # 'improved_policy': improved_policy,
            # 'eval_results': eval_results,
        }


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("RECAP Implementation for π*0.6 Reproduction")
    print("=" * 60)
    print()
    print("Phase 1: Based on openpi (π0.5) + LIBERO simulation")
    print()
    
    # 配置
    vf_config = ValueFunctionConfig()
    adv_config = AdvantageConfig()
    
    trainer = RECAPTrainer(
        value_fn_config=vf_config,
        advantage_config=adv_config,
        cfg_beta=1.5,
        advantage_dropout_rate=0.3,
    )
    
    print("Configuration:")
    print(f"  Value function bins: {vf_config.num_value_bins}")
    print(f"  N-step lookahead: {adv_config.n_step}")
    print(f"  Advantage threshold: {adv_config.advantage_threshold_pct:.0%} percentile")
    print(f"  CFG beta: 1.5")
    print(f"  Advantage dropout: 30%")
    print()
    print("Ready for Phase 1A: Environment setup + SFT baseline")
