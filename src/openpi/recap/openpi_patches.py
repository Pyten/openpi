"""
openpi 源码补丁 - RECAP Advantage Conditioning
用于修改 π0.5 架构以支持优势条件策略训练和 CFG 推理

应用方式:
  cd openpi
  git apply recap_advantage.patch
  # 或手动将以下代码集成到对应文件

硬件: NVIDIA A800 80GB
作者: RECAP Reproduction Project

注意: 本文件提供可直接导入的 Python 类 (非字符串)。
      实际部署时需要对照 openpi 源码适配具体 API。
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field


# ============================================================
# 补丁 1: 带优势条件的 π0.5 策略类
# 目标文件: src/openpi/models/pi0_with_advantage.py
# ============================================================

class Pi0WithAdvantage(nn.Module):
    """
    π0.5 Flow Matching Policy + RECAP Advantage Conditioning

    在原始 π0.5 的 VLA prefix 前插入 advantage token:
    - advantage=1 (positive): 编码高优势行为的 embedding
    - advantage=0 (negative/unconditional): 编码一般行为的 embedding

    训练时 30% dropout → 推理时支持 CFG

    集成方式:
      本类包装 openpi 的原始 π0.5 模型 (通过 base_policy 参数),
      在 embed_prefix() 阶段注入 advantage token,
      其余 flow matching 逻辑完全委托给 base_policy。
    """

    # --- 配置 ---
    hidden_dim: int = 2048          # Gemma-2B hidden dim
    num_advantage_classes: int = 2  # positive/negative
    advantage_dropout_rate: float = 0.30
    num_flow_steps: int = 10
    action_dim: int = 7
    action_horizon: int = 10

    # --- base policy (运行时注入) ---
    # 注意: Flax Module 不能直接持有另一个 Module 作为属性,
    # 需要通过方法参数传入或使用 nn.compact 子模块。
    # 这里用 setup() 初始化。

    def setup(self):
        """初始化优势 embedding 层"""
        self.advantage_embedding = nn.Embed(
            num_embeddings=self.num_advantage_classes,
            features=self.hidden_dim,
            name="advantage_embedding",
        )

    def inject_advantage_token(
        self,
        prefix_tokens: jnp.ndarray,
        advantage_labels: Optional[jnp.ndarray] = None,
        train: bool = True,
    ) -> jnp.ndarray:
        """
        在 prefix tokens 前插入 advantage token

        Args:
            prefix_tokens: (B, L, D) 图像+语言编码的 prefix tokens
            advantage_labels: (B,) 优势标签 {0, 1}
            train: 训练/推理模式

        Returns:
            conditioned_prefix: (B, L+1, D) 带 advantage token 的 prefix
        """
        B = prefix_tokens.shape[0]

        if advantage_labels is not None:
            if train:
                # 30% dropout: 随机将优势标签设为 0 (unconditional)
                dropout_mask = jax.random.bernoulli(
                    self.make_rng("dropout"),
                    p=1.0 - self.advantage_dropout_rate,
                    shape=(B,),
                ).astype(jnp.int32)
                effective_labels = advantage_labels * dropout_mask
            else:
                effective_labels = advantage_labels
        else:
            # 无优势标签 → 默认 unconditional
            effective_labels = jnp.zeros(B, dtype=jnp.int32)

        # 获取优势 embedding
        adv_tokens = self.advantage_embedding(effective_labels)  # (B, D)

        # 在 prefix 前插入优势 token
        conditioned_prefix = jnp.concatenate(
            [adv_tokens[:, None, :], prefix_tokens], axis=1
        )  # (B, L+1, D)

        return conditioned_prefix

    def __call__(
        self,
        prefix_tokens: jnp.ndarray,
        actions: Optional[jnp.ndarray] = None,
        advantage_labels: Optional[jnp.ndarray] = None,
        train: bool = True,
        base_policy: Optional[Any] = None,
    ):
        """
        Args:
            prefix_tokens: (B, L, D) 图像+语言编码的 prefix tokens
            actions: (B, H, D_act) 目标动作序列 (训练时)
            advantage_labels: (B,) 优势标签 {0, 1}
            train: 训练/推理模式
            base_policy: openpi 的原始 π0.5 模型实例

        Returns:
            训练: flow matching loss (scalar)
            推理: 生成的动作 (B, H, D_act)
        """
        # 注入 advantage token
        conditioned_prefix = self.inject_advantage_token(
            prefix_tokens, advantage_labels, train
        )

        # 委托给 base policy 的 flow matching 实现
        if base_policy is None:
            raise ValueError(
                "base_policy is required. Pass the original π0.5 model instance."
            )

        if actions is not None:
            # 训练模式: 计算 flow matching loss
            # 调用 base_policy 的 flow matching loss, 使用 conditioned_prefix
            return self._flow_matching_loss(
                base_policy, conditioned_prefix, actions
            )
        else:
            # 推理模式: 生成动作
            return self._flow_matching_sample(
                base_policy, conditioned_prefix
            )

    def _flow_matching_loss(
        self,
        base_policy: Any,
        prefix: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Flow matching 训练损失

        委托给 base_policy 的 flow matching 实现。
        openpi 的 π0.5 使用 conditional flow matching:
          1. 采样噪声 ε ~ N(0, I)
          2. 插值: x_t = (1-t) * ε + t * actions
          3. 预测速度: v_θ(x_t, t, prefix)
          4. 损失: ||v_θ - (actions - ε)||²

        Args:
            base_policy: openpi 的原始 π0.5 模型
            prefix: (B, L+1, D) 带 advantage token 的 prefix
            actions: (B, H, D_act) 目标动作

        Returns:
            loss: scalar, flow matching MSE loss
        """
        # TODO: 对照 openpi 源码实现
        # 参考: src/openpi/models/pi0.py 中的 flow matching 实现
        # 关键: 使用 prefix 替代原始 prefix, 其余逻辑不变
        #
        # 伪代码:
        # B, H, D_act = actions.shape
        # ε = jax.random.normal(self.make_rng("noise"), actions.shape)
        # t = jax.random.uniform(self.make_rng("time"), (B, 1, 1))
        # x_t = (1 - t) * ε + t * actions
        # target_velocity = actions - ε
        # predicted_velocity = base_policy.predict_velocity(
        #     x_t, t, prefix, train=True
        # )
        # loss = jnp.mean((predicted_velocity - target_velocity) ** 2)
        # return loss

        raise NotImplementedError(
            "Flow matching loss must be implemented by integrating with "
            "openpi's actual π0.5 model. See src/openpi/models/pi0.py "
            "for the reference implementation."
        )

    def _flow_matching_sample(
        self,
        base_policy: Any,
        prefix: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Flow matching 采样 (Euler 积分)

        从噪声逐步积分到动作:
          x_0 ~ N(0, I)
          for step in range(num_flow_steps):
              t = step / num_flow_steps
              v = base_policy.predict_velocity(x_t, t, prefix)
              x_{t+1} = x_t + v * dt

        Args:
            base_policy: openpi 的原始 π0.5 模型
            prefix: (B, L+1, D) 带 advantage token 的 prefix

        Returns:
            actions: (B, H, D_act) 生成的动作
        """
        # TODO: 对照 openpi 源码实现
        # 参考: src/openpi/models/pi0.py 中的采样实现
        #
        # 伪代码:
        # B = prefix.shape[0]
        # x = jax.random.normal(
        #     self.make_rng("sample"), (B, self.action_horizon, self.action_dim)
        # )
        # dt = 1.0 / self.num_flow_steps
        # for step in range(self.num_flow_steps):
        #     t = jnp.full((B, 1, 1), step / self.num_flow_steps)
        #     v = base_policy.predict_velocity(x, t, prefix, train=False)
        #     x = x + v * dt
        # return x

        raise NotImplementedError(
            "Flow matching sampling must be implemented by integrating with "
            "openpi's actual π0.5 model. See src/openpi/models/pi0.py "
            "for the reference implementation."
        )


# ============================================================
# 补丁 2: CFG 推理服务
# 目标文件: src/openpi/scripts/serve_policy_cfg.py
# ============================================================

@dataclass
class CFGServerConfig:
    """CFG 推理服务配置"""
    cfg_beta: float = 2.0
    num_flow_steps: int = 10
    action_dim: int = 7
    action_horizon: int = 10
    port: int = 8000


class CFGPolicyServer:
    """
    支持 CFG 推理的策略服务

    基于 openpi 的 serve_policy.py, 新增:
    1. 同时维护 conditional 和 unconditional 两条推理路径
    2. 用 CFG 公式组合速度场
    3. 支持 β 参数动态调整

    启动方式:
    uv run scripts/serve_policy_cfg.py \
        --config pi05_libero \
        --cfg-beta 2.0 \
        --port 8000

    注意: 实际部署时需要对照 openpi 的 serve_policy.py 适配:
    - WebSocket 通信协议
    - 观测预处理 (图像编码、状态归一化)
    - 动作后处理 (去归一化)
    """

    def __init__(
        self,
        policy: Pi0WithAdvantage,
        base_policy: Any,  # openpi 的原始 π0.5 模型
        config: Optional[CFGServerConfig] = None,
    ):
        self.policy = policy
        self.base_policy = base_policy
        self.config = config or CFGServerConfig()

    def predict_action_with_cfg(
        self,
        observation: Dict[str, np.ndarray],
        beta: Optional[float] = None,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> np.ndarray:
        """
        CFG 推理流程

        对于 flow matching 的每个积分步骤:
        1. 条件推理: v_cond = policy(x_t, t, obs, advantage=1)
        2. 无条件推理: v_uncond = policy(x_t, t, obs, advantage=0)
        3. CFG 组合: v = v_uncond + β * (v_cond - v_uncond)
        4. Euler 步: x_{t+1} = x_t + v * dt

        Args:
            observation: 观测字典, 包含 'image', 'state', 'language' 等
            beta: CFG 强度, 默认使用 config.cfg_beta
            rng: JAX 随机 key

        Returns:
            action: (H, D_act) 生成的动作
        """
        beta = beta or self.config.cfg_beta
        action_shape = (self.config.action_horizon, self.config.action_dim)

        if rng is None:
            rng = jax.random.PRNGKey(0)

        # 初始噪声
        rng, noise_key = jax.random.split(rng)
        x = jax.random.normal(noise_key, action_shape)
        dt = 1.0 / self.config.num_flow_steps

        # 预处理观测 (需要对照 openpi 的预处理逻辑)
        # TODO: 调用 base_policy 的 preprocess_observation()
        processed_obs = observation  # placeholder

        for step in range(self.config.num_flow_steps):
            t = jnp.full((1, 1, 1), step / self.config.num_flow_steps)

            # 条件推理 (advantage = positive)
            v_cond = self._predict_velocity(
                x, t, processed_obs, advantage_label=1
            )

            # 无条件推理 (advantage = negative/unconditional)
            v_uncond = self._predict_velocity(
                x, t, processed_obs, advantage_label=0
            )

            # CFG 组合
            v_guided = v_uncond + beta * (v_cond - v_uncond)

            # Euler 积分
            x = x + v_guided * dt

        return np.asarray(x)

    def _predict_velocity(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        observation: Dict,
        advantage_label: int,
    ) -> jnp.ndarray:
        """
        预测速度场

        需要对照 openpi 的 base_policy 实现:
        1. 编码观测 → prefix tokens
        2. 注入 advantage token
        3. 调用 base_policy 的速度预测网络

        Args:
            x_t: (H, D_act) 当前噪声动作
            t: (1, 1, 1) 当前时间步
            observation: 预处理后的观测
            advantage_label: 0 或 1

        Returns:
            velocity: (H, D_act) 预测的速度
        """
        # TODO: 对照 openpi 源码实现
        # 伪代码:
        # 1. prefix = base_policy.encode_observation(observation)
        # 2. B = 1, prefix = prefix[None, ...]  # add batch dim
        # 3. advantage_labels = jnp.array([advantage_label])
        # 4. conditioned_prefix = policy.inject_advantage_token(
        #     prefix, advantage_labels, train=False
        # )
        # 5. velocity = base_policy.predict_velocity(
        #     x_t[None, ...], t, conditioned_prefix, train=False
        # )
        # 6. return velocity[0]  # remove batch dim

        raise NotImplementedError(
            "Velocity prediction must be implemented by integrating with "
            "openpi's actual π0.5 model. The key steps are:\n"
            "1. Encode observation using base_policy's encoder\n"
            "2. Inject advantage token using policy.inject_advantage_token()\n"
            "3. Call base_policy's velocity prediction network\n"
            "See src/openpi/models/pi0.py for the reference implementation."
        )


# ============================================================
# 补丁 3: 训练数据 pipeline
# 目标文件: src/openpi/data/recap_dataset.py
# ============================================================

@dataclass
class RECAPDataConfig:
    """RECAP 数据处理配置"""
    advantage_threshold: float = 0.40
    dropout_rate: float = 0.30
    n_step: int = 50
    gamma: float = 1.0  # 折扣因子


class RECAPDataPipeline:
    """
    RECAP 数据处理流水线

    将 rollout 数据与 demo 数据合并, 附加优势标签
    输出格式兼容 openpi 的 LeRobot 数据加载器

    修复: 优势计算现在使用 gamma 折扣因子
    """

    def __init__(self, config: Optional[RECAPDataConfig] = None):
        self.config = config or RECAPDataConfig()

    def merge_datasets(
        self,
        demo_data: List[Dict],
        rollout_data: List[Dict],
        correction_data: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        合并演示+自主+纠错数据 (论文 Algorithm 1)

        D = D_demo ∪ D_autonomous ∪ D_correction
        """
        combined = list(demo_data) + list(rollout_data)
        if correction_data:
            combined.extend(correction_data)
        return combined

    def label_advantages(
        self,
        episodes: List[Dict],
        value_fn_predict: Callable,
    ) -> List[Dict]:
        """
        为每个 timestep 计算并标注优势标签

        流程:
        1. 用价值函数预测每个 timestep 的 V(o_t, ℓ)
        2. 计算 N-step 优势 A_t (带 gamma 折扣)
        3. 按阈值二值化: positive/negative

        修复: 之前版本缺少 gamma 折扣
        """
        n_step = self.config.n_step
        gamma = self.config.gamma
        all_advantages = []

        for ep in episodes:
            T = len(ep['rewards'])
            rewards = np.array(ep['rewards'])
            value_preds = np.array([
                value_fn_predict(obs) for obs in ep['observations']
            ])

            # N-step advantage (带 gamma 折扣)
            advantages = np.zeros(T, dtype=np.float32)
            for t in range(T):
                # 计算折扣 N-step 回报
                n_return = 0.0
                for k in range(min(n_step, T - t)):
                    n_return += (gamma ** k) * rewards[t + k]

                # Bootstrap 价值 (如果 t+N < T)
                if t + n_step < T:
                    bootstrap = (gamma ** n_step) * value_preds[t + n_step]
                else:
                    bootstrap = 0.0

                advantages[t] = n_return + bootstrap - value_preds[t]

            ep['advantages'] = advantages
            ep['value_predictions'] = value_preds
            all_advantages.extend(advantages.tolist())

        # 全局二值化
        all_advantages = np.array(all_advantages)
        threshold = np.percentile(
            all_advantages, (1 - self.config.advantage_threshold) * 100
        )

        for ep in episodes:
            ep['advantage_labels'] = np.where(
                ep['advantages'] > threshold, 1, 0
            ).tolist()

        pos_ratio = np.mean(all_advantages > threshold)
        print(f"  Labeled {len(episodes)} episodes")
        print(f"  Positive ratio: {pos_ratio:.2%} "
              f"(target: {self.config.advantage_threshold:.0%})")

        return episodes

    def create_training_samples(
        self,
        labeled_episodes: List[Dict],
        apply_dropout: bool = True,
    ) -> List[Dict]:
        """
        从标注后的 episodes 创建训练样本

        每个样本格式 (兼容 LeRobot):
        {
            'observation': {'image': ..., 'state': ...},
            'action': ...,
            'language': ...,
            'advantage_label': 0 or 1,
        }

        训练时 30% dropout:
        - 每个 sample 以 30% 概率将 advantage_label 设为 0
        """
        samples = []

        for ep in labeled_episodes:
            T = len(ep['observations'])
            for t in range(T - 1):
                label = ep['advantage_labels'][t]

                # 30% dropout
                if apply_dropout and np.random.random() < self.config.dropout_rate:
                    label = 0  # 设为 unconditional

                samples.append({
                    'observation': ep['observations'][t],
                    'action': ep['actions'][t],
                    'language': ep['language'],
                    'advantage_label': label,
                })

        return samples


# ============================================================
# 补丁应用说明
# ============================================================

PATCH_INSTRUCTIONS = """
openpi RECAP 补丁应用指南 (A800)
===================================

1. 克隆 openpi:
   git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
   cd openpi
   GIT_LFS_SKIP_SMUDGE=1 uv sync
   GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

2. 应用补丁:

   方式 A - 手动集成 (推荐):
   - 将 Pi0WithAdvantage 类添加到 src/openpi/models/pi0_with_advantage.py
   - 将 CFGPolicyServer 添加到 src/openpi/scripts/serve_policy_cfg.py
   - 将 RECAPDataPipeline 添加到 src/openpi/data/recap_dataset.py

   方式 B - 自动补丁:
   - patch -p1 < recap_advantage.patch

3. 下载权重:
   uv run scripts/download_checkpoints.py

4. 计算 LIBERO 统计:
   uv run scripts/compute_norm_stats.py --config-name pi05_libero

5. 开始训练:
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 uv run scripts/train.py pi05_libero --exp-name recap_sft_baseline --overwrite

6. 显存使用估算 (A800 80GB):
   ┌─────────────────────────┬──────────┬──────────┐
   │ 模型                    │ 参数量   │ VRAM     │
   ├─────────────────────────┼──────────┼──────────┤
   │ π0.5 全量微调           │ 3.3B     │ ~52GB    │
   │ π0.5 LoRA微调           │ 3.3B+LoRA│ ~22GB    │
   │ 价值函数 (Gemma-2B)     │ ~670M    │ ~12GB    │
   │ 价值函数 + 策略模型(推理)│ ~4B      │ ~18GB    │
   └─────────────────────────┴──────────┴──────────┘

   A800 单卡策略:
   - 全量微调和价值函数训练分时使用 (不同时训练)
   - Rollout 采集: 加载策略模型推理 (~6GB), 剩余供环境
   - RECAP训练: 全量微调 ~52GB, 剩余 ~28GB buffer

7. 待实现 (需要对照 openpi 源码):
   - Pi0WithAdvantage._flow_matching_loss(): 参考 src/openpi/models/pi0.py
   - Pi0WithAdvantage._flow_matching_sample(): 参考 src/openpi/models/pi0.py
   - CFGPolicyServer._predict_velocity(): 集成 base_policy 的速度预测
   - 观测预处理: 对照 openpi 的 preprocess_observation()
"""


if __name__ == "__main__":
    print(PATCH_INSTRUCTIONS)
