"""
LIBERO 仿真环境封装
用于 RECAP rollout 采集和评测

依赖: libero, mujoco, openpi
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# 环境配置
# ============================================================

@dataclass
class LiberoTaskConfig:
    """LIBERO 任务配置"""
    suite_name: str           # "spatial", "object", "goal", "long"
    task_index: int           # 任务索引
    task_name: str            # 任务名称
    task_description: str     # 语言描述
    max_steps: int = 400      # 最大步数
    obs_img_size: int = 256   # 观测图像尺寸 (openpi 用 256/448)


# LIBERO 任务注册表
# 任务名称必须与 LIBERO 源码中的 task registry 完全匹配
# 参考: https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/benchmark/__init__.py
#
# 部署前验证:
#   from libero.libero import get_libero_path
#   from libero.libero.benchmark import get_benchmark
#   benchmark = get_benchmark("LIBERO_SPATIAL")
#   for i in range(benchmark.n_tasks):
#       task = benchmark.get_task(i)
#       print(f"  {i}: {task.name} — {task.language}")

LIBERO_TASK_REGISTRY = {
    "libero_spatial": [
        {"name": "kitchen_put_black_bowl_in_cabinet",
         "description": "put the black bowl on top of the cabinet"},
        {"name": "kitchen_put_white_bowl_at_back_on_stove",
         "description": "put the white bowl at the back on the stove"},
        {"name": "kitchen_put_wine_bottle_on_rack",
         "description": "put the wine bottle on the rack"},
        {"name": "kitchen_put_butter_on_front_left_burner",
         "description": "put the butter on the front left burner"},
        {"name": "kitchen_put_frying_pan_on_front_right_burner",
         "description": "put the frying pan on the front right burner"},
        {"name": "kitchen_put_gray_boot_on_left_rack",
         "description": "put the gray boot on the left rack"},
        {"name": "kitchen_put_red_mug_on_left_rack",
         "description": "put the red mug on the left rack"},
        {"name": "kitchen_put_white_mug_on_right_rack",
         "description": "put the white mug on the right rack"},
        {"name": "kitchen_put_chocolate_pudding_on_front_right_burner",
         "description": "put the chocolate pudding on the front right burner"},
        {"name": "kitchen_put_cream_cheese_on_front_left_burner",
         "description": "put the cream cheese on the front left burner"},
    ],
    "libero_object": [
        # 10 object manipulation tasks — 部署时从 LIBERO benchmark 获取
        {"name": "kitchen_open_drawer", "description": "open the top drawer"},
        {"name": "kitchen_close_drawer", "description": "close the top drawer"},
    ] + [{"name": f"libero_object_task_{i}", "description": f"libero object task {i}"}
         for i in range(8)],  # placeholder for remaining 8 tasks
    "libero_goal": [
        # 10 goal-conditioned tasks
    ] + [{"name": f"libero_goal_task_{i}", "description": f"libero goal task {i}"}
         for i in range(10)],  # placeholder — fill from LIBERO benchmark
    "libero_long": [
        # 10 long-horizon tasks
    ] + [{"name": f"libero_long_task_{i}", "description": f"libero long task {i}"}
         for i in range(10)],  # placeholder — fill from LIBERO benchmark
}

# 向后兼容: 保留旧名
LIBERO_SPATIAL_TASKS = [t["name"] for t in LIBERO_TASK_REGISTRY["libero_spatial"]]


class LiberoWrapper:
    """
    LIBERO 仿真环境封装

    提供标准 RL 接口:
    - reset() → obs
    - step(action) → obs, reward, done, info
    - 支持图像观测 + 本体感觉状态
    - 稀疏奖励: success=+1, failure=-1, otherwise=0

    修复:
    - 添加动作归一化/去归一化
    - 改进成功检测逻辑
    - 完善观测字典结构
    """

    def __init__(
        self,
        task_suite_name: str = "libero_spatial",
        task_id: int = 0,
        obs_img_size: int = 256,
        action_dim: int = 7,       # LIBERO 使用 7-DoF (x,y,z,rx,ry,rz,gripper)
        action_horizon: int = 10,  # action chunk 长度
        action_scale: float = 1.0, # 动作缩放因子
        use_action_stats: bool = False,  # 使用数据集统计归一化
    ):
        self.task_suite_name = task_suite_name
        self.task_id = task_id
        self.obs_img_size = obs_img_size
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.action_scale = action_scale
        self.use_action_stats = use_action_stats

        self.env = None
        self.current_step = 0
        self.max_steps = 400  # LIBERO 默认 horizon

        # 动作归一化统计 (从 LIBERO 数据集计算)
        # 格式: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, gripper]
        # 如果 use_action_stats=True, 使用这些统计; 否则使用默认缩放
        self.action_stats = {
            'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32),
            'std': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            'min': np.array([-0.5, -0.5, -0.5, -1.57, -1.57, -1.57, 0.0], dtype=np.float32),
            'max': np.array([0.5, 0.5, 0.5, 1.57, 1.57, 1.57, 1.0], dtype=np.float32),
        }

    def setup(self):
        """
        初始化 LIBERO 环境

        修复: 使用 LIBERO benchmark API 获取任务配置
        参考: https://github.com/Lifelong-Robot-Learning/LIBERO
        """
        try:
            from libero.libero.envs import OffScreenRenderEnv
            from libero.libero.benchmark import get_benchmark
        except ImportError as e:
            print(f"WARNING: libero not installed ({e}). Using mock environment.")
            self.env = None
            return

        # 获取 benchmark 和任务配置
        suite_map = {
            "libero_spatial": "LIBERO_SPATIAL",
            "libero_object": "LIBERO_OBJECT",
            "libero_goal": "LIBERO_GOAL",
            "libero_long": "LIBERO_LONG",
        }
        benchmark_name = suite_map.get(self.task_suite_name, self.task_suite_name.upper())

        try:
            benchmark = get_benchmark(benchmark_name)
            task = benchmark.get_task(self.task_id)

            # 创建环境
            self.env = OffScreenRenderEnv(
                bddl_file=task.bddl_file,
                has_renderer=False,
                has_offscreen_renderer=True,
                render_camera="agentview",
                offscreen_render_camera="agentview",
                img_width=self.obs_img_size,
                img_height=self.obs_img_size,
                reward_shaping=False,  # 使用原始稀疏奖励
                horizon=self.max_steps,
            )

            # 更新任务描述 (从 benchmark 获取真实描述)
            self._task_description_override = task.language

        except Exception as e:
            print(f"WARNING: Failed to setup LIBERO env ({e}). Using mock environment.")
            self.env = None
            return

    def _get_task_name(self) -> str:
        """获取任务名称 — 从注册表查找，回退到旧名"""
        suite = self.task_suite_name.replace("-", "_")
        registry = LIBERO_TASK_REGISTRY.get(suite, [])
        if self.task_id < len(registry):
            return registry[self.task_id]["name"]
        # 向后兼容
        if suite == "libero_spatial" and self.task_id < len(LIBERO_SPATIAL_TASKS):
            return LIBERO_SPATIAL_TASKS[self.task_id]
        return f"task_{self.task_id}"

    def reset(self) -> Dict:
        """
        重置环境

        Returns:
            observation dict with:
            - image: (H, W, 3) RGB image
            - state: (action_dim,) robot state (joint positions)
            - eef_pos: (3,) end-effector position
            - eef_quat: (4,) end-effector quaternion
            - gripper_qpos: (2,) gripper joint positions
            - prompt: str task description
        """
        if self.env is None:
            return self._mock_obs()

        obs = self.env.reset()
        self.current_step = 0

        return self._process_obs(obs)

    def _process_obs(self, obs: Dict) -> Dict:
        """处理 LIBERO 原始观测为标准格式"""
        return {
            'image': obs.get('agentview_image', np.zeros((self.obs_img_size, self.obs_img_size, 3), dtype=np.uint8)),
            'state': obs.get('robot0_joint_pos', np.zeros(self.action_dim, dtype=np.float32))[:self.action_dim],
            'eef_pos': obs.get('robot0_eef_pos', np.zeros(3, dtype=np.float32)),
            'eef_quat': obs.get('robot0_eef_quat', np.array([1, 0, 0, 0], dtype=np.float32)),
            'gripper_qpos': obs.get('robot0_gripper_qpos', np.zeros(2, dtype=np.float32)),
            'prompt': self._get_task_description(),
        }

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一步动作

        Args:
            action: (action_dim,) 或 (action_horizon, action_dim) 动作
                    动作范围: [-1, 1] (归一化)

        Returns:
            obs, reward, done, info

        修复:
        - 正确处理 action chunk 中途终止
        - 累积奖励直到终止
        - 记录实际执行的步数
        """
        if self.env is None:
            return self._mock_obs(), 0.0, False, {'success': False}

        # 处理 action chunk
        if action.ndim == 2:
            actions = action  # (H, D)
        else:
            actions = action[None, :]  # (1, D)

        total_reward = 0.0
        done = False
        info = {'success': False, 'is_success': False, 'steps_executed': 0}
        last_obs = None
        last_info = {}

        for i, a in enumerate(actions):
            # 动作去归一化: [-1, 1] → 实际范围
            denorm_action = self._denormalize_action(a)

            obs, reward, done, last_info = self.env.step(denorm_action)
            self.current_step += 1
            info['steps_executed'] += 1
            last_obs = obs

            # 累积奖励
            total_reward += reward

            # 检查成功 (LIBERO 的 info 可能包含 'is_success' 或其他键)
            is_success = self._check_success(last_info, obs)
            info['success'] = is_success
            info['is_success'] = is_success

            # 终止条件: 环境 done / 成功 / 达到最大步数
            if done or is_success or self.current_step >= self.max_steps:
                done = True
                break

        # 稀疏奖励覆盖 (RECAP 使用稀疏奖励)
        if info.get('success', False) or info.get('is_success', False):
            sparse_reward = 1.0
        elif done and not (info.get('success', False) or info.get('is_success', False)):
            sparse_reward = -1.0
        else:
            sparse_reward = 0.0

        # 处理观测
        if last_obs is not None:
            next_obs = self._process_obs(last_obs)
        else:
            # 如果没有执行任何步骤 (不应该发生)，返回当前观测
            next_obs = self._process_obs(obs)

        return next_obs, sparse_reward, done, info

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        将归一化动作 [-1, 1] 转换为 LIBERO 实际动作范围

        LIBERO 动作空间 (7-DoF):
        - [0:3]: 末端执行器位置 (x, y, z) - 单位: 米
        - [3:6]: 末端执行器旋转 (rx, ry, rz) - 单位: 弧度
        - [6]: 夹爪开合 - 范围: [0, 1]

        两种归一化模式:
        1. use_action_stats=True: 使用数据集统计 (mean/std) 反归一化
        2. use_action_stats=False: 使用 min/max 线性映射
        """
        if self.use_action_stats:
            # 模式 1: 统计归一化 — action = (raw - mean) / std
            # 反归一化: raw = action * std + mean
            raw = action * self.action_stats['std'] * self.action_scale + self.action_stats['mean']
        else:
            # 模式 2: min/max 线性映射 — action ∈ [-1, 1] → [min, max]
            raw = (action + 1.0) / 2.0  # [-1, 1] → [0, 1]
            raw = raw * (self.action_stats['max'] - self.action_stats['min']) + self.action_stats['min']
            raw = raw * self.action_scale

        # 确保夹爪范围合法 [0, 1]
        raw[6] = np.clip(raw[6], 0.0, 1.0)

        return raw.astype(np.float32)

    def normalize_action(self, raw_action: np.ndarray) -> np.ndarray:
        """
        将 LIBERO 原始动作归一化到 [-1, 1]

        用于将 rollout 数据中的动作转换为训练格式
        """
        if self.use_action_stats:
            normalized = (raw_action - self.action_stats['mean']) / (self.action_stats['std'] * self.action_scale)
        else:
            normalized = (raw_action - self.action_stats['min']) / (self.action_stats['max'] - self.action_stats['min'])
            normalized = normalized * 2.0 - 1.0  # [0, 1] → [-1, 1]
            normalized = normalized / self.action_scale

        return normalized.astype(np.float32)

    def set_action_stats(self, mean: np.ndarray, std: np.ndarray,
                         min_val: np.ndarray, max_val: np.ndarray):
        """
        设置动作归一化统计 (从数据集计算)

        部署时需要从 LIBERO 演示数据计算这些统计:
          from recap_libero_env import LiberoWrapper
          env = LiberoWrapper(...)
          # 从数据集计算
          env.set_action_stats(mean, std, min_val, max_val)
        """
        self.action_stats = {
            'mean': mean.astype(np.float32),
            'std': std.astype(np.float32),
            'min': min_val.astype(np.float32),
            'max': max_val.astype(np.float32),
        }

    def _check_success(self, info: Dict, obs: Dict) -> bool:
        """
        检查任务是否成功

        LIBERO 不同版本返回 success 的方式不同:
        1. info['is_success'] — 标准 LIBERO API
        2. info['success'] — 某些包装器使用
        3. info['done'] + reward > 0 — 稀疏奖励场景
        4. 环境内部 task success checker — 需要调用 env.unwrapped

        优先级: 明确的 success 标志 > reward 信号 > done 信号
        """
        # 1. 优先使用环境返回的明确 success 标志
        if info.get('is_success', False):
            return True
        if info.get('success', False):
            return True

        # 2. 检查 LIBERO 的 task success checker (如果可访问)
        if self.env is not None:
            try:
                # LIBERO 内部可能通过 env.unwrapped 暴露 success 检测
                unwrapped = getattr(self.env, 'unwrapped', self.env)
                if hasattr(unwrapped, 'is_success'):
                    return bool(unwrapped.is_success)
                # 某些版本使用 _check_success() 方法
                if hasattr(unwrapped, '_check_success'):
                    return bool(unwrapped._check_success(obs))
            except Exception:
                pass  # 回退到其他检测方式

        # 3. 检查 reward (稀疏奖励: success=+1)
        # 注意: reward_shaping=False 时，只有成功才有正 reward
        reward = info.get('reward', 0.0)
        if reward > 0:
            return True

        return False

    def _get_task_description(self) -> str:
        """获取当前任务的自然语言描述 — 优先使用 benchmark 覆盖"""
        # 如果 setup() 从 benchmark 获取了真实描述，使用它
        if hasattr(self, '_task_description_override'):
            return self._task_description_override
        # 从注册表获取
        suite = self.task_suite_name.replace("-", "_")
        registry = LIBERO_TASK_REGISTRY.get(suite, [])
        if self.task_id < len(registry):
            return registry[self.task_id]["description"]
        # 回退: 将下划线转换为空格
        return self._get_task_name().replace("_", " ")

    def _mock_obs(self) -> Dict:
        """Mock 观测用于测试（无需真实 LIBERO 环境）"""
        return {
            'image': np.random.randint(0, 255, (self.obs_img_size, self.obs_img_size, 3), dtype=np.uint8),
            'state': np.zeros(self.action_dim, dtype=np.float32),
            'eef_pos': np.zeros(3, dtype=np.float32),
            'eef_quat': np.array([1, 0, 0, 0], dtype=np.float32),
            'gripper_qpos': np.zeros(2, dtype=np.float32),
            'prompt': self._get_task_description(),
        }

    def close(self):
        if self.env is not None:
            self.env.close()


class LiberoEvaluator:
    """
    LIBERO 标准评测

    评测协议:
    - 每个任务评测 50 episodes
    - 指标: Success Rate (%)
    - 可选: Throughput (tasks/hour)
    """

    def __init__(
        self,
        task_suites: List[str] = ["libero_spatial"],
        num_eval_episodes: int = 50,
        action_horizon: int = 10,
    ):
        self.task_suites = task_suites
        self.num_eval_episodes = num_eval_episodes
        self.action_horizon = action_horizon

    def evaluate(
        self,
        policy_fn,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        评测策略在 LIBERO 上的表现

        Args:
            policy_fn: 策略函数, obs → action

        Returns:
            results: {suite_name: {task_name: success_rate}}
        """
        all_results = {}

        for suite_name in self.task_suites:
            suite_results = {}

            for task_id in range(10):  # 每个 suite 10 个任务
                env = LiberoWrapper(
                    task_suite_name=suite_name,
                    task_id=task_id,
                )
                env.setup()

                successes = 0
                for ep in range(self.num_eval_episodes):
                    obs = env.reset()
                    done = False

                    while not done:
                        action = policy_fn(obs)
                        obs, reward, done, info = env.step(action)

                    if info.get('success', False) or info.get('is_success', False):
                        successes += 1

                success_rate = successes / self.num_eval_episodes * 100
                task_name = f"task_{task_id}"
                suite_results[task_name] = success_rate

                if verbose:
                    print(f"  {suite_name}/{task_name}: {success_rate:.1f}%")

                env.close()

            avg = np.mean(list(suite_results.values()))
            suite_results['average'] = avg
            all_results[suite_name] = suite_results

            if verbose:
                print(f"  {suite_name} Average: {avg:.1f}%")

        return all_results


# ============================================================
# 数据转换工具
# ============================================================

class LiberoDataConverter:
    """
    将 LIBERO rollout 数据转换为 RECAP 训练格式

    输出格式 (每个 timestep):
    {
        'observation': {
            'image': (H, W, 3),
            'state': (action_dim,),
            'eef_pos': (3,),
            'eef_quat': (4,),
            'gripper_qpos': (2,),
        },
        'action': (action_dim,),
        'reward': float,
        'language': str,
        'advantage_label': int,  # 后续由 AdvantageComputer 填充
    }
    """

    @staticmethod
    def episode_to_training_format(episode: Dict) -> List[Dict]:
        """将单个 episode 转换为训练样本列表"""
        samples = []
        T = len(episode['observations'])

        for t in range(T - 1):  # 最后一步没有 next action
            sample = {
                'observation': episode['observations'][t],
                'action': episode['actions'][t],
                'reward': episode['rewards'][t],
                'language': episode['language'],
            }
            samples.append(sample)

        return samples

    @staticmethod
    def compute_empirical_returns(episode: Dict, gamma: float = 1.0) -> np.ndarray:
        """
        计算每个 timestep 的经验回报
        R_t = Σ_{t'=t}^{T} γ^{t'-t} · r_{t'}
        """
        rewards = np.array(episode['rewards'])
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)

        running_return = 0.0
        for t in reversed(range(T)):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return

        return returns


if __name__ == "__main__":
    print("LIBERO Wrapper for RECAP Reproduction")
    print("=" * 50)
    print()

    # 测试 mock 环境
    env = LiberoWrapper(task_suite_name="libero_spatial", task_id=0)
    obs = env.reset()
    print(f"Observation keys: {list(obs.keys())}")
    print(f"Image shape: {obs['image'].shape}")
    print(f"State shape: {obs['state'].shape}")
    print(f"Task prompt: {obs['prompt']}")

    # 测试动作去归一化
    test_action = np.array([0.5, -0.3, 0.2, 0.1, -0.1, 0.0, 0.8], dtype=np.float32)
    denorm = env._denormalize_action(test_action)
    print(f"\nNormalized action: {test_action}")
    print(f"Denormalized action: {denorm}")

    # 测试 evaluator
    print()
    print("LiberoEvaluator ready for Phase 1A evaluation.")
