# π*0.6 RECAP 算法复现方案

> 基于 openpi (π0.5) + LIBERO 仿真环境的 RECAP 算法复现计划
> 论文：π*0.6: a VLA That Learns From Experience (arXiv:2511.14759)
> 硬件：NVIDIA A800 80GB × 1（单卡全量微调 + 价值函数训练）

---

## 一、论文核心算法提炼

### 1.1 RECAP 三步循环

```
Repeat until convergence:
  1. 数据采集: 用当前策略在环境中 rollout，记录 episode 结果（成功/失败），可选人工纠错
  2. 价值函数训练: 用全部数据训练分布式价值函数 V^π_ref(o_t, ℓ)
  3. 优势条件策略训练: 基于价值函数计算优势 → 二值化 → 加入策略前缀作为条件 → 训练改进策略
```

### 1.2 分布式价值函数 (Distributional Value Function)

**架构**：与 VLA 策略相同架构，但使用更小的 VLM 主干（~670M 参数）

**输出**：p_φ(V | o_t, ℓ) ∈ Δ_B，将价值离散化为 B 个 bin 的概率分布

**训练目标**：交叉熵损失
```
L_value = -log p_φ(V = R_t(τ) | o_t, ℓ)
```
其中 R_t(τ) = Σ_{t'=t}^{T} r_{t'} 是从 t 时刻到 episode 结束的经验回报

**关键细节**：
- 混入少量多模态网页数据防止过拟合
- 语言条件化：输入任务描述 ℓ

### 1.3 奖励定义

**稀疏奖励**：
- r_t = 0, for t < T
- r_T = +1 (成功)
- r_T = -1 (失败)

**时间惩罚变体**（论文中部分实验使用）：
- 每步小负奖励鼓励更快完成

### 1.4 优势估计 (Advantage Estimation)

**Post-training 阶段**（N-step，更精确）：
```
A^π(o_t, a_t) = Σ_{t'=t}^{t+N-1} r_{t'} + V^π(o_{t+N}) - V^π(o_t)
```
- N = 50 (lookahead steps)
- o_{t+N} 从同一轨迹中 t+N 步采样

**Pre-training 阶段**（Monte Carlo，更高效）：
```
A^π(o_t, a_t) = Σ_{t'=0}^{T} r_{t'} - V^π(o_t)
```
- N = T，即完整 episode 回报减去当前价值估计
- 只需一次 value function 推理，适合大规模预训练

### 1.5 优势二值化 (Advantage Binarization)

```
I_t = positive,  if A^π(o_t, a_t) > ε_ℓ
I_t = negative,  otherwise
```

**阈值设定**：
| 阶段 | 标准 |
|------|------|
| 预训练 | 每个任务约 30% 演示数据为 positive（从 10k 采样点估计） |
| 微调 | 约 40% 评估 rollout 为 positive |
| 严格标准任务 | 约 10% 为 positive（如衣领朝上的叠衣） |

### 1.6 优势条件策略训练 (Advantage-Conditioned Policy Extraction)

**核心公式**：
```
π^(a|o) ∝ π_ref(a|o) · p(I | A^π_ref(o,a))^β
```

**训练目标**（KL 散度最小化）：
```
min_θ E_{s ~ ρ_π_ref} [KL(π^, π_θ)]
```

**具体实现**：
- 将优势指示器 I (positive/negative) 作为额外 token 加入 VLA 的 prefix
- 训练时：30% 概率随机丢弃优势条件（dropout），使模型同时学习有条件和无条件策略
- 这使得推理时可以使用 CFG (Classifier-Free Guidance)

### 1.7 推理时的 CFG

```
π^(a|o,ℓ) ∝ π_ref(a|o,ℓ) · (π_ref(a|o,I,ℓ) / π_ref(a|o,ℓ))^β
```

- β ∈ [1.5, 2.5] 为推荐范围
- β > 1 时增强高优势行为
- 过高的 β 会导致动作过于激进

**Flow matching 推理中的实现**：
```
∇_a log π^(a|o,ℓ) = ∇_a log π_θ(a|o,ℓ) + β · (∇_a log π_θ(a|o,I,ℓ) - ∇_a log π_θ(a|o,ℓ))
```
即在 flow matching 的每一步积分中，用上述梯度替代原始梯度。

### 1.8 数据聚合策略 (Algorithm 1)

每个迭代的数据集组成：
- D_demo: 人类演示数据
- D_autonomous: 自主 rollout 数据（带结果标签）
- D_correction: 人工纠错干预数据

合并后统一用于价值函数训练和策略训练。

---

## 二、复现架构设计

### 2.1 整体流水线

```
                    ┌─────────────────────┐
                    │  π0.5 Base Model    │
                    │  (openpi 开源权重)   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Step 0: SFT on     │
                    │  LIBERO Demo Data   │
                    │  (pi05_libero config)│
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼──────┐ ┌──────▼───────┐ ┌──────▼───────┐
    │ SFT Baseline   │ │ Value Func   │ │ Rollout      │
    │ (评测基准)      │ │ Training     │ │ Collection   │
    │                │ │              │ │ (LIBERO Sim) │
    └────────────────┘ └──────┬───────┘ └──────┬───────┘
                              │                │
                    ┌─────────▼────────────────▼────────┐
                    │  Advantage Computation            │
                    │  (N-step returns + value est.)    │
                    └──────────────────┬────────────────┘
                                       │
                    ┌───────────────────▼────────────────┐
                    │  Advantage-Conditioned Policy      │
                    │  Training (RECAP core)             │
                    │  - Add advantage token to prefix   │
                    │  - 30% dropout for CFG             │
                    └──────────────────┬────────────────┘
                                       │
                    ┌───────────────────▼────────────────┐
                    │  Evaluation on LIBERO              │
                    │  (with optional CFG, β∈[1.5,2.5])  │
                    └────────────────────────────────────┘
```

### 2.2 代码模块结构

```
recap/
├── configs/
│   ├── recap_libero.yaml          # 主配置文件
│   └── value_function.yaml        # 价值函数配置
├── models/
│   ├── value_function.py          # 分布式价值函数
│   ├── advantage_conditioner.py   # 优势条件模块（修改VLA prefix）
│   └── cfg_policy.py             # CFG推理策略
├── training/
│   ├── train_value.py            # 价值函数训练
│   ├── train_recap.py            # RECAP策略训练
│   └── compute_advantages.py     # 优势计算与标注
├── data/
│   ├── rollout_collector.py      # 仿真环境rollout采集
│   ├── libero_wrapper.py         # LIBERO环境封装
│   └── data_aggregator.py        # 数据聚合（demo+rollout+correction）
├── evaluation/
│   ├── evaluate_libero.py        # LIBERO评测
│   └── compare_results.py        # 结果对比
├── scripts/
│   ├── run_phase1_sft.sh         # 阶段1：SFT baseline
│   ├── run_phase1_value.sh       # 阶段1：价值函数训练
│   ├── run_phase1_rollout.sh     # 阶段1：rollout采集
│   ├── run_phase1_recap.sh       # 阶段1：RECAP训练
│   └── run_phase1_eval.sh        # 阶段1：评测
└── README.md
```

### 2.3 与 openpi 的集成方式

- **基座模型**：直接使用 openpi 的 π0.5 实现
- **训练框架**：基于 openpi 的 JAX 训练栈（FSDP + 混合精度）
- **数据格式**：LeRobot 格式（openpi 原生支持）
- **推理服务**：使用 openpi 的 WebSocket policy server

**关键修改点**：
1. `embed_prefix()` 方法：增加优势条件 token
2. 训练循环：增加优势条件 dropout 逻辑
3. 推理流程：实现 CFG 采样

---

## 三、第一阶段实施计划

### Phase 1A：环境搭建 + SFT Baseline（预计 1-2 天）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 1 | 克隆 openpi，安装依赖 | 可运行的 openpi 环境 |
| 2 | 下载 π0.5 base + LIBERO 权重 | 预训练 checkpoint |
| 3 | 使用 pi05_libero 配置 fine-tune | SFT baseline checkpoint |
| 4 | 在 LIBERO-Spatial 上评测 | Baseline 成功率 |

### Phase 1B：价值函数实现（预计 2-3 天）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 1 | 实现分布式价值函数模型 | value_function.py |
| 2 | 基于 π0.5 架构构建价值网络（小VLM主干） | 可训练的价值模型 |
| 3 | 在 LIBERO demo 数据上训练价值函数 | 训练好的价值函数 |
| 4 | 可视化价值函数预测（任务进度估计） | 验证价值函数质量 |

### Phase 1C：Rollout 采集 + 优势计算（预计 2-3 天）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 1 | 封装 LIBERO 为标准 RL 环境 | libero_wrapper.py |
| 2 | 用 SFT 策略采集自主 rollout | 自主经验数据集 |
| 3 | 计算每步的 N-step advantage | 带优势标签的数据集 |
| 4 | 二值化优势（30%/40% 阈值切分） | positive/negative 标签 |

### Phase 1D：RECAP 核心训练（预计 3-5 天）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 1 | 修改 π0.5 架构，添加优势条件 token | advantage_conditioner.py |
| 2 | 实现 30% 优势条件 dropout | 训练数据增强 |
| 3 | 训练 advantage-conditioned 策略 | RECAP 策略 checkpoint |
| 4 | 实现 CFG 推理 | cfg_policy.py |
| 5 | 完整评测对比 | SFT vs RECAP 结果表 |

### Phase 1E：迭代改进 + 消融实验（预计 2-3 天）

| 步骤 | 内容 | 产出 |
|------|------|------|
| 1 | 第2轮 rollout + RECAP 训练 | 迭代改进结果 |
| 2 | 消融：RECAP vs AWR vs PPO | 方法对比 |
| 3 | 消融：不同 β 值的 CFG 效果 | CFG 参数扫描 |
| 4 | 消融：不同 N-step 值 | 优势估计精度影响 |

---

## 四、关键技术细节与超参数

### 4.1 价值函数超参数

| 参数 | 值 | 来源 |
|------|------|------|
| VLM 主干 | Gemma-2B (PaliGemma 的 2B 变体) | 论文 V-B 节 |
| 价值 bin 数 B | 100 | 论文 IV-A 节 |
| 价值范围 | [-1, 0] (归一化到 episode 长度) | 推断 |
| 训练数据混合 | 95% 机器人 + 5% 多模态网页 | 论文 V-C 节 |
| 学习率 | 1e-4 (cosine decay) | 参考 openpi |
| Batch size | 256 | 参考 openpi |
| 训练步数 | 50k-100k | 需调参 |

### 4.2 RECAP 策略训练超参数

| 参数 | 值 | 来源 |
|------|------|------|
| 优势条件 dropout | 30% | 论文 A-F |
| N-step lookahead | 50 | 论文 A-F |
| 优势阈值 (预训练) | 30% 分位 | 论文 A-F |
| 优势阈值 (微调) | 40% 分位 | 论文 A-F |
| CFG β (推理) | 1.5-2.5 | 论文 A-E |
| 学习率 | 5e-5 (cosine decay) | 需调参 |
| Batch size | 256 | 参考 openpi |
| 训练步数 | 30k | 参考 openpi |

### 4.3 Rollout 采集超参数

| 参数 | 值 | 来源 |
|------|------|------|
| 每轮采集 episode 数 | 300-600 | 论文 A-F |
| Episode 最大步数 | 取决于 LIBERO 任务 | - |
| 采集环境数 | 1 (LIBERO 仿真) | - |
| 评测 episode 数 | 50-100 per task | 标准 |

### 4.4 LIBERO 评测协议

| Benchmark | 任务数 | 关注重点 |
|-----------|--------|---------|
| LIBERO-Spatial | 10 | 空间关系理解 |
| LIBERO-Object | 10 | 物体泛化 |
| LIBERO-Goal | 10 | 目标泛化 |
| LIBERO-Long | 10 | 长时序任务 |

评测指标：Success Rate (%), 以及 Throughput (tasks/hour)

---

## 五、A800 显存与性能估算

| 模型 | 参数量 | 训练VRAM | 推理VRAM | 训练时间 (30k/50k steps) |
|------|--------|---------|---------|------------------------|
| π0.5 全量微调 | 3.3B | ~52GB | ~6GB | 4-6h (30k) |
| π0.5 LoRA微调 | 3.3B+LoRA | ~22GB | ~6GB | 3-4h (30k) |
| 价值函数 (Gemma-2B) | ~670M | ~12GB | ~2GB | 2-3h (50k) |
| RECAP策略训练 | 3.3B+adv | ~53GB | ~6GB | 4-6h (30k) |

**A800 单卡策略**：
- 全量微调和价值函数训练**分时使用**（不同时训练，80GB充裕）
- Rollout采集：加载策略模型推理(~6GB)，剩余供环境
- CFG推理：2次前向/step，~100ms/step，episode ~40s
- 建议设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.92`

---

## 六、风险与缓解

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| π0.5 架构修改复杂（添加 advantage token） | 高 | 先在简化模型上验证，再迁移到完整架构 |
| 价值函数在 LIBERO 上训练不收敛 | 中 | 增大 bin 数、调整学习率、增加数据增强 |
| GPU 资源不足 | 低 | A800 80GB 充裕，全量微调~52GB，价值函数~12GB，分时使用 |
| Rollout 采集效率低 | 中 | 并行化环境、降低采样数量 |
| RECAP 改进不显著 | 低 | 增大 rollout 数据量、调整优势阈值 |

---

## 六、第二阶段规划（概要）

Phase 2 目标：在 Phase 1 验证 RECAP 有效性的基础上，完整复现 π*0.6 的效果

- 升级到 Gemma3 4B 主干 + 860M Action Expert
- 多任务联合训练
- 更大规模 rollout 采集
- 与论文结果对齐的完整评测
- 真实机器人部署验证（如有条件）
