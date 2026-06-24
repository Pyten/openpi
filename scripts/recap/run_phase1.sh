#!/bin/bash
# ============================================================
# RECAP Phase 1 完整执行脚本
# 基于 openpi (π0.5) + LIBERO 仿真环境
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OPENPI_DIR="${PROJECT_DIR}/openpi"
RECAP_DIR="${PROJECT_DIR}/recap-reproduction"

echo "============================================================"
echo "RECAP Phase 1: π*0.6 Reproduction on LIBERO"
echo "============================================================"
echo "Project dir: ${PROJECT_DIR}"
echo ""

# ============================================================
# Phase 1A: 环境搭建 + SFT Baseline
# ============================================================

phase_1a() {
    echo ""
    echo ">>> Phase 1A: Environment Setup + SFT Baseline"
    echo ""
    
    # 1. 克隆 openpi
    if [ ! -d "${OPENPI_DIR}" ]; then
        echo "[1/5] Cloning openpi repository..."
        git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "${OPENPI_DIR}"
    else
        echo "[1/5] openpi already cloned, skipping."
    fi
    
    cd "${OPENPI_DIR}"
    
    # 2. 安装依赖
    echo "[2/5] Installing dependencies with uv..."
    GIT_LFS_SKIP_SMUDGE=1 uv sync
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
    
    # 3. 验证安装
    echo "[3/5] Verifying installation..."
    uv run python -c "import openpi; print('openpi installed successfully')"
    
    # 4. 计算 LIBERO 归一化统计
    echo "[4/5] Computing normalization statistics for pi05_libero..."
    uv run scripts/compute_norm_stats.py --config-name pi05_libero
    
    # 5. SFT 训练 (π0.5 on LIBERO)
    echo "[5/5] Training π0.5 SFT baseline on LIBERO..."
    echo "  Config: pi05_libero"
    echo "  Steps: 30000"
    echo "  This requires >= 70GB VRAM for full fine-tuning"
    echo "  Or >= 22.5GB for LoRA fine-tuning"
    echo ""
    echo "  Full fine-tuning command:"
    echo "    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name recap_sft_baseline --overwrite"
    echo ""
    echo "  LoRA fine-tuning command (lower memory):"
    echo "    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_low_mem_finetune --exp-name recap_sft_baseline --overwrite"
    echo ""
    echo "  NOTE: Actual training requires GPU with sufficient VRAM."
    echo "  Run the appropriate command above on your GPU machine."
}

# ============================================================
# Phase 1B: 价值函数训练
# ============================================================

phase_1b() {
    echo ""
    echo ">>> Phase 1B: Value Function Training"
    echo ""
    
    echo "[1/3] Preparing value function training data..."
    echo "  - Converting LIBERO demo data to value function format"
    echo "  - Computing empirical returns for each timestep"
    echo "  - Mixing with 5% web multimodal data"
    echo ""
    echo "[2/3] Training distributional value function..."
    echo "  Architecture: Gemma-2B VLM backbone + value head"
    echo "  Value bins: 100"
    echo "  Loss: Cross-entropy against empirical return bins"
    echo ""
    echo "[3/3] Validating value function quality..."
    echo "  - Visualizing value predictions on held-out episodes"
    echo "  - Checking monotonicity (value should increase toward success)"
    echo ""
    echo "  Run: python ${RECAP_DIR}/recap_core.py --phase value_training"
}

# ============================================================
# Phase 1C: Rollout 采集 + 优势计算
# ============================================================

phase_1c() {
    echo ""
    echo ">>> Phase 1C: Rollout Collection + Advantage Computation"
    echo ""
    
    echo "[1/4] Setting up LIBERO simulation environment..."
    echo "  Tasks: LIBERO-Spatial (10 tasks)"
    echo "  Max steps per episode: 400"
    echo ""
    echo "[2/4] Collecting SFT policy rollouts..."
    echo "  Episodes per task: 300"
    echo "  Using π0.5 SFT baseline as policy"
    echo ""
    echo "[3/4] Computing N-step advantages..."
    echo "  N = 50 (lookahead)"
    echo "  Using trained value function for bootstrap"
    echo ""
    echo "[4/4] Binarizing advantages..."
    echo "  Threshold: 40th percentile (≈40% positive)"
    echo "  Labels: positive (1) / negative (0)"
    echo ""
    echo "  Run: python ${RECAP_DIR}/recap_core.py --phase rollout_and_advantage"
}

# ============================================================
# Phase 1D: RECAP 核心训练
# ============================================================

phase_1d() {
    echo ""
    echo ">>> Phase 1D: RECAP Core Training"
    echo ""
    
    echo "[1/4] Modifying π0.5 architecture for advantage conditioning..."
    echo "  - Adding advantage token to VLA prefix"
    echo "  - Implementing 30% advantage dropout"
    echo "  - Creating conditional/unconditional policy paths"
    echo ""
    echo "[2/4] Training advantage-conditioned policy..."
    echo "  Data: demo + rollout + labeled advantages"
    echo "  Steps: 30000"
    echo "  Dropout: 30% on advantage conditioning"
    echo ""
    echo "[3/4] Implementing CFG inference..."
    echo "  β range: [1.5, 2.5]"
    echo "  Conditional + unconditional velocity combination"
    echo ""
    echo "[4/4] Full evaluation on LIBERO..."
    echo "  Tasks: LIBERO-Spatial, Object, Goal, Long"
    echo "  Metrics: Success Rate, Throughput"
    echo "  Comparison: SFT baseline vs RECAP"
    echo ""
    echo "  Run: python ${RECAP_DIR}/recap_core.py --phase recap_training"
}

# ============================================================
# Phase 1E: 消融实验
# ============================================================

phase_1e() {
    echo ""
    echo ">>> Phase 1E: Ablation Studies"
    echo ""
    
    echo "[1/4] RECAP vs alternative methods..."
    echo "  - RECAP (advantage conditioning)"
    echo "  - AWR (advantage weighted regression)"
    echo "  - PPO (proximal policy optimization)"
    echo ""
    echo "[2/4] CFG β sweep..."
    echo "  β values: [1.0, 1.5, 2.0, 2.5, 3.0]"
    echo ""
    echo "[3/4] N-step advantage comparison..."
    echo "  N values: [1, 10, 25, 50, T(MC)]"
    echo ""
    echo "[4/4] Advantage threshold sensitivity..."
    echo "  Positive ratios: [10%, 20%, 30%, 40%, 50%]"
}

# ============================================================
# 主入口
# ============================================================

usage() {
    echo "Usage: $0 {1a|1b|1c|1d|1e|all}"
    echo ""
    echo "  1a   Phase 1A: Environment setup + SFT baseline"
    echo "  1b   Phase 1B: Value function training"
    echo "  1c   Phase 1C: Rollout collection + advantage computation"
    echo "  1d   Phase 1D: RECAP core training"
    echo "  1e   Phase 1E: Ablation studies"
    echo "  all  Run all phases sequentially"
}

case "${1:-}" in
    1a) phase_1a ;;
    1b) phase_1b ;;
    1c) phase_1c ;;
    1d) phase_1d ;;
    1e) phase_1e ;;
    all)
        phase_1a
        phase_1b
        phase_1c
        phase_1d
        phase_1e
        ;;
    *) usage ;;
esac
