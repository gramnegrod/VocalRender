# Local Diffusion Transformer (LocDiT) 模块介绍

## 1. 概述
**LocDiT (Local Diffusion Transformer)** 是 VoxCPM 的生成后端。它接收来自 TSLM（语义骨架）和 RALM（声学细节）的混合条件，通过流匹配（Flow Matching）算法生成高质量、连续的音频 Latent 特征。它被设计为 "Local" 的，意味着它生成的不是全局序列，而是基于条件自回归地生成一个个 Patch 的 Latent。

## 2. 代码结构
该模块位于 `src/vocalrender/modules/locdit/`，主要包含两个文件：
- **`local_dit.py`**: 定义了 DiT 的神经网络架构 (`VoxCPMLocDiT`)。
- **`unified_cfm.py`**: 定义了训练和推理时的流匹配逻辑 (`UnifiedCFM`)。

## 3. 核心架构 (`local_dit.py`)

### `VoxCPMLocDiT` 类
这是一个基于 Transformer 的条件去噪网络。

#### 主要组件：
1.  **输入投影 (`in_proj`)**: 将带噪输入 `x` (Latent) 映射到隐藏层维度。
2.  **条件投影 (`cond_proj`)**: 将条件 `cond` (Prefix Context) 映射到隐藏层维度。
3.  **时间嵌入 (`SinusoidalPosEmb`, `TimestepEmbedding`)**:
    -   `time_mlp`: 编码当前的扩散时间步 `t`。
    -   `delta_time_mlp`: 编码 `dt` (用于速度场预测，虽然代码中主要使用 `t`)。
4.  **Transformer Backbone (`decoder`)**: 使用 `MiniCPMModel` 作为骨干网络。
5.  **输出投影 (`out_proj`)**: 将 Transformer 输出映射回数据维度 (Latent Dim)。

#### 前向流程 (`forward`):
- **输入**:
    - `x`: 当前时刻的 Noisy Latent Patch。
    - `mu`: 来自 LM (TSLM+RALM) 的 Conditional Embedding，作为全局 Condition。
    - `t`: 扩散时间步。
    - `cond`: 前序生成的 Latent Patch (Prefix)，用于保证生成的连续性。
- **处理**:
    1.  对 `x` 和 `cond` 进行投影。
    2.  计算时间嵌入 `t_emb`。
    3.  **拼接**: 将 `mu + t_emb`（作为第一个 Token）、`cond`（Prefix Tokens）、`x`（Noisy Tokens）在序列维度拼接。
    4.  输入 Transformer。
    5.  截取对应 `x` 部分的输出，经过 `out_proj` 得到预测的速度场/噪声。

## 4. 流匹配逻辑 (`unified_cfm.py`)

### `UnifiedCFM` 类
实现了 Conditional Flow Matching (CFM) 的训练目标和 Euler ODE 采样器。

#### 核心功能：
1.  **训练 (`compute_loss`)**:
    -   **数据构建**: 采样时间步 `t`，构建加噪数据 `x_t` 和目标速度 `v_target`。
    -   **Masking**: 支持 Classifier-Free Guidance (CFG) 的 Drop 策略（训练时随机 Drop 条件 `mu`）。
    -   **Loss 计算**: 计算模型预测速度 `v_pred` 与 `v_target` 之间的 MSE Loss。
    -   **Adaptive Loss**: 支持自适应 Loss 权重。

2.  **推理 (`forward` / `solve_euler`)**:
    -   使用 Euler 方法求解常微分方程 (ODE)，从高斯噪声逐步生成目标 Latent。
    -   **Sway Sampling**: 支持非线性的时间调度策略。
    -   **Classifier-Free Guidance (CFG)**: 在推理由不仅输入 Condition，也输入 Null Condition，通过 `cfg_value` 调整生成的引导强度，平衡生成质量和相关性。

## 5. 总结
LocDiT 是 VoxCPM "高保真" 的关键。它不直接预测离散 Token，而是在连续空间中建模分布。通过结合 LM 提供的强语义/声学条件 (`mu`) 和自身的上下文条件 (`cond`)，它能够生成细节丰富且连贯的语音特征。
