# VoxCPM Model (Core) 模块介绍

## 1. 概述
`VoxCPMModel` 是整个系统的核心容器类，它将 TSLM（文本语义 LM）、RALM（残差声学 LM）、LocEnc（音频编码器）、LocDiT（扩散解码器）和 FSQ（量化层）组装在一起，定义了完整的模型结构、前向传播（用于训练）和推理生成流程。

## 2. 代码位置
- **路径**: `src/vocalrender/model/voxcpm.py`
- **主要类**: `VoxCPMConfig`, `VoxCPMModel`

## 3. 模型架构组件 (`__init__`)

初始化时，`VoxCPMModel` 构建了以下关键子模块：
1.  **`base_lm` (TSLM)**: 基于 MiniCPM4 的语义语言模型。
2.  **`residual_lm` (RALM)**: 同样基于 MiniCPM4 结构，但层数较少，用于声学残差补全。
3.  **`feat_encoder` (LocEnc)**: 用于将音频 Latent 编码为 Embedding。
4.  **`feat_decoder` (LocDiT/UnifiedCFM)**: 扩散生成模块。
5.  **`fsq_layer`**: `ScalarQuantizationLayer`，连接 `base_lm` 和 `residual_lm` 的瓶颈层。
6.  **投影层**:
    -   `enc_to_lm_proj`: LocEnc -> LM 维度。
    -   `lm_to_dit_proj`: TSLM -> DiT 维度。
    -   `res_to_dit_proj`: RALM -> DiT 维度。
7.  **`stop_head`**: 一个简单的分类头，用于预测生成是否结束。
8.  **`audio_vae`**: 加载预训练的 Audio VAE 模型。

## 4. 训练流程 (`forward`)

`forward` 函数定义了端到端的训练逻辑：

1.  **特征准备**:
    -   文本输入 -> Embedding (`text_embed`)。
    -   音频输入 -> VAE Encode -> LocEnc -> Embedding (`feat_embed`)。
    -   组合输入: 根据 Mask 将文本和音频 Embedding 融合。

2.  **第一阶段：语义规划 (TSLM & FSQ)**
    -   输入 `base_lm`。
    -   输出经过 `fsq_layer` 量化。这一步强制模型学习高度压缩的、稳定的语义骨架。
    -   **Stop Prediction**: 基于 TSLM 的输出计算 `stop_loss`。

3.  **第二阶段：声学补全 (RALM)**
    -   输入是 TSLM 的 **未量化输出** (residual connection) + 音频 Embedding。
    -   `residual_lm` 预测被 FSQ 丢弃的细节信息。

4.  **第三阶段：扩散生成 (LocDiT)**
    -   构建 DiT 条件：`cond = lm_to_dit(lm_hidden) + res_to_dit(residual_hidden)`。
    -   `feat_decoder.compute_loss`:基于该条件和 Ground Truth 音频 Latent 计算扩散 Loss (`diff_loss`)。

5.  **Loss 汇总**: 返回 `diff_loss` 和 `stop_loss`。

## 5. 推理生成流程 (`_generate` / `_inference`)

推理过程是一个自回归循环：

1.  **Prompt 处理**: 如果有提示音频，先编码并作为上下文。
2.  **Loop (逐 Patch 生成)**:
    -   计算当前步的 DiT 条件 (`lm_hidden` + `residual_hidden`)。
    -   **LocDiT 采样**: 调用 `feat_decoder` 生成当前 Patch 的 Audio Latent (`pred_feat`)。
    -   **Latent 编码**: 将生成的 Latent 通过 `feat_encoder` 编码回 Embedding，准备输入下一轮 LM。
    -   **Kv-Cache 更新**:
        -   `base_lm` 更新：输入生成的 Audio Embedding，计算新的 `lm_hidden`。
        -   `residual_lm` 更新：基于新的 `lm_hidden` 和 Audio Embedding，计算 `residual_hidden`。
    -   **停止检测**: 检查 `stop_head` 输出。

3.  **流式/非流式输出**:
    -   支持 `yield` 返回音频块实现流式合成。
    -   最终将 Latent 通过 VAE 解码为波形。

## 6. 特性
- **Prompt Cache**: 支持 `build_prompt_cache`，预计算提示音频的特征，加速多次相同 Prompt 的生成。
- **LoRA 集成**: 内置了对 LoRA (Low-Rank Adaptation) 的支持，可针对 LM 或 DiT 部分进行微调。
- **混合精度**: 显式处理 `bfloat16` 等数据类型转换。
