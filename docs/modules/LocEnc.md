# Local Encoder (LocEnc) 模块介绍

## 1. 概述
**LocEnc (Local Audio Encoder)** 是 VoxCPM 模型中负责对输入的音频特征进行编码的模块。它的主要作用是将通过 AudioVAE 提取的音频 Latent 特征（通常维度较低，如 64 维）映射到语言模型（LM）的隐藏层维度（如 1024 维），以便与文本 Token 的 Embedding 进行融合。

## 2. 代码位置
- **路径**: `src/vocalrender/modules/locenc/local_encoder.py`
- **类名**: `VoxCPMLocEnc`

## 3. 核心逻辑

### 3.1 初始化 (`__init__`)
- **配置 (`MiniCPM4Config`)**: 接收 MiniCPM 的配置对象，主要使用其 `hidden_size` 等参数。
- **特殊 Token (`special_token`)**: 初始化一个可学习的特殊向量，维度为 `[1, 1, 1, hidden_size]`。该 Token 被拼接到每个音频 Patch 的开头，类似于 BERT 的 `[CLS]` Token，用于汇聚该 Patch 的全局信息。
- **输入投影 (`in_proj`)**: 一个线性层 `nn.Linear(input_dim, hidden_size)`，将输入音频维度映射到模型维度。
- **Encoder (`MiniCPMModel`)**: 这是一个标准的 MiniCPM Transformer 模型（`vocab_size` 被设置为 0，因为不涉及 Token 预测），用于处理序列数据。

### 3.2 前向传播 (`forward`)
输入 `x` 的形状为 `[B, T, P, D]`，其中：
- `B`: Batch Size
- `T`: 序列时间步 (Time Steps)
- `P`: Patch Size (每个时间步包含的帧数)
- `D`: Input Dimension (音频特征维度)

**处理流程**:
1.  **投影**: `x = self.in_proj(x)`，维度变为 `[B, T, P, hidden_size]`。
2.  **拼接 Special Token**: 将 `special_token` 扩展后拼接到每个 Patch 序列的最前面 (dim=2)，维度变为 `[B, T, P+1, hidden_size]`。
3.  **重排**: 将 Batch 和 Time 维度合并，视为独立的序列进行处理：`[B*T, P+1, hidden_size]`。
4.  **Encoder 编码**: 输入 Transformer Encoder。虽然这里使用的是 `MiniCPMModel`，但实际上是把它当作一个通用的 Transformer Encoder 来提取特征。
5.  **提取特征**: 取出每个序列的第一个 Token（即对应 `special_token` 位置的输出）作为该 Patch 的聚合表示：`outputs[:, 0, :]`。
6.  **恢复形状**: 还原回 `[B, T, hidden_size]`。

## 4. 总结
LocEnc 通过 "Patch 内部编码" 的方式，将一段高频的音频特征（例如 2 帧或更多）压缩为一个单一的向量，作为 TSLM 和 RALM 的输入。这有效地降低了序列长度，使模型能够处理更长的上下文。
