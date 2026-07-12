# Audio VAE 模块介绍

## 1. 概述
**Audio VAE (Variational Autoencoder)** 是 VoxCPM 的数据压缩前端。它负责将高维的原始音频波形（例如 16kHz 采样率）压缩为低频、低维的连续 Latent Representation（例如 25Hz, 64维）。这大大降低了核心生成模型 (TSLM/RALM/DiT) 的计算负担，同时保持了高质量的音频重建能力。

## 2. 代码位置
- **路径**: `src/vocalrender/modules/audiovae/audio_vae.py`
- **主要类**: `AudioVAE`, `CausalEncoder`, `CausalDecoder`

## 3. 核心架构

该 VAE 采用了全卷积架构，并特别设计为 **因果 (Causal)** 结构，这意味着当前的输出只依赖于过去和现在的输入，而看不见未来。这是实现流式生成的必要条件。

### 3.1 Causal Encoder (`CausalEncoder`)
- **结构**: 由一系列 `CausalEncoderBlock` 堆叠而成。
- **下采样**: 通过跨步卷积 (Strided Convolution) 逐步降低时间分辨率。配置中的 `strides=[2, 4, 8, 8]` 意味着总下采样率为 $2 \times 4 \times 8 \times 8 = 512$。对于 16kHz 音频，这导致 Latent 的帧率约为 $16000 / 512 \approx 31.25 Hz$ (注：代码默认配置可能与论文稍有出入，论文提及 25Hz 640x 下采样，此处以代码 `strides` 为准)。
- **输出**: 预测分布的均值 `mu` 和对数方差 `logvar`。在推理时通常直接使用 `mu`。

### 3.2 Causal Decoder (`CausalDecoder`)
- **结构**: `CausalEncoder` 的逆过程，由 `CausalDecoderBlock` 组成。
- **上采样**: 通过转置卷积 (Transpose Convolution) 将 Latent 逐步还原回原始波形分辨率。
- **激活函数**: 使用 `Snake1d` 激活函数，这是一种周期性激活函数，被证明对音频波形建模非常有效。
- **输出**: 重建的音频波形。

### 3.3 关键组件
- **`Snake1d`**: 自适应周期激活函数 $x + \frac{1}{\alpha} \sin^2(\alpha x)$，有助于捕捉音频的周期性结构。
- **`WNCausalConv1d`**: 权重归一化 (Weight Norm) + 因果填充 (Causal Padding) 的一维卷积。因果填充通过在左侧填充 (Padding) 并在右侧裁剪，确保卷积核不会看到“未来”的信息。

## 4. 功能接口

### `encode(self, audio_data, sample_rate)`
- **输入**: 原始波形 `[B, 1, T]`。
- **处理**: 预处理（填充） -> Encoder 前向传播 -> 取 `mu`。
- **输出**: Latent `[B, D, T']`。

### `decode(self, z)`
- **输入**: Latent `[B, D, T']`。
- **处理**: Decoder 前向传播。
- **输出**: 重建波形 `[B, 1, T]`。

## 5. 总结
AudioVAE 是 VoxCPM 的“压缩机”和“解压机”。其因果特性是该模型区别于许多非流式 TTS 模型（使用双向卷积）的关键，使得 VoxCPM 能够一边生成 Latent，一边实时解码出音频，实现低延迟流式输出。
