# VocalRender

[English](README.md) | **简体中文**

**VocalRender** 是一个基于 [OpenBMB VoxCPM](https://github.com/OpenBMB/VoxCPM)
构建的歌声合成(SVS)系统。它将 VoxCPM 无 tokenizer 的 TTS 架构迁移到歌唱场景,
把纯文本 prompt 替换为**「字 / 音高 / 音符」交错的乐谱 prompt**:

```
<BPM_90> 感<P_62><NOTE_8> 受<P_62><NOTE_DOT_16><P_60><NOTE_16> 停<P_59><NOTE_8> ... <audio_start> [audio latents...]
```

每个歌词字后面跟一个或多个 `(音高, 音符时值)` token 对,再加一个全局 BPM token
——模型将乐谱渲染为 48 kHz 的歌声音频,并可选地从同一首歌的 prompt 音频片段中
克隆音色。

本仓库是最小化开源版本,包含:数据预处理、训练与推理。

## 仓库结构

```
conf/               训练 / 推理 / 预处理的 YAML 配置
scripts/            入口脚本(预处理、训练、推理)
src/vocalrender/    核心包(模型、训练、推理、评测)
nanovllm-voxcpm/    可选的 nano-vllm 推理后端(git 子模块)
docs/               架构与使用文档
```

完整目录树见 [docs/structure.md](docs/structure.md)。

## 安装

```bash
git clone --recurse-submodules https://github.com/pymaster17/VocalRender.git
cd VocalRender

python -m venv .venv && source .venv/bin/activate
pip install -e .

# 可选:五线谱乐谱渲染(save_score: true)
pip install -e ".[viz]"

# 可选:nano-vllm 推理后端(连续批处理)
pip install -e ./nanovllm-voxcpm
```

## 预训练权重与 tokenizer

1. 将 **VoxCPM2** 预训练 checkpoint 下载到 `pretrained_models/VoxCPM2`
   (需包含 `config.json`、模型权重以及 tokenizer 文件)。
2. 为 tokenizer 扩展 SVS token(128 个音高 + 12 个音符时值 + 256 个 BPM token):

```bash
python scripts/setup_svs_tokenizer.py \
    --tokenizer_path pretrained_models/VoxCPM2 \
    --save_path pretrained_models/VoxCPM2   # 原地覆盖,或指定新目录
```

模型 embedding 会在训练开始时自动 resize。

## 数据预处理

为每个音频片段标注 字 / 音高 / 音符 字段(schema 说明见
[conf/svs_preprocess.yaml](conf/svs_preprocess.yaml)):

```jsonc
{
  "2001000001": {
    "word":       ["AP", "感", "受", "SP"],
    "word_dur":   [0.14, 0.31, 0.42, 0.20],
    "pitch":      [0, 62, 62, 0],
    "note":       ["<NOTE_8>", "<NOTE_8>", "<NOTE_DOT_16>", "<NOTE_8>"],
    "pitch_dur":  [0.14, 0.31, 0.42, 0.20],
    "pitch2word": [0, 1, 2, 3],
    "bpm":        90
  }
}
```

然后将音频编码为 AudioVAE-V2 latent(Arrow 分片):

```bash
python scripts/preprocess_svs_data.py conf/svs_preprocess.yaml
```

## 训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/train_vocalrender_svs.py --config_path conf/svs_train.yaml
```

任意配置项都可以用点分路径覆盖,例如
`--set train.batch_size=32 --set runtime.save_path=checkpoints/run2`。
训练会自动从 `save_path` 下最新的 checkpoint 恢复。

验证阶段会将 loss 及音质指标(SingMOS、Audiobox Aesthetics)和采样音频记录到
TensorBoard。

## 推理

**批量推理**(在预处理好的验证集上运行,输出 wav、可选的乐谱 PNG 以及
`metrics_summary.json`):

```bash
python scripts/infer_vocalrender_svs.py --config_path conf/svs_infer.yaml
```

提供两种后端(详见 `docs/inference_backends.md`):
`multi_gpu`(默认,进程内;支持 prompt 音频 + 乐谱渲染)和
`nano_vllm`(连续批处理,仅算指标时更快)。

**单条推理**(从一个 label JSON):

```bash
python scripts/infer_vocalrender_svs_single.py \
    --ckpt_dir checkpoints/svs_v2/latest \
    --json_file data/labels/opencpop.json \
    --item_name 2001000001 \
    --output svs_output.wav
```

## 评测指标

- **SingMOS** —— 歌声 MOS 预测器(通过 `torch.hub` 加载,需要 `s3prl`)。
- **AES** —— Audiobox Aesthetics 维度(CE = 内容愉悦度,PQ = 制作质量)。
- `vocalrender.evaluation.svs_metrics` 中提供可插拔的 `register_metric_backend`
  接口,无需修改评测器即可添加自定义指标。

## 致谢

- [VoxCPM](https://github.com/OpenBMB/VoxCPM)(OpenBMB)—— 本工作所基于的
  TTS 基础模型;模型架构(TSLM / LocEnc / LocDiT / AudioVAE)与预训练权重
  均来自 VoxCPM 项目。
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) —— 轻量级 vLLM
  实现,`nano_vllm` 推理后端在此基础上改编。
- [SingMOS](https://github.com/South-Twilight/SingMOS) 与
  [audiobox-aesthetics](https://github.com/facebookresearch/audiobox-aesthetics)
  —— 评测模型。

## 许可证

Apache-2.0(与上游 VoxCPM 一致)。见 [LICENSE](LICENSE)。
