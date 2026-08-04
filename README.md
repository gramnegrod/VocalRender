# VocalRender

**English** | [简体中文](README.zh-CN.md)

**Paper:** [VocalRender: Score-Native Singing Voice Synthesis for Real-World Composition](https://arxiv.org/abs/2607.27768)

**VocalRender** is a singing voice synthesis (SVS) system built on top of
[OpenBMB VoxCPM](https://github.com/OpenBMB/VoxCPM). It adapts VoxCPM's
tokenizer-free TTS architecture to singing by replacing the plain-text prompt
with a **word / pitch / note interleaved score prompt**:

```
<BPM_90> 感<P_62><NOTE_8> 受<P_62><NOTE_DOT_16><P_60><NOTE_16> 停<P_59><NOTE_8> ... <audio_start> [audio latents...]
```

Each lyric word is followed by one or more `(pitch, note-duration)` token
pairs, plus a global BPM token — the model renders the score into 48 kHz
singing audio, optionally cloning timbre from a same-song prompt-audio clip.

This repository is the minimal open-source release: data preprocessing,
training, and inference.

## Overview

Unlike duration-based SVS systems that require an exact duration for every
word or phoneme, or reference-based systems that depend on time-aligned audio
or an F0 curve, VocalRender takes the same symbolic information used by a
composer: lyrics, MIDI pitches, note values, and a global tempo. The model is
free to realize natural timing and expressive deviations while following the
score.

![Comparison of duration-based, reference-based, and the proposed score-native singing voice synthesis inputs](assets/intro.png)

VocalRender is built around three ideas:

1. **Score-native interleaved representation.** The music-score tokenizer
   serializes BPM followed by each lyric syllable and all of its associated
   `(pitch, note-value)` pairs. This explicitly preserves lyric-to-note
   alignment, including melisma, where one syllable spans multiple notes.
2. **Continuous acoustic representation.** An Audio VAE encodes singing into
   compact continuous latents, retaining fine-grained pitch, timbre, and
   articulation information without discrete acoustic quantization.
3. **Autoregressive diffusion generation.** The AR Transformer generates a
   prosody sketch and predicts the sequence length patch by patch, while
   LocDiT reconstructs high-fidelity local acoustic latents. The VAE decoder
   then renders the completed latent sequence into waveform audio. This avoids
   an explicit duration predictor and time-aligned acoustic guidance.

![Overall architecture of VocalRender and its music-score tokenization process](assets/structure.png)

## Repository layout

```
conf/               Training / inference / preprocessing YAML configs
scripts/            Entry-point scripts (preprocess, train, infer)
src/vocalrender/    The package (model, training, inference, evaluation)
nanovllm-voxcpm/    Optional nano-vllm inference backend (git submodule)
docs/               Architecture and usage documentation
```

See [docs/structure.md](docs/structure.md) for the full tree.

## Installation

```bash
git clone --recurse-submodules https://github.com/pymaster17/VocalRender.git
cd VocalRender

python -m venv .venv && source .venv/bin/activate
pip install -e .

# Optional: staff-notation score rendering (save_score: true)
pip install -e ".[viz]"

# Optional: nano-vllm inference backend (continuous batching)
pip install -e ./nanovllm-voxcpm
```

## Pretrained weights & tokenizer

1. Download the **VoxCPM2** pretrained checkpoint into
   `pretrained_models/VoxCPM2` (must contain `config.json`, model weights,
   and the tokenizer files).
2. Extend the tokenizer with the SVS tokens (128 pitch + 12 note-duration +
   256 BPM tokens):

```bash
python scripts/setup_svs_tokenizer.py \
    --tokenizer_path pretrained_models/VoxCPM2 \
    --save_path pretrained_models/VoxCPM2   # overwrite in place, or a new dir
```

Model embeddings are resized automatically at training start.

## Data preprocessing

Annotate each audio segment with word/pitch/note fields (see the schema
comment in [conf/svs_preprocess.yaml](conf/svs_preprocess.yaml)):

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

`word_dur` and `pitch_dur` are optional input fields used only for
visualization and evaluation. They are not used to construct the score prompt
or train the model. The required score fields are `word`, `pitch`, `note`,
`pitch2word`, and `bpm`.

Then encode audio into AudioVAE-V2 latents (Arrow shards):

```bash
python scripts/preprocess_svs_data.py conf/svs_preprocess.yaml
```

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/train_vocalrender_svs.py --config_path conf/svs_train.yaml
```

Any config key can be overridden with dotted paths, e.g.
`--set train.batch_size=32 --set runtime.save_path=checkpoints/run2`.
Training resumes automatically from the latest checkpoint under `save_path`.

Validation logs loss plus audio-quality metrics (SingMOS, Audiobox
Aesthetics) and sample audio to TensorBoard.

## Inference

**Batch inference** over a preprocessed validation set (writes wavs,
optional score PNGs, and `metrics_summary.json`):

```bash
python scripts/infer_vocalrender_svs.py --config_path conf/svs_infer.yaml
```

Two backends are available (`docs/inference_backends.md`):
`multi_gpu` (default, in-process; supports prompt audio + score rendering)
and `nano_vllm` (continuous batching, faster for metric-only runs).

**Single sample** from a label JSON:

```bash
python scripts/infer_vocalrender_svs_single.py \
    --ckpt_dir checkpoints/svs_v2/latest \
    --json_file data/labels/opencpop.json \
    --item_name 2001000001 \
    --output svs_output.wav
```

## Metrics

- **SingMOS** — singing MOS predictor (loaded via `torch.hub`, requires `s3prl`).
- **AES** — Audiobox Aesthetics axes (CE = content enjoyment, PQ = production quality).
- A pluggable `register_metric_backend` seam in
  `vocalrender.evaluation.svs_metrics` lets you add custom metrics without
  editing the evaluator.

## Acknowledgements

- [VoxCPM](https://github.com/OpenBMB/VoxCPM) (OpenBMB) — the TTS foundation
  model this work builds on; the model architecture (TSLM / LocEnc / LocDiT /
  AudioVAE) and pretrained weights come from the VoxCPM project.
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — the lightweight
  vLLM implementation adapted for the `nano_vllm` inference backend.
- [SingMOS](https://github.com/South-Twilight/SingMOS) and
  [audiobox-aesthetics](https://github.com/facebookresearch/audiobox-aesthetics)
  — evaluation models.

## License

Apache-2.0 (same as upstream VoxCPM). See [LICENSE](LICENSE).
