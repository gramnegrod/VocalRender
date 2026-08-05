# VocalRender

**English** | [简体中文](README.zh-CN.md)

[![arXiv](https://img.shields.io/badge/arXiv-2607.27768-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2607.27768)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-VocalRender-FFD21E?logo=huggingface)](https://huggingface.co/pymaster/VocalRender)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-CrawlSinger--OS-FFD21E?logo=huggingface)](https://huggingface.co/datasets/pymaster/CrawlSinger-OS)
[![Audio Demo](https://img.shields.io/badge/Audio-Demo-183d32.svg)](https://pymaster17.github.io/VocalRender/)

**VocalRender** is a score-native singing voice synthesis (SVS) system designed
for real-world composition. It directly renders composer-oriented symbolic
scores into singing audio through an original combination of an interleaved
lyric--note representation, continuous acoustic latents, and autoregressive
diffusion modeling. Its input is a **word / pitch / note interleaved score
prompt**:

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

## Pretrained checkpoints & tokenizer

Ready-to-run **VocalRender** and **VocalRender-Pro** checkpoints are available
from the [Hugging Face model repository](https://huggingface.co/pymaster/VocalRender).
Each variant includes the model weights, AudioVAE, configuration, and extended
SVS tokenizer.

Download one variant into `pretrained_models/`:

```bash
# VocalRender
hf download pymaster/VocalRender \
    --include "VocalRender/*" \
    --local-dir pretrained_models

# Or VocalRender-Pro
hf download pymaster/VocalRender \
    --include "VocalRender-Pro/*" \
    --local-dir pretrained_models
```

The resulting checkpoint paths are `pretrained_models/VocalRender` and
`pretrained_models/VocalRender-Pro`, respectively. Each download is about
9.5 GB. Full audio generation requires a CUDA-capable compute node.

The following steps are needed only when preparing the **VoxCPM2 base model
for training**, not when using the released VocalRender checkpoints:

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

The released training data is available as
[CrawlSinger-OS](https://huggingface.co/datasets/pymaster/CrawlSinger-OS),
which contains Muse, Muchin, SongFormDB, OpenSinger, M4Singer, and GTSinger
for training, plus Opencpop as the held-out validation set.
Download the independently sharded archives and restore the directory expected
by [conf/svs_preprocess.yaml](conf/svs_preprocess.yaml):

```bash
hf download pymaster/CrawlSinger-OS \
    --repo-type dataset \
    --local-dir data/CrawlSinger-OS-release

mkdir -p data/CrawlSinger-OS
find data/CrawlSinger-OS-release -type f -name '*.tar' -print0 |
    while IFS= read -r -d '' shard; do
        tar -xf "$shard" -C data/CrawlSinger-OS
    done

for dataset in opensinger m4singer gtsinger opencpop; do
    cp "data/CrawlSinger-OS-release/${dataset}/annotations.json" \
       "data/CrawlSinger-OS/${dataset}/annotations.json"
done
```

The archives contain the three folder-based datasets directly and place the
audio for each of the four JSON-based datasets under `<dataset>/audio/`. The copied
annotation files complete the layout consumed by the default preprocessing
configuration. Keep enough disk space for both the downloaded archives and
the extracted data.

Annotate each audio segment with word/pitch/note fields (see the schema
comment in [conf/svs_preprocess.yaml](conf/svs_preprocess.yaml)):

```jsonc
[
  {
    "item_name": "Alto-1#newboy#0000",
    "wav_fn":    "Alto-1#newboy/0000.wav",
    "word":       ["AP", "感", "受", "SP"],
    "word_dur":   [0.14, 0.31, 0.42, 0.20],
    "pitch":      [0, 62, 62, 0],
    "note":       ["<NOTE_8>", "<NOTE_8>", "<NOTE_DOT_16>", "<NOTE_8>"],
    "pitch_dur":  [0.14, 0.31, 0.42, 0.20],
    "pitch2word": [0, 1, 2, 3],
    "bpm":        90
  }
]
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
    --ckpt_dir pretrained_models/VocalRender \
    --json_file examples/inference_input.json \
    --item_name demo \
    --output svs_output.wav
```

The example runs without reference audio. To clone a timbre, add a
`--prompt_item_name` whose JSON entry contains `wav_fn`; see
`python scripts/infer_vocalrender_svs_single.py --help` for the related audio
path options.

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
