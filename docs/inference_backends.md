# TTS Inference Backends

Two interchangeable backends live under
[src/vocalrender/inference/backends/](../src/vocalrender/inference/backends/).
They share the [TTSInferenceBackend](../src/vocalrender/inference/backends/base.py)
ABC and are built via
[build_tts_backend](../src/vocalrender/inference/backends/factory.py).

| Aspect | `multi_gpu` | `nano_vllm` |
|---|---|---|
| Process model | Persistent worker processes, one model per GPU | Out-of-process server pool (subprocesses) |
| Batching | Static batch at dispatch, left-pad AR decode | PagedAttention + continuous batching |
| Throughput (V2, ≥100 h data) | baseline | 2–4× |
| V1 / V1.5 | ✅ | ⚠️ boots, but audio-only (no latent dump, no `prompt_audio_feats`, no `ref_audio_latents`) |
| `prompt_audio_feats` (custom prompt latents) | ✅ | ✅ V2 only — wrapper serialises the `[T,P,D]` np.ndarray to bytes and ships via the server's existing `ref_audio_latents` path (the `<103>[zeros×T]<104>` prefix is structurally identical). V1/V1.5 still raise. |
| `ref_audio_latents` (V2 cloning) | ✅ (CPU tensor, broadcast per-batch) | ✅ V2-only (server-side encode once, broadcast as bytes) |
| `fsq_temperature` / `temperature_mode` (diversity sampling) | ✅ | ❌ (raises `NotImplementedError` on non-default) |
| SVS-finetune checkpoints (extended vocab) | ✅ | ✅ — `NanoVLLMBackend` auto-writes a patched-vocab side-car at `<ckpt>.nanovllm_patched/` since SVS ckpts inherit a stale `vocab_size` from the base model |

## When to pick which

- **`nano_vllm`** — external batch jobs at scale (metric-heavy runs over
  large validation sets).
- **`multi_gpu`** — V1/V1.5 paths, `use_prompt_audio`, `save_score`, and
  diversity sampling (`fsq_temperature` / `temperature_mode`).
  External scripts reach it through the shared `TTSInferenceBackend`
  factory. Training-time validation does not use the backend factory; it
  shards work across DDP ranks in
  [src/vocalrender/training/val_audio.py](../src/vocalrender/training/val_audio.py) and
  calls the shared evaluator path in
  [src/vocalrender/evaluation/inference.py](../src/vocalrender/evaluation/inference.py).

## YAML

```yaml
inference_backend:
  type: nano_vllm               # or multi_gpu
  devices: auto                 # "auto" / [0,1] / ["cuda:0"]
  # nano_vllm tunables — both `max_num_seqs` and `concurrency_multiplier`
  # are *per-GPU*; the backend scales the driver-side admission pool by
  # `len(devices)` internally, so these values stay fixed across GPU counts.
  max_num_seqs: 32              # scheduler depth per server (= per GPU)
  max_num_batched_tokens: 8192
  max_model_len: 4096
  gpu_memory_utilization: 0.90
  concurrency_multiplier: 2     # driver oversubscription per server;
                                # global in-flight cap =
                                # max_num_seqs × concurrency_multiplier × len(devices)
  load_audio_vae: false         # true iff you need return_audio or encode_reference_wav
  inference_timesteps: 10
  # multi_gpu tunables
  # batch_size: 32
  # return_latent_dtype: float32
```

## SVS scenarios

`infer_vocalrender_svs.py` auto-routes on the YAML's `inference_backend.type`:

- `multi_gpu` (default) — full feature parity: supports `use_prompt_audio`,
  `save_score`, and `lyrics_only` rebuild.
- `nano_vllm` — covers plain V2 SVS and `use_prompt_audio`
  (`same_song` / `static_ref` / `pre_extracted`) via wrapper-side
  `prompt_audio_feats → ref_audio_latents` byte translation, and
  `save_score` (the GT score notation is rendered driver-side from the
  dataset row's `bpm/word/pitch/note`, so it's identical to the multi_gpu
  output regardless of backend). Reference audio is decoded by a
  driver-side standalone `AudioVAE` *after* the backend has shut down (so
  nano-vllm workers release VRAM first, avoiding the double-GPU OOM that
  would otherwise hit at `gpu_memory_utilization ≥ 0.9`).
