#!/usr/bin/env bash
# Render "Smoke and Honey" (3 phrases) with VocalRender, Amy-timbre prompt audio.
set -e
cd "$(dirname "$0")"

PY=".venv/Scripts/python.exe"
CKPT="pretrained_models/VocalRender"
JSON="examples/smoke_and_honey.json"
PROMPT="/c/ai/local-llm/seed-vc/_amy_train_data/06-Love-Is-A-Losing-Game__chunk005.wav"

mkdir -p outputs

for ITEM in smoke_01_intro smoke_02_verse smoke_03_hook; do
  echo "=== $ITEM ==="
  "$PY" scripts/infer_vocalrender_svs_single.py \
    --ckpt_dir "$CKPT" \
    --json_file "$JSON" \
    --item_name "$ITEM" \
    --prompt_audio "$PROMPT" \
    --output "outputs/${ITEM}.wav"
done

# Stitch the three phrases into one track with short gaps.
"$PY" - <<'PYEOF'
import soundfile as sf, numpy as np
parts = ["outputs/smoke_01_intro.wav", "outputs/smoke_02_verse.wav", "outputs/smoke_03_hook.wav"]
chunks, sr = [], None
for p in parts:
    x, s = sf.read(p)
    if x.ndim > 1:
        x = x.mean(axis=1)
    sr = s
    chunks.append(x.astype(np.float32))
    chunks.append(np.zeros(int(0.35 * s), dtype=np.float32))
song = np.concatenate(chunks[:-1])
peak = float(np.max(np.abs(song))) or 1.0
song = song / peak * 0.95
sf.write("outputs/smoke_and_honey.wav", song, sr, subtype="PCM_16")
print(f"wrote outputs/smoke_and_honey.wav  {len(song)/sr:.2f}s @ {sr} Hz")
PYEOF
