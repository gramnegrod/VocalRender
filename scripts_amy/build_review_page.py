#!/usr/bin/env python3
"""Build a local HTML review page for listening to renders side by side.

Writes outputs/review.html with <audio> players pointing at the wav files by
relative path, so it works offline in a browser with no server and no copying
of audio. Transcribes each embedded file at build time so the text shown is
what is actually in that file, rather than a transcript from some other run.

    python scripts_amy/build_review_page.py
    start outputs/review.html
"""

from __future__ import annotations

import html
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs" / "review.html"


def load_eval_wer():
    spec = importlib.util.spec_from_file_location(
        "eval_wer_mod", ROOT / "scripts_amy" / "eval_wer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ev = load_eval_wer()
    entries = {e["item_name"]: e
               for e in json.load(open(ROOT / "data/amy/annotations_val.json",
                                       encoding="utf-8"))}

    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
    cache: dict[str, str] = {}

    def transcribe(rel: str) -> str:
        p = ROOT / "outputs" / rel
        if not p.exists():
            return ""
        if rel in cache:
            return cache[rel]
        segs, _ = asr.transcribe(str(p), language="en", beam_size=5)
        txt = " ".join(s.text for s in segs).strip()
        cache[rel] = txt
        return txt

    # --- three-way comparison rows -------------------------------------
    trio = []
    for chunk in ("000", "009", "011"):
        item = f"05-Back-To-Black__chunk{chunk}"
        ref = " ".join(ev.reference_words(entries[item])) if item in entries else ""
        cols = [
            ("Real Amy", f"ab_test/{chunk}_1_REAL-AMY.wav"),
            ("VocalRender @ cfg 3.0", f"ab_test/{chunk}_2_VOCALRENDER_amy-voice_bad-words.wav"),
            ("SoulX-Singer", f"ab_test/{chunk}_3_SOULX_good-words_wrong-voice.wav"),
        ]
        trio.append((item, ref, [(lbl, rel, transcribe(rel)) for lbl, rel in cols]))

    # --- all cfg 3.0 renders for extended listening ---------------------
    halluc_dir = ROOT / "outputs" / "halluc" / "vocalrender_v2"
    extended = []
    for wav in sorted(halluc_dir.glob("*.wav")):
        item = wav.stem
        rel = f"halluc/vocalrender_v2/{wav.name}"
        ref = " ".join(ev.reference_words(entries[item])) if item in entries else ""
        extended.append((item, ref, rel, transcribe(rel)))

    def player(rel: str) -> str:
        return f'<audio controls preload="none" src="{html.escape(rel)}"></audio>'

    trio_html = ""
    for item, ref, cols in trio:
        cards = ""
        for lbl, rel, hyp in cols:
            tone = ("real" if "Real" in lbl else
                    "vr" if "VocalRender" in lbl else "sx")
            cards += f"""
          <div class="card {tone}">
            <h4>{html.escape(lbl)}</h4>
            {player(rel)}
            <p class="hyp">{html.escape(hyp) or "&mdash;"}</p>
          </div>"""
        trio_html += f"""
      <section class="trio">
        <div class="trio-head"><span class="item">{html.escape(item)}</span>
          <span class="ref">score lyrics: &ldquo;{html.escape(ref)}&rdquo;</span></div>
        <div class="cards">{cards}
        </div>
      </section>"""

    ext_html = ""
    for item, ref, rel, hyp in extended:
        ext_html += f"""
        <tr>
          <td class="mono">{html.escape(item)}</td>
          <td>{player(rel)}</td>
          <td class="ref-cell">{html.escape(ref)}</td>
          <td class="hyp-cell">{html.escape(hyp)}</td>
        </tr>"""

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VocalRender listening review</title>
<style>
 :root {{ --ink:#16203a; --muted:#5b6677; --line:#e4e9f2; --blue:#2563eb;
          --green:#15803d; --amber:#b45309; --red:#b91c1c; --soft:#f6f8fc; }}
 *{{box-sizing:border-box;margin:0;padding:0}}
 body{{font:15px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);
       background:#eef1f6;padding:28px 16px}}
 .page{{max-width:1060px;margin:0 auto;background:#fff;padding:40px 44px 56px;
        border-radius:10px;box-shadow:0 1px 30px rgba(22,32,58,.09)}}
 h1{{font-size:28px;letter-spacing:-.02em;margin-bottom:8px}}
 .sub{{color:var(--muted);margin-bottom:22px}}
 h2{{font-size:19px;margin:34px 0 12px;padding-top:18px;border-top:2px solid var(--line)}}
 h2:first-of-type{{border-top:none}}
 table{{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0 6px}}
 th{{text-align:left;font-size:11px;letter-spacing:.06em;text-transform:uppercase;
     color:var(--muted);padding:8px 10px;border-bottom:2px solid var(--line)}}
 td{{padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:middle}}
 .mono{{font-family:ui-monospace,Consolas,monospace;font-size:12px;white-space:nowrap}}
 audio{{height:34px;width:250px}}
 .trio{{border:1px solid var(--line);border-radius:9px;padding:16px 18px;margin-bottom:14px}}
 .trio-head{{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:baseline;margin-bottom:12px}}
 .item{{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;font-weight:600}}
 .ref{{color:var(--muted);font-size:13px}}
 .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}
 .card{{border:1px solid var(--line);border-radius:8px;padding:12px 14px;background:var(--soft)}}
 .card h4{{font-size:12.5px;margin-bottom:8px}}
 .card.real h4{{color:var(--green)}} .card.vr h4{{color:var(--blue)}}
 .card.sx h4{{color:var(--amber)}}
 .hyp{{font-size:12.5px;color:#2b3950;margin-top:8px;font-style:italic}}
 .ref-cell{{color:var(--muted);font-size:12.5px}}
 .hyp-cell{{font-size:12.5px;font-style:italic}}
 .note{{background:var(--soft);border-left:4px solid var(--blue);padding:14px 18px;
        border-radius:4px;margin:14px 0;font-size:14px}}
 .warn{{background:#fffaf2;border-left:4px solid var(--amber)}}
 .big{{font-size:26px;font-weight:700}}
 .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:16px 0 4px}}
 .kpi{{border:1px solid var(--line);border-radius:8px;padding:14px 16px}}
 .kpi span{{display:block;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}}
 .win{{color:var(--green)}} .lose{{color:var(--red)}}
</style></head><body><div class="page">

<h1>VocalRender listening review</h1>
<p class="sub">Everything below was rendered on this machine. Audio plays from
disk &mdash; no network. Transcripts were produced by Whisper large-v3 from the
exact file next to them.</p>

<div class="kpis">
  <div class="kpi"><span>VocalRender @ cfg 3.0</span><b class="big">27.0%</b> word errors</div>
  <div class="kpi"><span>Voice identity</span><b class="big win">9.0</b> = real Amy (9.0)</div>
  <div class="kpi"><span>SoulX words</span><b class="big">22.5%</b> not sig. better (p=0.10)</div>
  <div class="kpi"><span>SoulX voice</span><b class="big lose">2&ndash;4</b> &ldquo;thin, lacks grit&rdquo;</div>
</div>

<h2>1. Three-way comparison</h2>
<p class="sub">Same line, same score, same prompt clip. Listen top to bottom.</p>
{trio_html}

<h2>2. The setting that changed everything</h2>
<table>
 <tr><th>cfg_value</th><th>mean WER</th><th>sd</th><th></th></tr>
 <tr><td>2.0 (old default)</td><td>42.55%</td><td>8.51</td><td></td></tr>
 <tr><td><b>3.0</b></td><td><b>26.95%</b></td><td><b>2.13</b></td><td class="win">best, and tightest spread</td></tr>
 <tr><td>4.0</td><td>35.70%</td><td>7.81</td><td>worse again</td></tr>
 <tr><td>5.0</td><td>35.22%</td><td>2.87</td><td>worse again</td></tr>
</table>
<div class="note">Pooled, cfg 3.0 against every other setting:
 <b>&minus;10.89 points, Welch p = 0.0020</b>. It is a minimum, not a trend &mdash;
 pushing guidance higher makes it worse.</div>

<h2>3. All cfg 3.0 renders</h2>
<p class="sub">The full held-out set, for a longer listen than three clips.</p>
<table>
 <tr><th>item</th><th>audio</th><th>score lyrics</th><th>what Whisper heard</th></tr>
 {ext_html}
</table>

<h2>4. Read this before trusting the numbers</h2>
<div class="note warn">
<b>The evaluation is weak.</b> 15 segments, 141 reference words, and the
&ldquo;correct&rdquo; lyrics were themselves produced by Whisper, which scores
35&ndash;45% word error on real human singing. Differences smaller than about
10 points are not measurable here. Every figure above is n=3 with a p-value for
that reason.</div>
<div class="note warn">
<b>Two code paths disagree by 8 points</b> at nominally identical settings, so
absolute WER is only comparable within one script. The cfg comparison above is
internally consistent (all through <span class="mono">eval_wer.py</span>).</div>
<div class="note warn">
<b>The 9.0 voice score deserves your ears, not just the judge.</b> It came from
an LLM audio judge with real Amy scoring 9.0 in the same batch, on three short
clips. Whether VocalRender really is indistinguishable from Amy is exactly what
section 1 is for.</div>

</div></body></html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc, encoding="utf-8")
    print(f"Wrote {OUT}  ({len(extended)} extended clips, {len(trio)} comparisons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
