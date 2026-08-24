# Evidence map: VocalRender to Suno-quality music creation

Final map after discovery, contradiction deepening, source verification, and a live hardware check on 2026-08-16.

| Decision claim | Best supporting evidence | Counterevidence / caveat | Final status |
|---|---|---|---|
| “Suno quality” now includes a workstation, not just one-click song quality. | Suno v5.5, Song Editor, Studio, and Studio 2 expose stems, section replacement, MIDI, effects, automation, and multitrack export. | Marketing pages do not prove every workflow is excellent. | Confirmed product target |
| VocalRender has a valuable but narrow edge: striking identity in the first ~2–3 seconds. | Direct Experiment 05 listening notes and measurements. | Identity deteriorates later; WER was about 27%; the source voice/data are not shippable. | Confirmed locally |
| Re-anchored short chunks are the highest-information next VocalRender experiment. | The failure begins after roughly one prompt-length window; an independent VoxCPM issue reports the same self-conditioning drift and proposes prompt reinjection. | No maintainer or controlled evidence proves that crossfaded chunks will preserve phrasing or remove seams. | Testable hypothesis, not a fix |
| VocalRender should remain in the system, but should not be forced to compose, arrange, sing, mix, and master an entire song. | Its current strength is timbre/score control; current frontier systems separate global planning, local rendering, stems, or refinement. | A later VocalRender architecture could absorb more of the stack. | Recommended architecture |
| ACE-Step 1.5 is the best immediate open full-song backbone to test on the machine as currently visible. | Released MIT model; 2B path fits documented 8–16 GB configurations; supports reference, cover, repaint, LoRA, and long songs. | Quality claims are author-run; its own table trails Suno v5 on some alignment measures. | Confirmed available; quality must be baked off |
| MiniMax Music3 is the highest-upside newly released open-weight full-song candidate. | Fresh official release, five-minute songs, hierarchical planner/renderer, structured vocal/arrangement prompts, public weights. | Official inference requires two CUDA GPUs; only the 12 GB RTX 4070 is currently visible. License adds attribution, consent, and revenue conditions. | High-priority when both GPUs are available |
| HeartMuLa is a credible secondary baseline, not the primary voice-identity engine. | Released Apache-2.0 3B model with single-GPU lazy loading and multi-minute generation. | Released reference conditioning is incomplete; the paper’s reference representation excludes speaker timbre; vendor rankings conflict with other papers. | Secondary bake-off |
| LeVo2 cannot be the current foundation despite strong paper results. | The paper reports excellent lyrical-song quality and a sensible hierarchical vocal/accompaniment design. | Official repository returned 404 and checkpoint access failed during live verification; previously published terms were restrictive. | Demoted until release/access changes |
| Stable Audio 3 Medium is a strong accompaniment, sound-design, and inpainting component. | Released long-form 1.4B model with sub-7 GB documented peak VRAM, licensed data, and audio editing. | Official examples/model card center instrumental audio, not lead lyrical singing. | Specialist, not song core |
| SoulX-Singer SVC and Vevo2 deserve a fresh vocal-conversion bake-off. | Both now expose singing conversion/editing capabilities that were not fully represented by the earlier local SVS comparison. | Product licensing/data provenance for Vevo2 needs a model-level audit; conversion quality depends on a good source performance. | High-priority vocal layer tests |
| Lyria 3 Pro is an unusually cheap commercial quality ceiling. | Official Gemini API offers full songs for $0.08/request, custom lyrics, section timing, 44.1 kHz stereo, and SynthID. | It is preview-only, single-turn, has restrictive rate limits, blocks artist imitation, and does not provide iterative editing. | Benchmark and optional beta backend |
| Eleven Music v2 is a strong editing/API reference but a poor invisible core dependency for this product. | Up to five minutes, references, section generation, inpainting, and $0.15/minute list pricing. | Pure-play music products must co-brand; resale and some media rights are restricted. | Benchmark or negotiated enterprise option |
| Training a frontier foundation model from scratch is not rational now. | Current systems disclose billions of parameters, hierarchical stacks, tens of millions of songs, and training on large GPU fleets. | A narrow licensed-data fine-tune or adapter is realistic. | Confirmed strategic boundary |
| Current rankings are not trustworthy without a same-prompt listener bake-off. | ACE, HeartMuLa, LeVo2, and newer papers use different model versions, prompts, raters, and metrics; automated audio judges have documented bias. | Human panels are slower and still need a disciplined protocol. | Confirmed evaluation requirement |
| Celebrity imitation cannot be the product wedge. | Local source restrictions, Tennessee voice law, service policies, and model licenses converge on consent; current commercial services require owned/authorized voices. | Exact legal exposure varies by jurisdiction and use; this is not legal advice. | Hard product boundary |

## Decision synthesis

The evidence favors a **model-agnostic hybrid workstation**:

1. A replaceable full-song backbone plans and renders the song.
2. Stems expose the lead vocal, accompaniment, and section-level regeneration.
3. VocalRender remains a short-window, score-controlled identity experiment.
4. SoulX-Singer, Seed-VC, and Vevo2 compete as post-generation voice layers.
5. Stable Audio 3 handles accompaniment alternatives, texture, and inpainting.
6. Lyria 3 Pro and Eleven Music establish the commercial quality ceiling.

This preserves the only locally demonstrated standout—the opening voice identity—without betting the product on its unsolved long-form failure.
