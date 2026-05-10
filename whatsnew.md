# mlx-audio-swift: What's New (tag-20260502 → tag-20260509)

## Upgrade Summary

Merged upstream changes from the Blaizzy/mlx-audio-swift `main` branch into our fork. Three upstream features landed alongside two local bug fixes already in our fork.

---

## New Features

### 1. MOSS-TTS Full Model (`MossTTSModel`)

**Commit:** `7734cd1` — Add MOSS-TTS model family (#179)

The full MOSS-TTS model uses a Qwen3 LLM backbone (instead of MossTTSNano's GPT-2 backbone) for higher-quality bilingual (Chinese/English) speech synthesis. It adds:

- `MossTTSModel` — Qwen3-based dual-transformer TTS with SNAC audio tokenizer
- `MossTTSProcessor` — text tokenization and preprocessing pipeline
- `MossTTSQwen3` — Qwen3 language model frontend
- `MossTTSConfig` — full config with decoder/flow settings
- `MossTTSFullSampling` — sampling strategy matching the Python reference

Already registered in `TTSModel.swift` dispatch table alongside MossTTSNano. No app-level code changes needed; models with matching repo ID will load automatically.

**iOS risk:** Medium. Large model (Qwen3 backbone). Memory budget check via `SystemMemoryHelper.willLoadAudioModel()` is already in place. Voice-cloning mode requires reference audio — same requirement as MossTTSNano.

---

### 2. Qwen3-ASR: Chunked Prefill + Repetition Penalty + Loop Guard

**Commit:** `e5ba0d3` — Qwen3-ASR: chunked prefill + asyncEval + repetition penalty + loop guard (#174)

Fixes a critical runaway-repetition bug on long-form audio (e.g. 57-minute recordings) that could consume up to **22 GB RAM** and stall for minutes.

**Changes to `STTGenerateParameters`:**
- New field `repetitionPenalty: Float` (default `1.0` = disabled, backward-compatible)
- New field `repetitionContextSize: Int` (default `32`)

**Internal Qwen3 improvements:**
- Chunked prefill (window = 2048 tokens) with `eval+clearCache` between chunks
- `asyncEval` pipelining matching `mlx_lm.generate.generate_step`
- `MLX.Memory.clearCache()` every 256 generated tokens
- Sign-aware repetition penalty applied before argmax
- Heuristic fail-safe: stop if last 24 tokens contain <=3 unique IDs

**Recommended call site (now used in MLXAudioASR.swift):**
```swift
STTGenerateParameters(language: lang, chunkDuration: 30.0,
                      repetitionPenalty: 1.15, repetitionContextSize: 32)
```

**iOS risk:** Low. Default `repetitionPenalty = 1.0` is fully backward-compatible. Enabling `1.15` for Qwen3 models is safe and eliminates the runaway-loop crash on long recordings. Performance improvement: ~8% faster on 57-minute audio.

---

### 3. Silero VAD Model (MLX Swift Port)

**Commit:** `4bc0e94` — feat(vad): add Silero VAD model (#176)

Swift/MLX port of the Python `mlx_audio.vad.silero_vad` implementation.

**API surface:**
- `SileroVAD.Model` — wraps dual SileroVADBranch (16kHz + 8kHz)
- **Streaming API:** `initialState()` / `feed(chunk:state:sampleRate:)`
- **Offline API:** `predictProba(audio:sampleRate:)` / `getSpeechTimestamps(audio:threshold:minSpeechDurationMs:...)`
- Compatible with `mlx-community/silero-vad` (v5) and `mlx-community/silero-vad-v6`
- `fromPretrained(repoId:)` downloads from HuggingFace

Our current `MLXAudioVAD.swift` wraps `SmartTurnModel` (endpoint detection). Silero VAD serves a different purpose (presence/absence VAD). Integration path: add a new `MLXAudioSileroVAD` wrapper if needed in a follow-up.

**iOS risk:** Low. New model type, no impact on existing SmartTurn/Endpoint detection flow.

---

## Bug Fixes (Our Fork)

### 4. MossTTSNano: Fixed 2x Slow Speed

**Commit:** `fd4271d` — Bugfix: MossTTSNanoModel fixed the 2x slow speed error

Previously `MossTTSNanoModel` generated audio at half the intended speed due to an incorrect frame step in the SNAC decoder. Fixed in our fork and already merged.

### 5. MossTTSNanoError: Surface Errors to Callers

**Commit:** `4873b33` — Improve: MossTTSNanoError can surface the errors to callers now

`MossTTSNanoError` now conforms to `LocalizedError` and surfaces descriptive messages to the TTS engine for display in the UI or logs.

---

## Risk Assessment

| Change | Risk | Notes |
|--------|------|-------|
| MOSS-TTS Full Model | Medium | Large model; memory guard already in place |
| Qwen3-ASR repetition fix | Low | Backward-compatible; critical fix for long audio |
| Silero VAD | Low | New model type, isolated from existing VAD path |
| MossTTSNano speed fix | None | Already in production since tag-20260502 |
| MossTTSNanoError | None | Additive error surfacing only |

**Overall upgrade risk: Low-Medium.** The Qwen3-ASR fix is urgently needed (prevents 22GB RAM stalls). MOSS-TTS Full is additive. Silero VAD is isolated.
