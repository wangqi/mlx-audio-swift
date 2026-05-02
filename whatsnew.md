# mlx-audio-swift Upgrade: tag-20260425 → tag-20260502

## Summary

Three upstream commits merged in this period. The primary addition is a new **MOSS-TTS-Nano** TTS model; the rest are library-wide cancellation reliability improvements.

---

## New Features

### MOSS-TTS-Nano TTS Model (`dfb9382`)

A new lightweight, fast bilingual (Chinese/English) TTS model based on a GPT2-style transformer architecture with SNAC audio tokenizer codebooks.

**Architecture:**
- Dual-transformer design: a global `MossGPT2Model` transformer for text-to-audio-token generation, plus a local transformer for fine-grained audio token refinement
- Multi-codebook SNAC audio tokenizer for encoding/decoding reference audio (voice cloning)
- `SentencePieceTokenizer` now supports both **Unigram** and **BPE** model types — the BPE variant is required by MOSS-TTS-Nano

**Capabilities:**
- Voice cloning: pass `refAudio` + `refText` to reproduce a target speaker's timbre
- Default generation parameters: `temperature=0.7`, `topP=0.9`, `topK=50`, `repetitionPenalty=1.1`, `maxTokens=375`
- Sample rate: determined by `config.audioTokenizerSampleRate` (typically 24 kHz)

**App impact:**
- Available via `TTS.loadModel(modelRepo:)` when model config declares `model_type: "moss_tts_nano"`
- Our `MLXAudioSpeaker` loads it transparently via the generic `generateSamplesStream` path — no code changes required
- Voice clone profiles work out of the box since MOSS-TTS-Nano accepts the same `refAudio`/`refText` signature

**Dependencies:**
- `Package.resolved` updated (new `originHash`) — standard submodule update behavior, no action needed

---

## Bug Fixes

### Improved Task Cancellation Across All TTS Models (`fc4fe22`)

All TTS model `generateStream()` implementations now properly:
- Call `Task.checkCancellation()` at key inference checkpoints (between chunk generations, before/after model forward pass)
- Wire `AsyncThrowingStream.Continuation.onTermination` so that if the consumer drops the stream, the internal generation `Task` is also cancelled

**Affected models:** Chatterbox, EchoTTS, FishSpeech, LlamaTTS, MarvisTTS, PocketTTS, Qwen3, Qwen3TTS, Soprano, KittenTTS (StyleTTS2), KokoroModel (StyleTTS2)

**App impact:**
- Our `MLXAudioSpeaker.stopSpeaking()` already cancels the wrapping Swift `Task`. These library changes make the library's internal inference loops respond to that cancellation faster — reducing audio generation stall time after `stopSpeaking()` is called
- No code changes required in `MLXAudioSpeaker`; the improved behavior is transparent

### Qwen3ASR Stream Cancellation Propagation (`d810daf`)

`Qwen3ASRModel.generateStream()` now correctly propagates Swift Task cancellation into the streaming inference session.

**App impact:**
- `MLXAudioASR.cancel()` sets `isCancelled = true` and our loop checks it. With this fix, the Qwen3ASR `generateStream()` now also terminates on Task cancellation — reducing the latency between `cancel()` and transcription stopping
- No code changes required

---

## Risks and Assessment

### Risk Level: **Low**

| Area | Risk | Mitigation |
|------|------|------------|
| MOSS-TTS-Nano loading | Low — follows standard `TTS.loadModel` factory pattern; only active if a MOSS model is configured | Gated by model config `model_type` field; existing models unaffected |
| BPE tokenizer support | Low — additive change to `SentencePieceTokenizer`; unigram path unchanged | Regression-free: unigram models fall back to same logic |
| Cancellation changes | Very low — makes existing cancellation more reliable; no behavior change when not cancelled | All changes are cooperative (add checkpoints, not interrupt-driven) |
| Package.resolved update | Minimal — only reflects dependency version pins used by the SPM package itself | Does not affect our xcframework-based integration |

### iOS Device Considerations

- MOSS-TTS-Nano has a relatively small memory footprint for a voice-cloning TTS model (GPT2-based, ~400M parameters typical for nano variants). Suitable for A-series devices with 6 GB+ RAM.
- The cancellation improvements are especially beneficial on iOS where the main thread is more resource-constrained — faster cancellation means less GPU/ANE time wasted after the user stops playback.
- No new OS APIs introduced; changes are pure Swift/MLX.

---

## App Code Changes Required

**None required in MLXAudioSpeaker or MLXAudioASR.** Both benefit from all three commits automatically:
- MOSS-TTS-Nano: loaded via existing `TTS.loadModel` factory with the `moss_tts_nano` model type, then uses the standard `generateSamplesStream` path
- Cancellation fixes: transparent improvement to existing task cancellation flow

The only app changes are adding MOSS-TTS-Nano to the supported model list in `LocalModelAboutView.swift` and bumping the engine version/date.
