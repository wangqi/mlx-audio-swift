# mlx-audio-swift Upgrade: tag-20260412 → tag-20260419

## Summary

Six commits merged between 2026-04-12 and 2026-04-19: three bug fixes, one ASR feature, one TTS performance enhancement, and one build housekeeping change.

---

## Changes

### 1. Fix: Kokoro textProcessor Not Forwarded in fromPretrained (#151)
**Severity: High**

`KokoroModel.fromPretrained(_:cache:textProcessor:)` accepted a `textProcessor` parameter but silently discarded it — never passing it through to `fromModelDirectory`. This bypassed the G2P pipeline (MisakiTextProcessor / KokoroMultilingualProcessor), sending raw English text directly to the character-level phoneme tokenizer and producing garbled speech output for all Kokoro models.

Fix: one-line pass-through added in `KokoroModel.fromPretrained`.

**Risk: None** — internal fix, no API change. `TTS.loadModel()` path corrected automatically.

---

### 2. Feature: Qwen3 ASR Context Support (#126)
**Severity: Enhancement**

`Qwen3ASRModel` now accepts a `context` string in `generate()` and `generateStream()`, allowing prior transcript text to guide decoding. Useful for multi-segment audio with technical vocabulary or speaker names.

The `STTGenerationModel` protocol conformance maps `STTGenerateParameters` → `context: ""` by default, so all existing callers are unaffected.

**Risk: None** — backward-compatible; empty context = prior behavior.

---

### 3. Performance: Qwen3 TTS Reference Audio Caching (#125)
**Severity: Enhancement (performance)**

`Qwen3TTS` now caches the speaker embedding computed from a reference audio array between successive generation calls. Cache is keyed on `ObjectIdentifier` of the `MLXArray` instance — when the same reference is reused across chunks, the speaker encoder runs only once instead of once per chunk.

Our `MLXAudioSpeaker.speakText()` already loads one `refAudioArray` and passes it to every chunk's `generateSamplesStream` call within a single session. This caching eliminates redundant speaker encoding for all multi-chunk voice-clone TTS sessions.

**Risk: Low** — cache is keyed on object identity; a new `MLXArray` (new speak call, new clone profile) bypasses the cache correctly. Memory overhead: one `ReferenceAudioContext` (speaker embedding tensor) kept alive per loaded model.

---

### 4. Fix: CohereTranscribe Quantized Checkpoint Loading (#153)
**Severity: High**

`CohereTranscribeModel` previously failed to load Python-produced quantized (int8/int4) checkpoints with opaque key-not-found errors. Fix adds:
- Python → Swift key prefix alias mapping (encoder/decoder/head prefixes)
- Structural Q/K/V weight merge for Conformer and Transformer decoder blocks
- Companion key derivation (`.scales` / `.biases` from base key)
- Pre-load inventory validation with actionable diagnostics

Our `MLXAudioASR.makeSTTModel()` routes Cohere repos to `CohereTranscribeModel.fromDirectory()` — the fix is transparent.

**Risk: None** — loading logic is internal; calling convention unchanged.

---

### 5. Build: Exclude In-Tree READMEs from Package Targets (#152)
**Severity: Housekeeping**

`Package.swift` updated to exclude nested `README.md` files from SwiftPM target source sets. No runtime or API changes.

**Risk: None** — build-time only.

---

## iOS-Specific Impact

| Change | iPhone / iPad Effect |
|--------|---------------------|
| Kokoro textProcessor fix | Kokoro TTS now produces correctly phonemized speech (previously garbled on iOS) |
| Qwen3 TTS caching | Voice-clone multi-chunk sessions faster; speaker encoding skipped after first chunk |
| Qwen3 ASR context | No behavior change; empty context = prior behavior |
| CohereTranscribe quant fix | Quantized (smaller, 8/4-bit) Cohere STT models now load correctly on device; better iOS memory compatibility |

---

## Upgrade Risk Assessment

**Overall risk: Low**

- All three bug fixes are internal to model classes with no API surface changes.
- The performance enhancement is additive and backward-compatible.
- `Package.swift` target structure is unchanged from our integration perspective.
- The `CohereTranscribeModel.fromDirectory()` calling convention in `MLXAudioASR.makeSTTModel()` is unchanged.
- iOS memory: Qwen3 TTS caches one `ReferenceAudioContext` per model instance — negligible.

---

## Required Code Changes in App

**None required.** All fixes and improvements apply automatically via the updated library.

### Optional Enhancement: Qwen3 ASR Context for Batch Transcription

`MLXAudioASR.transcribe()` currently passes `STTGenerateParameters(language:chunkDuration:)` with no context. The new `context` parameter in Qwen3 ASR could improve accuracy for multi-turn dictation by passing previously transcribed text. This would require adding a `context` field to `STTGenerateParameters` or `ASRProtocol.transcribe()`. Not urgent — the streaming path (`StreamingInferenceSession`) handles context internally.
