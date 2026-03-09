# mlx-audio-swift Upgrade: tag-20260302 to tag-20260309

## Summary

This upgrade merges 5 upstream commits (excluding merge commits) from the Blaizzy/mlx-audio-swift main branch, covering two main areas: Qwen3 TTS performance improvements, Parakeet V2 accuracy fixes, and a major new feature — MLXAudioLID (Language Identification).

---

## New Features

### MLXAudioLID: Spoken Language Identification Module (#80)

A new `MLXAudioLID` Swift Package Manager module has been added, implementing Wav2Vec2-based language identification supporting 256 languages (MMS-LID-256 by Facebook).

**Architecture:**
- Full Wav2Vec2 backbone: 7-layer feature extractor, feature projection, positional conv embedding, 48 transformer encoder layers
- Custom attention with HuggingFace-compatible key names
- Auto-normalization (zero-mean, unit-variance) in `predict()`
- Weight norm precomputed in `sanitize()` for efficiency

**Second model: ECAPA-TDNN (VoxLingua107)**
- Supports 107 languages via SpeechBrain ECAPA-TDNN architecture
- ~16x faster and ~47x smaller than MMS-LID-256
- GPU-accelerated mel spectrogram (Hamming, HTK, top_db=80)
- Includes TDNNBlock, Res2NetBlock, SEBlock, SERes2NetBlock, ASP layers

**Implementation details:**
- Shared ECAPA-TDNN backbone extracted to codec module for reuse
- LID CLI demo tool added
- 35+ unit tests across both models
- Integration tests gated behind `MLXAUDIO_ENABLE_NETWORK_TESTS` env var

### VoicesApp: Speech-to-Speech (STS) View (#78)

A new STS (Speech-to-Speech) view added to the VoicesApp demo application. This is a demo/example app change only; no production library API was altered.

---

## Bug Fixes

### Parakeet V2 Mel Scale Fix (#86)

**Problem:** Parakeet V2 (and V3) models were trained with NeMo's `AudioToMelSpectrogramPreprocessor` using the HTK mel scale, but the library was using the Slaney scale, causing accuracy degradation.

**Fix:**
- Changed mel filter scale from `.slaney` to `.htk` for Parakeet models
- Added `eval(jointOut)` calls in both TDT and standard RNN-T decode paths to ensure joint network output is materialized before argmax
- Added `parakeet-tdt-0.6b-v2` to the supported models list

**Impact:** This is a correctness fix. Transcription accuracy for Parakeet V2 models improves significantly. Models already in production were producing slightly incorrect outputs.

---

## Performance Improvements

### Qwen3 TTS Performance (#81, #82)

Two successive rounds of performance improvements for Qwen3 TTS:
- Reduced latency and improved throughput for Qwen3 TTS inference on Apple Silicon
- RTF reporting changed from RTF (Real-Time Factor) to RTFx (inverse, higher = faster) for clearer benchmarking

**Impact:** Qwen3 TTS generation speed improves on iPhone and Mac. Both PRs target the Qwen3 TTS generation pipeline with internal optimizations.

---

## Risk Assessment

| Area | Risk Level | Notes |
|------|-----------|-------|
| Qwen3 TTS performance changes | **Low** | Internal optimization only; outputs should be identical |
| Parakeet V2 mel scale fix | **Medium** | Correct fix, but existing users may notice transcription output differences |
| MLXAudioLID new module | **Low** | Additive new module; no existing API changed |
| ECAPA-TDNN shared backbone extraction | **Low** | Refactor of codec module; existing codec behavior unchanged |
| VoicesApp STS view | **None** | Demo app only |

### Key Considerations for iOS

1. **MLXAudioLID is not yet integrated** into the main app. Adding LID support will require new model download flows and UI. The MMS-LID-256 model is ~3.9GB — not suitable for default download; ECAPA-TDNN (~80MB) is the practical option for on-device use.
2. **Parakeet V2 mel scale fix**: If any users have been using Parakeet V2 for ASR, they may notice improved (but different) transcriptions after upgrade. This is expected and correct behavior.
3. **Qwen3 TTS changes**: No API-level changes. App-level integration is unaffected; users benefit from faster generation automatically.
4. **No breaking API changes** detected across all 5 commits. Existing integrations for TTS, ASR, VAD, Diarization, STS remain unchanged.

---

## Upgrade Recommendation

**Proceed with upgrade.** All changes are either additive (LID module) or corrective (Parakeet mel scale, Qwen3 performance). No breaking changes to existing APIs. The Parakeet mel scale fix is the most impactful correctness improvement and should be rolled out to users as soon as possible.
