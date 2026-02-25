# mlx-audio-swift Upgrade: tag-20260218 → tag-20260224

**Date:** 2026-02-24
**Commits:** 16 (non-merge)
**Lines changed:** +4,872 / -828

---

## New Features

### 1. LFM-2.5-Audio Model (Speech-to-Speech) — PR #53
- Added full LFM-2.5-Audio model implementation under `MLXAudioSTS`
- New files: `Conformer.swift`, `Detokenizer.swift`, `LFMAudioConfig.swift`, `LFMAudioModel.swift`, `Processor.swift`, `Transformer.swift`
- Includes STS smoke tests (`MLXAudioSTSTests.swift`, `MLXAudioSmokeTests.swift`)
- New `STSModel` factory class for loading STS models (mirrors TTS factory pattern)

### 2. Smart Turn v3 VAD Model — PR #64
- New voice activity detection model under `MLXAudioVAD/Models/SmartTurn/`
- Files: `SmartTurn.swift`, `SmartTurnConfig.swift`, `SmartTurnFeatures.swift`
- Includes comprehensive unit tests (`MLXAudioVADTests.swift`, 290 lines)
- Enables semantic turn-taking detection for conversational AI

### 3. Qwen3TTS Streaming Audio Chunks — PR #70
- `generateVoiceDesign()` now accepts an `onAudioChunk` callback for real-time audio streaming
- Audio chunks are yielded during generation (~2s chunks at 12.5Hz codec rate)
- Overlap/crossfade between chunks for smooth transitions (25-token context window)
- Non-streaming path preserved for backward compatibility

### 4. Streaming PCM Buffer Support — PR #63
- New `Generation.swift` extensions: `generateSamplesStream()` and `generatePCMBufferStream()`
- New `PCMStreamConverter` utility for format conversion during streaming
- `AudioPlayer` (renamed from `AudioPlayerManager`) gains `play(stream:)` for `AsyncThrowingStream<AVAudioPCMBuffer>` playback
- Speaking-state callbacks (`onSpeakingStateChanged`, `onDidFinishStreaming`)

### 5. Semantic VAD Example — PR #67
- New `ConversationController` and `SemanticVAD` example in `Examples/SimpleChat/`
- Demonstrates smart turn-taking with the v3 model
- Replaces old `SimpleVAD` / `SpeechController` examples

### 6. Configurable Cache Location — PR #60
- All `fromPretrained()` and `resolveOrDownloadModel()` methods now accept `cache: HubCache` parameter
- Defaults to `HubCache.default` — fully backward compatible
- Enables custom model storage locations (useful for iOS App Groups, external storage)

### 7. Speech-Text Alignment Timestamps — PR #74
- CLI tool (`mlx-audio-swift-tts`) gains `--timestamps` flag
- Emits token-level timing data for speech-to-text alignment

---

## Bug Fixes & Improvements

### 8. Swift 6.2 Compilation Fixes — PR #66
- Added `@Sendable` to all `progressHandler` closure parameters (3 files)
- Fixed `[.all]` → `.all` for OptionSet type inference (13 files)
- Resolves Xcode 26.2 / Swift 6.2 compilation errors

### 9. Qwen3 TTS Quantization Fix — PR #61
- Fixed weight loading for quantized Qwen3 TTS checkpoints
- Now properly detects `.scales` tensors and applies `quantize()` before loading talker weights
- Supports both global and per-layer quantization configurations

### 10. macOS Audio Entitlements Fix — PR #69
- Added audio input entitlements for macOS (`VoicesApp.entitlements`)
- Improved file validation: checks file size > 0 for required files in cache
- Clears incomplete cached models automatically

### 11. Dead Code Removal — PR #62
- Removed `ConvWeighted.swift` (123 lines) and `MLX+Extensions.swift` (184 lines)
- Fixed all compiler warnings

### 12. Audio Resampling Support
- `loadAudioArray(from:sampleRate:)` now accepts optional target sample rate
- New `resampleAudio(_:from:to:)` public function using `AVAudioConverter`
- Improved error types in `AudioUtils.AudioUtilsErrors` with `LocalizedError` conformance

---

## API Breaking Changes

| Change | Old API | New API | Backward Compat |
|--------|---------|---------|-----------------|
| TTS factory rename | `TTSModelUtils.loadModel()` | `TTS.loadModel()` | Yes — `typealias TTSModelUtils = TTS` provided |
| Error type rename | `TTSModelUtilsError` | `TTSModelError` | Yes — deprecated typealias provided |
| Audio player rename | `AudioPlayerManager` | `AudioPlayer` | **No** — no typealias provided |
| File rename | `TTSModelUtils.swift` | `TTSModel.swift` | N/A (same module) |
| File rename | `AudioPlayerManager.swift` | `AudioPlayer.swift` | N/A (same module) |
| Published setters | `AudioPlayer.isPlaying` was `public set` | Now `public private(set)` | **Breaking** if set externally |
| Cache parameter | `fromPretrained(_:)` | `fromPretrained(_:cache:)` | Yes — default parameter |
| loadAudioArray | `loadAudioArray(from:)` | `loadAudioArray(from:sampleRate:)` | Yes — default parameter |

### New Dependency
- `Package.swift` now requires `mlx-swift >= 0.30.6` (was `>= 0.30.3`)
- `MLXAudioSTS` target adds dependency on `MLXLLM` (from `mlx-swift-lm`)

---

## Risk Assessment

### Overall Risk: LOW

**Rationale:** The MLXAudio frameworks are linked into both iOS and macOS targets but **no application code currently imports or calls any MLXAudio APIs**. The only reference is in `AboutView.swift` as an open-source acknowledgement. Therefore, these changes affect only the compiled binary size and transitive dependency resolution.

### Risk Breakdown

| Area | Risk | Detail |
|------|------|--------|
| **Compilation** | LOW | Swift 6.2 fixes (#66) actually *improve* build compatibility. The new `MLXLLM` dependency in `MLXAudioSTS` may require updating the Xcode project if not already linked. |
| **Binary size** | LOW-MEDIUM | +4,872 lines of new code (LFM-2.5-Audio, Smart Turn v3, PCMStreamConverter, Generation.swift) will increase binary size even if unused. Dead code removal (-307 lines) partially offsets this. |
| **API stability** | LOW | `AudioPlayerManager` → `AudioPlayer` rename is breaking, but our app doesn't use it. `TTSModelUtils` → `TTS` rename has backward-compat typealias. |
| **Dependency chain** | LOW | mlx-swift minimum bumped from 0.30.3 to 0.30.6 — minor version bump, unlikely to conflict. New `MLXLLM` dependency is already in the project via mlx-swift-lm. |
| **Runtime** | NONE | No app code calls these APIs, so zero runtime risk. |
| **Future integration** | POSITIVE | Streaming TTS, configurable cache, and PCM buffer streams are valuable features for future TTS integration. Smart Turn v3 VAD enables better voice conversation UX. |

### Action Items Before Integration

1. **Verify build succeeds** on both iOS and macOS targets after submodule update
2. **Check if `MLXLLM` framework** needs to be added to Xcode project link phase (new dependency of `MLXAudioSTS`)
3. **Monitor binary size** delta — new models add ~5K lines of compiled code
4. When eventually using TTS APIs, use `TTS.loadModel()` (not the deprecated `TTSModelUtils`)
5. When eventually using audio playback, use `AudioPlayer` (not `AudioPlayerManager`)
