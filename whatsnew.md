# mlx-audio-swift Upgrade: tag-20260419 → tag-20260425

## Summary

This upgrade merges 9 upstream commits spanning April 21–25, 2026. Major additions: DeepFilterNet speech enhancement (new STS category), Parakeet batch generation with hybrid TDT decoder and bf16 compute, and a Qwen3TTS generation-parameter correctness fix. A package upgrade to mlx-swift-lm 3.31.3 and mlx-swift 0.31.3 is the foundation for all changes.

---

## New Features

### DeepFilterNet Speech Enhancement (STS) — commit 5ddd0f6

A full Swift port of DeepFilterNet V1/V2/V3 for real-time and offline noise reduction, added to `Sources/MLXAudioSTS/`. This is the first true Speech-to-Speech (STS) model in the library.

**Capabilities:**
- Supports V1, V2, and V3 architectures from `mlx-community/DeepFilterNet-mlx`
- Real-time streaming mode with 10 ms hop frames (`DeepFilterNetStreamer`)
- Offline fast path for file processing
- Accelerate-optimized GRU via `vDSP_mmul` for sequential recurrent layers (avoids thousands of tiny GPU dispatches)

**Measured performance (Apple Silicon, release build):**
- V1 offline: 0.24 s per 10.6 s audio (2.3× faster than Python reference)
- V2 offline: 6.4× speedup over naive GRU (0.17 s vs 1.09 s for 10 s audio)
- V3 offline: 3.2× speedup (0.19 s vs 0.61 s for 10 s audio); 2-hour files at ~100× real-time
- V3 streaming RTF: 0.36 (3× real-time capable)

**API:** `DeepFilterNetModel.fromPretrained(repo:subfolder:)` · `enhance(_:)` · `createStreamer(config:)` / `enhanceStreaming(audio:config:)`

**STSModel routing:** `STSModel.fromPretrained()` now auto-routes repos matching `"deepfilter"` or `"dfn"` to DeepFilterNet.

**iOS relevance:** Full Apple Silicon support, no platform restrictions. Well-suited for live call/recording cleanup before TTS input or after STT output.

---

### Parakeet Batch Generation + Hybrid TDT + bf16 API — commit 4ea370d

Major performance and capability upgrade to `ParakeetModel`.

**Batch transcription (`generateBatch`):**
- Batched forward pass with `relPosAttention` fix for heterogeneous-length inputs
- Shape guards for ragged batches
- Encoder compile seam for faster repeated inference

**Hybrid TDT decoder:**
- Batched TDT decoding path alongside the existing CTC path
- `ParakeetModel.fromPretrained(computeDType:)` defaults to `.bfloat16`
- Pre-allocated `currentToken` for stable JIT compilation; single `.asArray` readback

**bf16 compute:**
- `computeDType: DType = .bfloat16` property on `ParakeetModel`
- All float parameters cast to `computeDType` at load time
- `fromPretrained(computeDType:)` and `fromDirectory(computeDType:)` accept the dtype

**iOS relevance:** The single-stream `generateStream` API used by our `MLXAudioASR` benefits automatically from the encoder and attention fixes. No code changes required.

---

### Qwen3TTSReferenceConditioning Public API — commit ae046cc

`Qwen3TTSReferenceConditioning` is now a public `Sendable` struct with a public initializer. This enables callers to pre-compute voice conditioning once and reuse it across generate calls with different text inputs.

**New public surface:**
```swift
public struct Qwen3TTSReferenceConditioning: @unchecked Sendable {
    public let speakerEmbedding: MLXArray?
    public let referenceSpeechCodes: MLXArray
    public let referenceTextTokenIDs: MLXArray
    public let resolvedLanguage: String
    public let codecLanguageID: Int?

    public init(...)
}

// On Qwen3TTSModel:
public func prepareReferenceConditioning(
    refAudio: MLXArray, refText: String, language: String?
) throws -> Qwen3TTSReferenceConditioning

// Overloaded generate APIs that accept pre-computed conditioning:
public func generateStream(conditioning: Qwen3TTSReferenceConditioning, text: String, ...) 
public func generate(conditioning: Qwen3TTSReferenceConditioning, text: String, ...)
```

**Internal note:** The existing `referenceAudioContext(for:)` already caches encoding by `ObjectIdentifier` of the `MLXArray`. The public API is primarily for callers that manage conditioning lifetime explicitly (e.g., batch UI flows, pre-warming before TTS begins).

---

## Bug Fixes

### Qwen3TTS: topK and minP Now Read from GenerateParameters — commit 1faee8e

`resolveVoiceDesignGenerationSettings` previously hardcoded `topK: 50` and `minP: 0.0` regardless of the caller's `GenerateParameters`. Fixed to pass `generationParameters.topK` and `generationParameters.minP` through.

**Impact for our app:** Our `MLXAudioSpeaker` passes `GenerateParameters` without explicitly setting topK/minP, so Qwen3TTS will now use the library's default sampling configuration for those fields. No behavior regression expected.

### MarvisTTS: Rope Caches Hidden from Strict Weight Verification — commit 4ab6539

`MarvisTTSModel` now marks rope cache buffers with `@Module(key: "none")` to prevent weight-count mismatch errors during model loading. Previously caused failures on some Marvis checkpoints.

### EnglishG2P: Escaped Underscore Member Reads — commit 65e228f

Fixes a Swift identifier parsing issue in `EnglishG2P.swift` where dictionary member lookups with underscore-prefixed keys caused compilation or runtime errors. Affects Kokoro multilingual TTS and KittenTTS (StyleTTS2).

---

## API / Infrastructure Changes

### resolveOrDownloadModel Now Accepts progressHandler — commit ddb803f

```swift
public static func resolveOrDownloadModel(
    ...,
    progressHandler: (@MainActor @Sendable (Progress) -> Void)? = nil
) async throws -> URL
```

When provided, the handler forwards to `HubClient.snapshot` for real download progress. Falls back to the existing print-based handler when nil.

**Opportunity for our app:** `DownloadManagerHF` currently uses its own download pipeline. This new hook could be wired to our `DownloadManagerHF` progress callback for models downloaded via the mlx-audio-swift path.

### Package Dependency Upgrades — commit ec54202

| Package | Previous | New |
|---------|---------|-----|
| mlx-swift | 0.30.x | 0.31.3 |
| mlx-swift-lm | < 3.31.3 | 3.31.3 |
| swift-transformers | 1.1.6 | 1.1.9 |

mlx-swift-lm 3.31.3 renamed/moved the `Tokenizer` type under `MLXLMCommon`, requiring all callers of `Tokenizers.Tokenizer` to use the fully-qualified name to disambiguate from `MLXLMCommon.Tokenizer`. Our fork's comments (`// Tokenizers.Tokenizer disambiguates from MLXLMCommon.Tokenizer`) document this invariant.

---

## Supported Model Matrix (New in this Upgrade)

| Model | Category | Notes |
|-------|----------|-------|
| DeepFilterNet V1/V2/V3 | STS | Noise suppression; Apple Silicon optimized |

---

## Risk Assessment

| Area | Risk | Reason |
|------|------|--------|
| Package upgrade (mlx-swift-lm 3.31.3) | **Medium** | Tokenizer type rename; handled in our fork, but any downstream type inference on `Tokenizer` without qualification may fail. Compile-time only. |
| DeepFilterNet addition | **Low** | New target only; existing code paths unchanged. No iOS entitlement or hardware requirement. |
| Parakeet batch API | **Low** | Additive only; existing `generateStream` unchanged. Our `MLXAudioASR` is not affected. |
| Qwen3TTS topK/minP fix | **Low-Medium** | Qwen3TTS now reads topK/minP from GenerateParameters instead of hardcoded values. If our code passes GenerateParameters with topK=0, Qwen3TTS sampling behavior may change slightly. Monitor audio quality on Qwen3-TTS models. |
| MarvisTTS rope fix | **Low** | Fixes a loading crash; no behavioral change on models that loaded previously. |
| Qwen3TTSReferenceConditioning | **Low** | Additive public API; existing generate paths unchanged. |
| resolveOrDownloadModel progressHandler | **Low** | Backward-compatible; nil default preserves existing behavior. |

### Recommended Post-Upgrade Validation

1. Load and run a Qwen3-TTS model — verify audio quality is unchanged.
2. Load a Parakeet model and run a transcription — check for regression.
3. Load a Marvis/CSM model if downloaded.
4. Confirm Kokoro multilingual TTS still produces correct phoneme outputs (EnglishG2P fix).
