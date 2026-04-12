# mlx-audio-swift What's New

## tag-20260412 (2026-04-12)

### Changes from tag-20260406

Six commits landed in this release, covering one new STT model family, two important bug fixes, a speaker-embedding feature for Qwen3-TTS CustomVoice, and convenience methods for loading models from local directories.

---

#### New Features

**Cohere Transcribe STT (commit d5394bd)**

A new `CohereTranscribeModel` is added under `Sources/MLXAudioSTT/Models/CohereTranscribe/`. It implements `STTGenerationModel` and follows the same `generateStream(audio:generationParameters:)` interface as Parakeet, Granite Speech, and others.

- Supports quantized checkpoints
- `fromDirectory(_ modelDir: URL)` for loading from a pre-downloaded local path
- Note: `fromPretrained` does not accept `HubCache`; use `fromDirectory()` when using an app-managed cache

**Qwen3-TTS CustomVoice Speaker Embedding (commits 56d0811, 9264f40)**

For models with `ttsModelType == "custom_voice"` (e.g. `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit`):

- Flexible decoding of `spk_id` (single `Int` vs. `[Int]`) and `spk_is_dialect` (`Bool` vs. dialect-name `String`) in `config.json`
- Speaker embedding injection: looks up the speaker name in `talkerConfig.spkId`, calls `talker.getInputEmbeddings()`, and injects the embedding between codec prefix and pad+bos tokens in `prepareGenerationInputs`
- Dialect language-ID override applied when `spk_is_dialect` flag is set

Effect: Voice identity is now preserved across sentences for CustomVoice models. Previously, each sentence defaulted to the model's base voice regardless of the `voice` parameter.

**`fromModelDirectory` Convenience Methods (commit da93511)**

Every major model class now has `fromModelDirectory(_ modelDir: URL)` alongside `fromPretrained`. Affected classes include `Qwen3ASRModel`, `GraniteSpeechModel`, `GLMASR`, `ChatterboxModel`, `EchoTTSModel`, `FishSpeechModel`, `LlamaTTS`, `PocketTTSModel`, `Qwen3TTS`, `Soprano`, `KittenTTSModel`, `KokoroModel`, `SmartTurn`, codecs (DACVAE, DescriptDAC, Encodec, SNAC), and audio enhancement models.

---

#### Bug Fixes

**Kokoro 8-bit Crash and NaN Duration Fix (commit 5196558)**

Three distinct bugs fixed for quantized Kokoro checkpoints:

1. Weight transposition crash: `sanitize()` was unconditionally transposing 3D weights; MLX-converted quantized weights are already in the correct layout. Fixed with an `isQuantized` guard.
2. NaN duration propagation from quantized encoder: added explicit NaN guard; NaN values are replaced with a minimum duration.
3. OOM from garbage duration values: raw `int32` cast of a NaN float produced huge index values. Fixed with a cap at 100 frames and silence return on empty indices.

Effect: Quantized Kokoro models (8-bit) now load and run correctly on devices with limited RAM.

**ParakeetQuantizationConfig Decoding Fix (commit ef4e10d)**

Enhanced config decoder to handle unexpected `model_type` values in quantized Parakeet checkpoints, preventing a decode failure when loading certain quantized Parakeet variants.

---

#### Upgrade Risk Assessment

| Area | Risk | Notes |
|------|------|-------|
| Kokoro TTS (quantized) | Low | Bug fix only; no API change |
| Qwen3-TTS CustomVoice | Low | Automatic via library; `generateSamplesStream()` unchanged |
| Parakeet quantized | Low | Config parsing fix; transparent |
| Cohere Transcribe STT | Low | New model type; requires explicit wiring in makeSTTModel() |
| fromModelDirectory API | None | Additive; no existing call sites affected |

Overall upgrade risk: Low. No breaking API changes. The Kokoro fix and Qwen3-TTS CustomVoice speaker embedding are the most impactful improvements for end users.

---

## tag-20260403 (2026-04-03)

### Changes from tag-20260328
