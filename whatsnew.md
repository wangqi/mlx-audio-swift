# mlx-audio-swift: What's New (tag-20260321 → tag-20260328)

## Upgrade Summary

Changes merged from upstream `Blaizzy:main` into our fork between `tag-20260321` and `tag-20260328`.

---

## New Features

### TTS: Kokoro with Multilingual Support (#124)
- New `KokoroModel` added — a StyleTTS2-based TTS model with multilingual phonemization.
- Ships with `KokoroMultilingualProcessor` using a ByT5-based G2P pipeline (100+ languages via `beshkenadze/g2p-multilingual-byT5-tiny-mlx`).
- English G2P uses the Misaki port with a BART fallback network.
- Voice style control via `voiceAliases` and `speedPriors` in config.
- 24 kHz output, multi-speaker support.

### TTS: KittenTTS (#123)
- StyleTTS2-based TTS model sharing the new `StyleTTS2/` shared blocks (BiLSTM, AdaIN, SourceModule, etc.).
- Uses `MisakiTextProcessor` for English phonemization.
- Factory integration via `TTSModel` with `kitten_tts` case.

### New: MLXAudioG2P Module (#121, #122)
- New `MLXAudioG2P` Swift package target for grapheme-to-phoneme conversion.
- Includes dictionary-based and neural (ByT5 encoder-decoder) backends.
- `NeuralPhonemizer` protocol conformance; word-level G2P for use with StyleTTS2 models.
- `MLXAudioNeuralG2P` supports 100+ languages.

### StyleTTS2 Shared Architecture (#122)
- Shared blocks: `BiLSTM`, `WeightNormedConv`, `AdaIN`, `SourceModule`, `ISTFTNetConfig`, `PLBertConfig`.
- `Albert.swift`: ALBERT encoder (6 Module-based classes) shared between KittenTTS and Kokoro.
- Used by both KittenTTS and Kokoro to reduce code duplication.

---

## Bug Fixes

### Fix: Parakeet Multilingual Recognition (#108)
- Parakeet mel filterbank was using a normalization mode that confused Russian phonemes with Polish/Latin transliteration.
- Fixed mel filterbank computation for NeMo-trained models (HTK normalization applied correctly).
- Special token filtering added to `ParakeetTokenizer.decode()` — `isSpecialToken()` now strips control tokens before output.
- Result: Russian (and other non-English) transcription now matches NeMo CUDA reference output.
- Tests added for special token filtering, eval mode, and mel invariants.

### Fix: Qwen3 ASR Language Autodetection (#110)
- Qwen3 ASR now supports `language: nil` in `STTGenerateParameters` to trigger in-model language autodetection.
- `normalizeLanguageName()` maps short codes and aliases (e.g. `"ru"` to `"Russian"`, `"zh"` to `"Chinese"`) to canonical names that the model understands.
- `mergeLanguages()` consolidates detected languages across chunks for accurate per-result language reporting.
- `parseGeneratedChunk()` extracts embedded language tokens from model output correctly.
- GLM ASR also updated to use `language: nil` as its default.
- CLI flag `--language` now documents: omit to allow autodetect.

---

## Performance Improvements

### MLXFast Dependency for MLXAudioCodecs (#128)
- `MLXAudioCodecs` now depends on `MLXFast` from `mlx-swift`.
- Enables hardware-accelerated codec operations (convolution, attention) via MLX's fast kernel path.
- Transparent: no API changes required; codec models (BigVGAN, DAC, FishS1DAC) benefit automatically.

---

## Risk Assessment for iOS Upgrade

| Area | Risk | Notes |
|------|------|-------|
| Parakeet fix | **Low** | Bug fix only; no API changes. Non-English ASR accuracy improves. |
| Qwen3 autodetect | **Low** | `language: nil` was already a valid parameter; model behavior change is intentional and beneficial. |
| Kokoro TTS | **Low** | New model type; `TTS.loadModel()` dispatches correctly. No changes needed in `MLXAudioSpeaker`. |
| KittenTTS | **Low** | New model type; same load path as other TTS models. |
| MLXAudioG2P | **Medium** | New Swift package target. Will increase binary size. Requires `MLXAudioG2P` dependency in Package.swift targets that use StyleTTS2 models. |
| MLXFast codec dependency | **Low** | Additive dependency; no API breakage. May slightly increase build time on first resolve. |
| StyleTTS2 shared blocks | **Low** | Internal to the package; no impact on our app code. |

### Required App-Side Changes

1. **`MLXAudioASR`**: Change the language fallback from `"English"` to `nil` to enable Qwen3/GLM language autodetection when no language is configured.
2. **`LocalModelAboutView`**: Update engine version, date, and what's new to reflect this upgrade.
3. No changes needed in `MLXAudioSpeaker` — `TTS.loadModel()` handles Kokoro/KittenTTS dispatch transparently.

### iOS Device Considerations
- Kokoro and KittenTTS include ByT5 G2P models that require a HuggingFace download on first use — ensure download flows handle this gracefully.
- MLXFast kernels are Metal-backed; all Apple Silicon iOS devices (A14+) are fully supported.
- Parakeet fix has no iOS-specific concerns.
