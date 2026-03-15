# mlx-audio-swift Upgrade Notes: tag-20260309 to tag-20260315

## Commits Included

| Hash | Title |
|------|-------|
| cae5593 | Fix window chunking for Qwen3 ASR (#92) |
| fcbd04d | Add Granite Speech 4 (#95) |
| 035f0e0 | Fill in some missing codecs (#96) |
| 8fb5fdf | Add Echo TTS (#97) |
| 00432632 | Add FireRed ASR 2 (#98) |
| db2635b | Add SenseVoice (#99) |

---

## New Features

### New STT Models

**SenseVoice** (db2635b)
- Multi-language speech recognition from Alibaba/FunASR
- Supports emotion recognition tags (`HAPPY`, `SAD`, etc.) and acoustic event detection
- 50+ language support including Chinese, English, Japanese, Korean, Cantonese
- Uses a custom SenseVoice tokenizer backed by the refactored `UnigramTokenizer` in `MLXAudioCore`
- model_type: `"sensevoice"`

**FireRed ASR 2** (00432632)
- Hybrid attention-based ASR from FireRed AI, optimized for Mandarin and English
- Transformer encoder with CTC decoder; custom tokenizer with SentencePiece
- model_type: `"fireredasr2"`

**IBM Granite Speech 4** (fcbd04d)
- IBM Research speech model combining a WhisperEncoder audio backbone with a Granite language model
- End-to-end encoder-decoder architecture with cross-attention projection
- model_type: `"granite_speech"` / `"granite"`

### New TTS Models

**Echo TTS** (8fb5fdf)
- DiT (Diffusion Transformer) based TTS model
- Uses FishS1DAC as the audio codec (newly added this cycle)
- Flow-matching generation with configurable diffusion steps
- Already registered in `TTS.loadModel()` factory under types `"echo_tts"` / `"echo"`
- model_type: `"echo_tts"`

### New Audio Codecs (035f0e0)

**BigVGAN** - NVIDIA neural vocoder; mel-spectrogram to waveform via multi-period discriminator

**Descript DAC** - High-fidelity general-purpose audio tokenizer using Residual Vector Quantization (RVQ)

**FishS1DAC** - FishAudio S1 dual-codebook codec used by Echo TTS; dual-stream Mamba + Transformer decoder

---

## Bug Fixes

**Qwen3 ASR window chunking** (cae5593)
- Fixed a variable scope bug: loop windows were appended with undefined index `i` instead of `windowIndex`
- For long audio with multiple transcript windows, this caused incorrect window ordering after sort, producing garbled output
- Added unit tests covering multi-window scenarios

---

## Refactoring

**UnigramTokenizer moved to MLXAudioCore** (db2635b)
- `UnigramTokenizer` and its protobuf parser (`SentencePieceModelParser`, `SentencePieceProtobufReader`) moved from `MLXAudioTTS/Models/PocketTTS/` to `MLXAudioCore`
- `PocketTTSConditioners` now calls `UnigramTokenizer(sentencePieceModelData:)` instead of the previously local custom parser
- New convenience initializers: `init(sentencePieceModelData:)`, `init(tokenizerJSONData:)`, plus static factories `from(tokenizerJSONURL:)` and `from(sentencePieceModelURL:)`
- **Conflict resolved in our fork**: The custom `parseSentencePieceProto` method in `SentencePieceTokenizer` was removed in favour of the upstream `UnigramTokenizer(sentencePieceModelData:)` which delegates to the rewritten `SentencePieceModelParser`

---

## iOS Compatibility Assessment

| Feature | iOS Risk | Notes |
|---------|----------|-------|
| Qwen3 ASR window fix | Low | Pure logic fix; no API changes |
| SenseVoice STT | Low | Pure Swift + MLX; no platform APIs |
| FireRed ASR 2 | Low | Pure Swift + MLX; SentencePiece binary tokenizer |
| Granite Speech 4 | Low | Pure Swift + MLX; dual-encoder cross-attention |
| Echo TTS | Low-Medium | FishS1DAC codec is new and memory-intensive |
| BigVGAN / Descript DAC | Low | Pure Swift + MLX; not used by any default model yet |
| FishS1DAC codec | Low-Medium | Mamba layers; may be demanding on older devices |
| UnigramTokenizer refactor | Low | Conflict resolved correctly; upstream parser is equivalent |

**Overall risk: Low**
- All changes are pure Swift/MLX with no platform-specific APIs
- No changes to audio I/O, AVFoundation, or CoreML paths
- Echo TTS / FishS1DAC is the most memory-intensive addition; recommend testing on iPhone 15 or later
- `MLXAudioASR` requires a code update to dispatch to the new STT model classes

---

## Required App-Side Changes

1. **MLXAudioASR.swift** (Done): Added multi-model dispatch factory. Without it, only Qwen3 ASR would load; SenseVoice, FireRed ASR 2, and Granite Speech would silently fall back to Qwen3.
2. **LocalModelAboutView.swift** (Done): Updated `mlxAudioSwiftInfo` version, date, supported model list, and whatsNew entries.
3. **MLXAudioSpeaker.swift**: No changes needed. Already uses `TTS.loadModel()` factory, which now includes Echo TTS automatically.
