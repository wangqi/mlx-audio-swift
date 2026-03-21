# mlx-audio-swift: What's New (tag-20260315 to tag-20260321)

## Commits Included

| Commit | Description |
|--------|-------------|
| a153236 | Add Chatterbox Turbo (#102) — ResembleAI voice cloning TTS port |
| 40918f5 | Update README.md (#104) — documentation update |
| 6b8fcc0 | Add Fish Audio S2 Pro model (#106) — FishSpeech S2 Pro TTS |
| 1755fc1 | Add README for Qwen3 TTS (#107) — documentation only |
| d54d2ee | Merge branch 'Blaizzy:main' into main |

---

## New Features

### 1. Chatterbox TTS / Chatterbox Turbo (PR #102)

**What it is**: A port of ResembleAI's Chatterbox TTS model to MLX Swift. Two variants:
- **Chatterbox Turbo** (`mlx-community/chatterbox-turbo-fp16`) — GPT-2 backbone, faster inference
- **Chatterbox TTS** (`mlx-community/Chatterbox-TTS-fp16`) — LLaMA backbone, higher quality

**Key capabilities**:
- Text-to-speech with 24kHz output
- Voice cloning via reference audio conditioning (`refAudio` / `refText` params)
- Streaming generation support (API-level; see iOS notes below)
- Default conditioning from bundled speaker embeddings (no reference audio required)
- Emotion tag support: `[laugh]`, `[sigh]`, etc. via S3Gen pipeline

**Architecture (multi-stage pipeline)**:
- **T3 (Text-to-Semantic)**: GPT-2 or LLaMA backbone generating semantic tokens
  - `T3GPT2Model`: 12-layer GPT-2 with learned position embeddings
  - `T3CondEnc`: Conditioning encoder fusing speaker embeddings + reference semantics
  - `Perceiver`: Cross-attention resampler for audio conditioning
- **S3Gen (Semantic-to-Speech)**:
  - `S3TokenizerV2`: Semantic speech tokenizer
  - `CAMPPlus`: Speaker encoder (Conformer backbone) for voice cloning
  - `ConformerEncoder`: 12-layer conformer for acoustic modeling
  - `FlowMatching`: Diffusion-based flow matching vocoder
  - `HiFTGenerator`: HiFT-based neural vocoder producing 24kHz audio

**Model type identifiers**: `"chatterbox"`, `"chatterbox_tts"`, `"chatterbox_turbo"`

**API conformance**: Implements `SpeechGenerationModel` — compatible with the existing `generateSamplesStream()` call in `MLXAudioSpeaker`. Voice cloning requires passing non-nil `refAudio`/`refText` (not currently exposed in app UI).

**Streaming note**: `generateStream()` internally calls `generate()` which produces the full audio before yielding. There is no true token-by-token streaming for Chatterbox — the 2-second `streamingInterval` parameter has no effect. Audio arrives as a single chunk after full generation.

---

### 2. Fish Speech S2 Pro (PR #106)

**What it is**: Fish Audio's S2 Pro model as a new `FishSpeechModel` class. Extends the existing `FishS1DAC` codec (added in tag-20260315) into a full TTS pipeline.

**Key capabilities**:
- Text-to-speech with voice cloning via `refAudio`/`refText`
- VQGAN-based codec tokenizer for semantic code generation
- Zero-shot voice cloning via `FishSpeechPrompt`

**Architecture**:
- `FishSpeechModel`: Main model conforming to `SpeechGenerationModel`
- `FishSpeechTokenizer`: VQGAN codec tokenizer for encoding/decoding
- `FishSpeechPrompt`: Prompt builder for voice conditioning
- Reuses `FishS1DAC` codec (already present in tag-20260315) for final audio decoding

**Model type identifiers**: `"fish_speech"`, `"fish_qwen3_omni"`

**API note**: The `voice` and `language` parameters are currently ignored in FishSpeech. Voice cloning uses `refAudio`/`refText` directly.

**AudioUtils extension**: Minor addition to `AudioUtils.swift` to support FishSpeech audio processing.

---

## iOS-Specific Considerations

### Memory Pressure

Both new models are large and memory-intensive:

| Model | Architecture | iOS Risk |
|-------|-------------|----------|
| Chatterbox Turbo FP16 | GPT-2 + multi-stage vocoder | High — 5-stage pipeline holds multiple large tensors |
| Chatterbox TTS FP16 | LLaMA + multi-stage vocoder | Very High — LLaMA backbone uses significantly more RAM |
| Fish Speech S2 Pro | VQGAN tokenizer + FishS1DAC codec | Medium-High — similar to FishSpeech S1 complexity |

The existing `MLXAudioSpeaker` memory pressure monitor (`.warning` and `.critical` thresholds) correctly releases `ttsModel` and clears the MLX cache. No code change needed.

### First-Audio Latency

Chatterbox does NOT true-stream. Full audio generation completes before any audio is yielded. On iPhone-class hardware, first-audio latency for Chatterbox may be 10-30+ seconds depending on text length. Users should be warned via the model description in the catalog.

### Voice Cloning API

Both models support voice cloning via `refAudio: MLXArray?` and `refText: String?`. The current `MLXAudioSpeaker` always passes `nil`, using default/bundled speaker conditioning. Voice cloning is a future UI enhancement opportunity.

### Model Recommendation

For iOS catalog, prefer exposing only **Chatterbox Turbo** (GPT-2 backbone). The LLaMA variant is likely to OOM on devices with less than 8GB RAM.

---

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|-----------|
| Memory OOM on iPhone with Chatterbox LLaMA variant | High | Expose only Chatterbox Turbo in catalog; memory pressure monitor handles cleanup |
| High first-audio latency for Chatterbox | Medium | Document in model description; no streaming means full wait before audio starts |
| API compatibility | None | Both models conform to `SpeechGenerationModel`; no code changes required |
| MLXAudioASR unaffected | None | No new STT models in this upgrade |
| FishS1DAC codec re-use | None | FishSpeech S2 Pro reuses codec from tag-20260315; well-tested path |

---

## App Code Impact

- **`MLXAudioSpeaker.swift`**: No changes needed. Both models are automatically dispatched by `TTS.loadModel()`.
- **`MLXAudioASR.swift`**: No changes needed. No new STT models in this upgrade.
- **`LocalModelAboutView.swift`**: Updated to reflect new models and version tag-20260321.

---

Upgrade date: 2026-03-21
