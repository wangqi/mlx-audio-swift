# mlx-audio-swift: What's New (tag-20261018 → tag-20261024)

Merged upstream `Blaizzy/mlx-audio-swift` `main` into our fork on 2026-10-24 (merge `4ef9820`).
**32 upstream commits**, 3 files with conflicts (all resolved, see §3), one new fork patch.

| | tag-20261018 | tag-20261024 |
|---|---|---|
| Head | `df803b0` | `4ef9820` (merge of `6fd9386`) |
| Diffstat, `Sources/` | | 219 files, +25002 / −1119 |
| Diffstat, `Tests/` | | 33 files, +5991 / −16 |
| Dependency pins | mlx-swift ≥ 0.30.6, mlx-swift-lm ≥ 3.32.3 | mlx-swift ≥ **0.32.3**, **mlx-swift-lm removed** (#287) |

Integration doc: `helper/docs/mlx-audio-swift.md`. Consumers: `libs/audio/mlxaudio/`, `libs/audio/voiceclone/mlx/`.

The most important item for the app is a **silent offline ship-blocker** found in P3: #291 + #288 together
made every catalogued MLX TTS load list the HF repo over the network, because folders filled by
`DownloadManagerHF` carry no `.mlx-audio-snapshot.json`. It is patched in `MLXAudioCore/ModelUtils.swift`
(§3). The second structural change is #287: MLXAudio no longer depends on MLXLMCommon; it vendors its own
`GenerateParameters`, samplers and KV caches into `MLXAudioCore`, which made `GenerateParameters`
ambiguous in `MLXAudioSpeaker.swift` (fixed by dropping its `import MLXLMCommon`).

---

## 1. Changes

Output-changing commits for models we ship are marked **[SHIPPED OUTPUT]** — those are what the P10
regression has to watch.

| Commit / PR | What changed | Path affected | Reaches the app? |
|---|---|---|---|
| `92ca66c` (#287) | **MLXLMCommon / MLXLLM dependency removed.** `GenerateParameters`, `LogitSampler`s, `KVCache*` (incl. TurboQuant, variance-normalized), `LanguageModel`, `StringOrNumber`, RoPE helpers vendored into `MLXAudioCore/LM/`, `MLXAudioCore/Generation/`. `Backbones/Qwen2` moved to Core later (#306). Every model now imports `MLXAudioCore` instead. | `Package.swift`, `MLXAudioCore/LM/**`, 106 files | Yes: every model. Compile fix in `MLXAudioSpeaker.swift`. Sampling semantics match our mlx-swift-lm fork (same samplers, own `RandomState`). |
| `219eeb2` (#288) | Gradium **Phonon TTS** in the PocketTTS family; PocketTTS reads `tokenizer.model` or `tokenizer.json` and throws if the vocabulary size differs from `n_bins`; quantization via `perLayerQuantization`; `hf://` voice URLs; `resolveModelType` always passes `["*.model"]`. **[SHIPPED OUTPUT]** PocketTTS model/FlowLM/MLP/Mimi adapter rewritten. | `PocketTTS/*`, `TTSModel.swift`, `ModelUtils.swift` | Yes: `mlx-audio-pocket-tts` |
| `a5506c1` (#289) | Translation example | `Examples/` | No |
| `a6c7289` (#272), `d36caf9` (#271), `ba6e46c` (#273) | Voxtral Realtime streaming: KV rows written into preallocated storage, Hann window reused, stream deltas built from token bytes (multi-byte characters no longer split). **[SHIPPED OUTPUT]** for live partial text. | `VoxtralRealtime/*` | Yes: `mlx-audio-voxtral-realtime-4b-4bit` live |
| `624fd54` (#274), `b7dd2e0` (#275) | Nemotron streaming computes only new mel frames, keeps MLX's buffer cache, and reuses unchanged encoder/decoder work. **[SHIPPED OUTPUT]** (incremental mel instead of full recompute). | `NemotronASR/*` | Yes: `mlx-audio-nemotron-asr-streaming-0.6b-8bit` live |
| `e2ecced` (#290) | **Qwen3-TTS speaking rate fix**: the non-cloned path (preset speakers, CustomVoice, VoiceDesign) now prefills all text paired with `codec_pad` instead of feeding it one token per step. **[SHIPPED OUTPUT]** | `Qwen3TTS.swift` (`prepareGenerationInputs`) | Yes: all 4 Qwen3-TTS rows (not the cloned ICL path) |
| `00abb7f` (#284) | Irodori-TTS v4.1 / v4.1-MF | `IrodoriTTS/*` | No (not catalogued) |
| `0ef2fed` (#291) | Cache folder without `.mlx-audio-snapshot.json` counts as incomplete when extra patterns are passed. | `ModelUtils.swift` | Yes: **offline regression, patched (§3)** |
| `865963e` (#292) | `compile` call fixes for mlx-swift main | `ParakeetModel.swift`, `Qwen3TTSCodePredictor.swift`, `Qwen3TTSTalker.swift` | Compile only |
| `5c4c12b` (#293) | STT loaders forward the `loadModel` cache (Canary, Cohere, LasrCTC, Moonshine, Voxtral Realtime, Wav2Vec2) | `MLXAudioSTT.swift` + 6 models | No (app loads Cohere/Voxtral via `fromDirectory`) |
| `19f2296` (#294) | **Qwen3-ASR `maxTokens` caps each chunk**, not the whole file — long audio no longer loses its later chunks. **[SHIPPED OUTPUT]** for audio long enough to exhaust 8192 tokens. | `Qwen3ASR.swift` | Yes: `mlx-audio-qwen3-asr-0.6b-4bit` |
| `8a34adc` (#299), `14b634e` (#297), `2e32b49` (#298), `262a391` (#300), `0f7a622` (#301), `013853a` (#302) | New STT: Mega-ASR, Phonon 2, Granite TurboCTC, Fun-ASR-Nano, Higgs Audio v3, Parakeet Redux. #297/#302 touch shared Parakeet/Nemo code (`Linear` → `UnaryLayer`, masked padding, valid-frame normalization) behind config flags that default off. | `MLXAudioSTT/Models/*`, `Parakeet/*`, `Nemo/NemoAttention.swift` | Parakeet TDT path only through defaults (expected unchanged) |
| `1e35e17` (#295) | Whisper `task: "translate"` in `STTGenerateParameters` (default `"transcribe"`) | `Generation.swift`, `WhisperModel.swift` | No |
| `654d4d8` (#296) | Reuse a cached snapshot that has no `config.json` | `ModelUtils.swift` | Loader only |
| `9162db0` (#303) | Granite Speech 4.1 / 5.0 / Plus / NAR. Existing Granite rewritten around `GraniteSpeechTask`, `GraniteSpeechRotaryEmbedding` (YaRN only when configured), audio features cast to the embedding dtype. Transcription prompt text unchanged for 4.0. **[SHIPPED OUTPUT]** possible (dtype casts, prompt builder). | `GraniteSpeech/*`, new `GraniteSpeechNAR/*` | Yes: `mlx-audio-granite-speech-1b-4bit` |
| `346be68` (#304), `902920c` (#305) | Nemotron-VoiceChat and MiMo codecs; `Mimi.fromModelDirectory` (offline), Mimi now set to `train(false)` after load. | `MLXAudioCodecs/*` | Yes: Marvis loads Mimi through it |
| `0cd69c0` (#307) | Kokoro / Kitten text processors use the `loadModel` cache | `StyleTTS2/*`, `TTSModel.swift` | No (not catalogued) |
| `1e1dc5f` (#306) | Qwen2-Audio STT; Qwen2 backbone moved to `MLXAudioCore/LM/Models/` | `Qwen2Audio/*`, `MLXAudioCore/LM/Models/*` | No |
| `4ce4146` (#308) | Offline Voxtral Mini; Voxtral Realtime tokenizer replaced by shared `TekkenTokenizer`; GLM-ASR attention goes through `attentionWithCacheUpdate`. **[SHIPPED OUTPUT]** possible (GLM-ASR attention path, Voxtral tokenizer). | `Voxtral/*`, `Shared/TekkenTokenizer.swift`, `GLMASR.swift` | Yes: GLM-ASR, Voxtral Realtime |
| `949be25` (#309), `d8e2caf` (#311), `e89b6c9` (#312), `6fd9386` (#313) | VibeVoice ASR (adds `sampleRate` to `STTGenerationModel`, default 16000), MiMo ASR, Ming 3.0 ASR Tiny, MOSS-Music | new `Models/*`, `Shared/Tokenizer.swift` | No |

---

## 2. Public API

| Symbol | Change | Our usage / fix |
|---|---|---|
| `GenerateParameters` | Now `MLXAudioCore.GenerateParameters` (vendored); two inits, the second requires `prefillStepSize:` | `MLXAudioSpeaker` builds it with `maxTokens/temperature/topP/repetitionPenalty/repetitionContextSize`. Was ambiguous with `MLXLMCommon.GenerateParameters`: **fixed by removing `import MLXLMCommon` from `MLXAudioSpeaker.swift`**. No app file imports both `MLXAudio*` and `MLXLMCommon` any more. |
| `KVCache`, `StringOrNumber`, `LogitSampler`, `LanguageModel` | Vendored into `MLXAudioCore` | Not referenced by app audio code |
| `Mimi.fromWeightsFile` (fork-only) | Gone; upstream `Mimi.fromModelDirectory(_:filename:)` | Marvis uses the upstream one; app never called it |
| `STTGenerateParameters(task:)` | New, default `"transcribe"` | Not passed |
| `STTGenerationModel.sampleRate` | New requirement with default 16000 | None |
| `Qwen3ASRModel.generate/generateStream(maxTokens:)` | Per-chunk cap | App leaves the 8192 default |
| `GraniteSpeechModel.generate(...)` | New `task/sourceLanguage/hotwords/punctuate/systemPrompt` params (defaulted) | Unchanged calls; `precondition` only if both `language` and `sourceLanguage` are set (app sets neither `sourceLanguage`) |
| `PocketTTSConfig.quantization`, `PocketTTSWeightQuantizationConfig` | Removed (fork-only) | Unused |

Swift 6 `sending`: no new annotations in this range; the `refAudio` reuse note from tag-20261018 still applies.

---

## 3. Conflicts and fork patches

| File | Resolution |
|---|---|
| `Codecs/Mimi/Mimi.swift` | Took upstream wholesale. `Mimi.fromModelDirectory` absorbs our 2026-09-29 `fromWeightsFile` split. `MarvisTTSModel.swift` now calls `Mimi.fromModelDirectory(codec/)`; marker reads `2026-09-29, 2026-10-24`. The mirror folder holds `codec/tokenizer-e351c8d8-checkpoint125.safetensors`, which is the preferred filename. 2 markers dropped. |
| `PocketTTS/PocketTTSConditioners.swift` | Took upstream; it reads `tokenizer.model` then `tokenizer.json` (absorbs our unmarked 2026-02-27 SentencePiece fallback). `hf://` tokenizer paths from `config.json` are filtered out, so no fetch. Our mirror has only `tokenizer.json` (Unigram, 4000 entries = `n_bins` 4000). |
| `PocketTTS/PocketTTSConfig.swift` | Took upstream; our unused `quantization` field dropped. |
| `Marvis/CSMModel.swift` | 2026-04-03 `import Tokenizers` removal moot (MLXLMCommon gone); upstream import restored. 1 marker dropped. |
| `Spark/SparkModel.swift` | 2026-09-18 `try tokenizer.encode` moot; upstream line. 1 marker dropped. |
| **`MLXAudioCore/ModelUtils.swift` (NEW patch, 2026-10-24)** | **Offline invariant.** With no `.mlx-audio-snapshot.json`, `isComplete` falls back to `true` (config.json + required weights suffice, as before #291). The manifest path is untouched. Without it every two-argument `TTS.loadModel(modelRepo:cache:)` lists the repo on the network and fails offline. |

Patch inventory: `df803b0` 19 files / 24 markers → working tree 17 files / 21 markers (−2 Mimi, −1 CSM,
−1 Spark, +1 ModelUtils). All other markers survived; 13 "Tokenizers.Tokenizer disambiguates from
MLXLMCommon" comments are now out of date but harmless. Kept: Marvis cosF32 filter + `.noUnusedKeys`,
named `generationTask`, Moss stereo-to-mono, `MossTTSNanoError: LocalizedError`.
`helper/docs/mlx-audio-swift.md` §6 is stale (rows 1–3) and needs a rewrite.

---

## 4. Dependencies

mlx-swift floor 0.30.6 → 0.32.3: satisfied by `thirdparty/mlx-swift` (contains tag 0.32.3).
mlx-swift-lm dependency removed. swift-transformers 1.1.6 and swift-huggingface 0.8.1 unchanged.
No dependency move.

---

## 5. Risk assessment — 7 risks (1 high, 3 medium, 3 low)

### R1 — Offline loads of every TTS row (HIGH)
**Problem:** #291 + #288 (above). **Verify:** P10 `./run_audio_tests.py --compare <P1 report>` (read-only, no network) — any TTS row FAILing at load means the patch is not in effect. **Status:** patched; P10.

### R2 — Qwen3-TTS non-cloned output changes (MEDIUM)
**Problem:** #290 prefills all text; preset/CustomVoice/VoiceDesign audio will differ (intended: correct speaking rate). Cloned (ICL) path untouched. **Verify:** P10 rows `mlx-audio-qwen3-tts-*` pass; duration of default pass vs P1 (should be similar or shorter, not truncated). **Status:** P10.

### R3 — PocketTTS rewrite (MEDIUM)
**Problem:** #288 rewrote model, FlowLM, Mimi adapter and tokenizer loading; new `vocabularySize == n_bins` throw. **Verify:** P10 `mlx-audio-pocket-tts` default + cloned pass. **Status:** P10.

### R4 — Streaming STT numeric drift (MEDIUM)
**Problem:** Nemotron incremental mel (#274/#275), Voxtral Realtime byte-level deltas and KV storage (#271–#273), Granite rewrite (#303), GLM-ASR attention path (#308). **Verify:** P10 batch + live similarity for Nemotron, Voxtral Realtime, Granite, GLM-ASR equal to P1. **Status:** P10.

### R5 — Vendored samplers / KV cache (LOW)
**Problem:** #287 copies mlx-swift-lm code into MLXAudioCore; fixes in our `thirdparty/mlx-swift-lm` fork no longer reach audio models, and the TurboQuant Metal kernels carry the same names as MLXLMCommon's (`turbo_fused_encode_*`). Only active with a `kvScheme`, which the app never sets. **Verify:** P10 sampling TTS rows (Qwen3-TTS, Chatterbox, Marvis, Vyvo, VoxCPM2, Soprano, MOSS Nano) pass. **Status:** P10.

### R6 — Qwen3-ASR long audio runs longer (LOW)
**Problem:** #294 removes the whole-file token budget; a runaway chunk can now spend 8192 tokens per chunk. The app already sets repetition penalty 1.15. **Verify:** P10 Qwen3-ASR row. **Status:** P10.

### R7 — Parakeet shared-code edits (LOW)
**Problem:** #297/#302 changed `ParakeetConformer`/`ParakeetAudio`/`NemoAttention` types and branches; all new behaviour is behind `phonon2`, `mask_padding`, `normalize_valid_frames` (default off). **Verify:** P10 `mlx-audio-parakeet-tdt-0.6b-v3` and `mlx-audio-sortformer-4spk`. **Status:** P10.

---

## 6. New models evaluated for the catalogue (2026-10-24)

Sizes are HF tree sums (2026-10-24). Limits: TTS ≤ 1000 MB, STT ≤ 700 MB.

| Model | Size | Decision | Reason |
|---|---|---|---|
| Parakeet Redux (`moondream/parakeet-redux`) | 179 MB | **Candidate** | STT, 25 European languages, CC-BY-4.0; existing `parakeet` dispatch (no new wrapper); ternary weights in fp container |
| Phonon 2 (`mlx-community/Phonon-2`) | 350 MB | **Candidate** | STT, English, CC-BY-4.0; `phonon2` model_type dispatched by `STT.loadModel` through the Parakeet loader; no app change |
| Granite Speech 5.0 TurboCTC 4-bit (`iky1e/...-turboctc-mlx-q4`) | 268 MB | Candidate (verify) | STT, English, Apache-2.0; third-party MLX conversion — loader compatibility unproven; IBM's own repo is 949 MB (over limit) |
| Gradium Phonon TTS 31m/40m (`gradium/phonon-*`) | 223–258 MB | Rejected | CC-BY-NC-4.0 and gated |
| Irodori-TTS v4.1 Small 8-bit / MF | 1389 / 1467 MB | Rejected | Over 1000 MB |
| Fun-ASR-Nano 4-bit | 1336 MB | Rejected | Over 700 MB |

Over the limit, not evaluated further: Mega-ASR (4178 MB), Higgs Audio v3 STT (5367 MB), Granite Speech
4.1 2B / 5.0 2B (≈ 4.9–5.1 GB), Granite NAR 5-bit (2377 MB), Qwen2-Audio 7B 4-bit (6575 MB), Voxtral Mini 3B
4-bit (3211 MB), VibeVoice ASR 4-bit (5714 MB), MiMo V2.5 ASR (4528 MB + 2576 MB tokenizer), Ming 3.0 ASR
Tiny 4-bit (5751 MB), MOSS-Music 8B 4-bit (6034 MB).

---

## 7. Opportunities (not built)

| Upstream feature | Consumer that would use it | Value |
|---|---|---|
| #293 cache forwarding for Canary/Lasr/Moonshine/Cohere/Voxtral Realtime | `MLXAudioASR.cacheIgnoringFamilies` | The refusal list can shrink; note `"voxtral"` substring would also route the new offline Voxtral Mini to `VoxtralRealtimeModel` |
| Whisper translate task (#295) | ASR settings | Translate-to-English option if a Whisper row is added |
| Granite `hotwords` / `punctuate` (#303) | `MLXAudioASR` for the Granite row | Keyword biasing and punctuated output |

---

## 8. Verification

P3: `swift build` (macOS) `Build complete!`. P5 macOS app build: Swift compile passes after the import fix;
link failed on app-side `LazyMarkdownV3` symbols unrelated to mlx-audio (rebuild pending).
P10 regression and `MLXAudioCatalogueSchemaTests`: pending.
