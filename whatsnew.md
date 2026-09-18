# mlx-audio-swift: What's New (tag-20260509 → tag-20260918)

Merged upstream `Blaizzy/mlx-audio-swift` `main` into our fork on 2026-09-18
(merge commit `8f3bd0e`). **50 commits** in range — 48 upstream, 2 local.

| | tag-20260509 | tag-20260918 |
|---|---|---|
| Swift files in `Sources/` | 261 | 320 |
| Swift LOC in `Sources/` | 79,989 | 102,715 (+28%) |
| Diffstat | | 120 files, +23,779 / −668 |

One conflict, in `Sources/MLXAudioTTS/Models/MossTTS/MossTTSModel.swift` —
resolved in upstream's favour. See [Our fork's patches](#our-forks-patches).

---

## 1. New model families

### STT — 7 new families

| Model | Commit | Notes |
|---|---|---|
| **Whisper** (full family) | `0c71eba` (#192) | Every `openai/whisper-*` size and `.en` variant; both HF `transformers` and OpenAI/mlx-whisper checkpoint layouts. Quantized checkpoints fixed in `4b609fe` (#235). |
| **Nemotron ASR** | `2766d9b` (#195) | NVIDIA streaming checkpoints, e.g. `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit`. Cache-aware streaming (`417df21` #196), incremental `NemotronASRStreamSession` for live mic (`bd9669f` #208), English streaming checkpoints (`3a25c7e` #236). |
| **MOSS-Transcribe-Diarize** | `d17a0ed` (#221) | Audio-conditioned Qwen3 decoder + Whisper encoder — timestamped transcription **with speaker labels in one pass**. Opt-in quantized KV cache for memory-bounded long-form (`c5d4054` #225). |
| **Canary** | `580e952` (#215) | NVIDIA Canary. |
| **Moonshine** | `580e952` (#215) | Useful Sensors Moonshine — small, low-latency. |
| **Wav2Vec2 CTC / MMS** | `580e952` (#215) | Massively Multilingual Speech CTC heads. |
| **LASR CTC** | `580e952` (#215) | |

### TTS — 5 new families

| Model | Commit | Notes |
|---|---|---|
| **OmniVoice** | `3cfa972` (#209) | Multilingual zero-shot TTS: bidirectional diffusion LM over a Qwen3 backbone, 9 RVQ codebooks @ 24 kHz, HiggsAudioV2 codec. Voice cloning on both fp32 and bf16 weights (`0ea78a5` #213). Adds reproducible seed (`3777187` #233), task cancellation (`898aedf` #227), and per-step denoise progress (`4266f98` #219). |
| **Spark-TTS** | `3e97855` (#261) | Qwen2 LM emitting BiCodec semantic + global tokens → 16 kHz. Controllable gender/pitch/speed **and** voice cloning via the full BiCodec encode path. |
| **IndexTTS** | `a011c35` (#220) | With its own BigVGAN vocoder. |
| **Irodori-TTS** | `26bffa6` (#206) | Japanese flow-matching (Echo-TTS family), Rectified-Flow DiT @ 48 kHz. **VoiceDesign** — voice described by a Japanese caption instead of a reference clip. |
| **Breeze TTS 2** | `d20cbd6` (#255) | Bilingual EN/ZH, natural-language voice design + zero-shot cloning. BF16 / 8-bit / 4-bit checkpoints. |

### VAD / codecs

- **FSMN VAD** (`72f2f07` #214).
- **`SpeechSegmenter`** — reusable speech segmenter for VAD pre-processing (`c587280` #178), wired into Cohere Transcribe as a Silero VAD pre-processor (`2f3c3ed` #177).
- **Codec parity with Python mlx-audio** (`72f2f07` #214): HiggsAudio tokenizer, StepAudio2. `MossAudioTokenizer`, `S3Tokenizer`, and the `S3Gen` stack (CAMPPlus, ConformerEncoder, FlowMatching, HiFTGenerator, S3GenMel) moved out of `MLXAudioTTS/Models/Chatterbox/` into shared `MLXAudioCodecs/`.

---

## 2. Performance — what matters on device

Ordered by relevance to models **we currently ship**.

| Change | Commit | Impact |
|---|---|---|
| **Sortformer: single bulk GPU→CPU readback in `predsToSegments`** | `d2035cd` (#193) | **~1.8× faster streaming diarization.** Replaces a per-frame `.item()` round-trip loop with one `asArray` readback. We ship `diar_streaming_sortformer_4spk-v2.1-fp16`, so this lands directly on `MLXAudioDiarizer`. Output is provably identical (verified below). |
| **Qwen3 text attention via `attentionWithCacheUpdate`** | `542fffa` (#228) | Fewer graph nodes per decode step on the Qwen3 TTS backbone. We ship two Qwen3-TTS models. |
| Voxtral Realtime: incremental mel/conv front end | `3fa0303` (#230) | **O(N²) → O(N) per utterance.** Not shipped by us, but the pattern matters if we ever enable Voxtral streaming. |
| Voxtral Realtime: stop clearing the Metal buffer pool every step | `25ef620` (#229) | Removes a per-step allocator stall. |
| Voxtral Realtime: hoist per-layer-invariant attention inputs out of layer loops | `6ea59e5` (#231) | |
| Voxtral Realtime: fix float32 leak in streaming | `3b0b114` (#226) | Memory regression fix — relevant class of bug for any long-running on-device session. |
| MOSS-Transcribe-Diarize: opt-in quantized KV cache | `c5d4054` (#225) | Bounded memory on long-form audio. Exposed generically as `STTGenerateParameters(kvBits:kvGroupSize:quantizedKVStart:)`. |
| Fish Speech: stream progressively | `12b32ff` (#237) | Lower time-to-first-audio. |

---

## 3. Correctness fixes — iOS-affecting

Two of these fix bugs we were exposed to.

- **`b917ab5` (#256) — STT: scale fbank input to int16 unconditionally in FireRedASR2 / SenseVoice.**
  **This fixes a real iOS bug we could hit.** The old code auto-detected scale with
  `if amplitude <= 1.0 { waveform *= 32768 }`. Lossy decoders — AAC via AVFoundation —
  overshoot past 1.0 on clipped content, which flipped the branch, skipped the scaling
  the Kaldi fbank recipe and CMVN stats expect, and **silently collapsed decoding to
  empty segments on iOS**. Now unconditional, matching sherpa-onnx's hard-coded
  `normalize_samples = false`. We ship `SenseVoiceSmall`.

- **`bf14ae0` (#247) — Qwen3-ASR mel frontend: Slaney mel scale + periodic Hann window.**
  Now byte-matches transformers' `WhisperFeatureExtractor`. `hanningWindow` gained a
  `periodic:` parameter and `melSpectrogram` a `melScale:` parameter, both defaulted to
  the legacy values so every other front end is untouched. **Transcription output for
  `Qwen3-ASR-0.6B-4bit` will change slightly — in the accuracy-positive direction.**

- **`63f33f5` (#189) — MossTTS: gate `homeDirectoryForCurrentUser` behind `#if os(macOS)`.**
  Upstream independently landed the same fix we carried locally. Superseded three
  commits later; see below.

- **`3f6b055` (#186) — Qwen3-TTS CustomVoice voice parsing.** `voice` is now parsed as
  `"speaker, instruction"`. Our seven CustomVoice speaker names contain no commas, so
  behaviour is unchanged and the instruction half is a new capability.

- **`10b7366` (#204) — Kokoro: iterate `unicodeScalars` in `tokenize`.** Swift's
  `for ch in String` fuses combining marks into grapheme clusters, so French nasal
  vowels (base vowel + U+0303) were dropped entirely — "bonjour" came out "bjour".

- **`3032ccc` (#234) — Honor Chatterbox emotion override with default conditioning.**
  We ship Chatterbox Turbo but never set an override, so no change for us today.

- **`3506fb9` (#253)** FireRedASR2 beam-search topK gathering across rows;
  **`856e04a` (#188)** FireRedASR2 hides CMVN runtime arrays from strict-verify reflection;
  **`8ed8188` (#232)** VoxtralRealtime loads checkpoints with a quantized tied embedding;
  **`4b609fe` (#235)** quantized Whisper checkpoints load correctly.

---

## 4. Architectural changes

- **`416f08c` (#197/#198) — shared NeMo-family module.** `ParakeetAttention`,
  `ParakeetRNNTLayers`, `ParakeetDecodingLogic`, `ParakeetAlignment` moved to
  `Models/Nemo/` as `Nemo*`, decoupling NemotronASR from Parakeet. Upstream ships
  `ParakeetNemoAliases.swift`, so **every `Parakeet*` symbol still resolves**.

- **`StreamingInferenceSession` refactored into a facade** over a
  `StreamingInferenceSessionCore` protocol with three cores (Qwen, Cohere, MOSS).
  A new `init(model: any STTGenerationModel, config:)` picks the core automatically.
  **The Qwen decode path is a byte-identical extraction** — diffing the old class body
  against `QwenStreamingInferenceSessionCore` yields only the class rename and
  access-modifier changes.

- **New `STT.loadModel(modelRepo:cache:)` / `STT.loadModel(modelRepo:modelType:cache:)`**
  — a first-party STT dispatch table covering all 16 STT families, mirroring
  `TTS.loadModel`.

- **`AudioGeneration` gained `case progress(Double)`** — exact fractional progress from
  models with a deterministic step count (diffusion denoise steps).

- **`StreamingConfig.language` became `String?`** (`nil` = model default/auto). The only
  source-breaking public change in the range; our call site uses the default.

- **`MLXAudioSTT` now depends on `MLXAudioVAD`**, and `MLXAudioTTS` on `MLXFFT` from
  mlx-swift. Our `thirdparty/mlx-swift` fork already vends `MLXFFT`.

---

## 5. Our fork's patches

All **26** `wangqi modified` markers survive the merge. The five
`Tokenizers.Tokenizer` disambiguation patches remain attached to their declarations —
including the one in `StreamingInferenceSession.swift`, which moved from line 567 to
1464 during the refactor and still guards the right parameter.

**The one conflict**, in `MossTTSModel.swift`, was our `#if os(macOS)` gate around
`findCachedHubSnapshot`. Upstream landed the identical fix themselves in `63f33f5`
(#189), then **deleted the entire function** in `50860ee` (#207), replacing the
hand-rolled `~/.cache/huggingface/hub` scan with `HubCache` + `hfToken` threaded
through `fromPretrained` / `fromModelDirectory`.

Resolved in upstream's favour. Nothing is lost:

- `homeDirectoryForCurrentUser` now appears **nowhere** in the package, so the iOS
  compile error our patch prevented cannot recur.
- The local-directory probe we relied on moved *into*
  `MLXMossAudioTokenizer.fromPretrained`, which still checks a tilde-expanded
  `config.json` before any network call.

Two local commits are also in range: `5c645df` (the original macOS fix, now superseded)
and `56770f3` (adapt `CSMModel` to the now-throwing `makePromptCache`).

---

## 6. Risk assessment

### Low risk — verified

| Area | Finding |
|---|---|
| **Compilation** | `swift build` on the merged package: **passed**, exit 0. |
| **Public API** | Every removed `public` declaration is a *relocation*, not a deletion. All nine model types our app names still resolve; every `fromPretrained(_:cache:)` we call is intact. |
| **Exhaustive switches** | `AudioGeneration` gained a case, but our `toSamplesStream()` uses `if case .audio(…)` — a pattern match, not an exhaustive switch. `STTGeneration` and `TranscriptionEvent` are unchanged, so our exhaustive switches over those still compile. |
| **TTS model routing** | `inferModelType` gained breeze/spark/irodori/omnivoice/indextts checks, three of them *prepended*. **None of our 8 shipped TTS repo names contain those substrings**, and existing ordering is untouched. MOSS-TTS-Nano still resolves to `moss_tts_nano`. |
| **Qwen3 streaming ASR** | Byte-identical extraction. Zero behavioural change. |
| **Sortformer rewrite** | Semantically equivalent: the trailing `if segStart >= 0` correctly replaces the old zero-pad/diff trick, and the float32 widening from fp16 is exact. |
| **BigVGAN `beta` → optional** | `BigVGANPeriodicActivation` has no consumers anywhere in the tree. Chatterbox's Snake is HiFTGenerator's, not this one. |
| **Platform safety** | No macOS-only API anywhere in `Sources/`, and upstream introduced **no new platform conditionals**. |

### Medium risk — accept, but watch

1. **Binary size. The largest practical iOS risk.** +22,726 Swift LOC (+28%), 59 new
   files, 12 new model families — all statically linked into the app whether or not a
   user downloads those weights. Worth measuring the `.ipa` delta on the next archive
   before submission.

2. **Output drift on two shipped STT models.** Qwen3-ASR (#247) and SenseVoice (#256)
   will produce different text than before. Both changes are corrections toward the
   Python reference, but any transcription golden-file tests need re-baselining.

3. **MOSS-TTS-Nano audio-tokenizer cache location moved.** `fromPretrained` now threads
   our `HubCache` into the audio-tokenizer fetch instead of using `.default`. In
   practice `ensureAudioTokenizer` checks the bundled `audio_tokenizer/` subfolder
   first, which `mlx-community/MOSS-TTS-Nano-100M` ships — so the network path is
   almost certainly never reached. If it ever is, expect one re-download into the app's
   models directory (which is the better location anyway).

### Resolved since this document was first written

**The app now builds on both destinations.** The `Multiple commands produce
'…/include/module.modulemap'` collision between FluidAudio's `NemoTextProcessing.xcframework` and
SwiftGit2's `libgit2.xcframework` fired at build-*planning* time and was never caused by this
merge; it is fixed. So the caveat that used to sit here — that the app-level compile of
`libs/audio/mlxaudio/*.swift` had been verified by API inspection rather than by the compiler — no
longer applies. Both `AIAssistant` and `AIAssistantMac` compile against the merged package.

**And it is no longer verified by compilation alone.**
`helper/scripts/model_regression/run_audio_tests.py` now loads and exercises every catalogue model
through the app's own wrappers on macOS and records a baseline that the next upgrade can
`--compare` against (see `helper/docs/mlx-audio-swift.md` §4.13). That is what closes item 2 under
*Medium risk* above: the output drift on Qwen3-ASR (#247) and SenseVoice (#256) is now measured and
recorded per model rather than assumed.

Building the gate also turned up a defect this document did not predict. `MLXAudioASR.makeSTTModel`
dispatched on six repo-name substrings and **fell through to `Qwen3ASRModel` for everything else**,
so every one of the seven new STT families would have loaded the wrong architecture rather than
failing. It now routes through `STT.loadModel`, which throws on an unknown repo. Note that
`STT.loadModel` is not a drop-in: five of its branches construct through a `fromPretrained` that
ignores the `HubCache` argument, which would have re-downloaded models outside app storage — see
§2.3 of the developer guide.

---

## 7. Integration opportunities for `libs/audio/mlxaudio/`

No changes are *required*. Three are worth making — see
`helper/docs/` follow-ups and the notes in `MLXAudioASR.swift`.

1. **`cancelStreaming()` should call `session.cancel()`, not `session.stop()`.**
   `stop()` awaits the in-flight decode, flushes the mel processor, encodes remaining
   windows and runs a **final decode pass** to emit `.ended`. On a user-initiated
   cancel that is pure wasted GPU work on-device, and we discard the result anyway —
   `cancelStreaming()` already finishes its own continuation first. `cancel()` tears
   down `decodeTask`/`stopTask`, resets the encoder and mel processor, and returns.
   (`cancel()` predates this upgrade; the mismatch is pre-existing.)

2. **Replace the private `makeSTTModel` dispatch with `STT.loadModel(modelRepo:cache:)`.**
   Our chain is 6 hand-written `lower.contains(…)` branches whose `else` falls through
   to Qwen3 — so an unrecognised repo silently loads as the wrong architecture instead
   of erroring. `STT.inferModelType` correctly classifies all five STT repos we ship
   and covers 16 families, throwing `STTModelError.unsupportedModelType` otherwise. It
   also removes our two special cases (Cohere's `fromDirectory` workaround, and the
   comment explaining why VoxtralRealtime is excluded), since upstream handles both.
   **Caveat:** the 2-argument form calls `ModelUtils.resolveModelType`, which may hit
   the network for a model type it cannot resolve locally. Prefer the 3-argument
   `loadModel(modelRepo:modelType:cache:)` form to stay strictly offline.

3. **`StreamingInferenceSession(model: any STTGenerationModel, config:)`** would let
   `startStreaming` drop its `item.repo.lowercased().contains("qwen3")` check and
   `as? Qwen3ASRModel` downcast. Note it is `preconditionFailure` — not `throw` — on an
   unsupported model, so the app must keep gating on `item.supportsStreaming`
   before constructing one.

Also available, not currently needed: `STTGenerateParameters(kvBits:)` for
memory-bounded long-form transcription, `SpeechSegmenter` for VAD pre-processing,
`KokoroModel.generateWithDurations` for per-phoneme timing (we use the Core AI /
sherpa Kokoro paths instead), and `AudioPlayer.unloadAudio()`.
