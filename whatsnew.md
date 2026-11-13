# mlx-audio-swift: What's New (tag-20261030 → tag-20261113)

Merged upstream `Blaizzy/mlx-audio-swift` `main` into our fork on 2026-11-13 (merge `36c146f`, of `cf095e0`).
**43 upstream commits**. Three files conflicted, all resolved (§3). Two fork markers were dropped as moot.

| | tag-20261030 | tag-20261113 |
|---|---|---|
| Head | `d084891` | `36c146f` (merge of `cf095e0`) |
| Diffstat, `Sources/` | | 122 files, +6332 / −958 |
| Diffstat, `Tests/` | | 31 files, +2445 / −45 |
| Dependency pins | mlx-swift ≥ 0.32.3 | **mlx-swift ≥ 0.33.0**, + `MLXFFT` product on MLXAudioSTT |

Integration doc: `helper/docs/mlx-audio-swift.md`. Consumers: `libs/audio/mlxaudio/`, `libs/audio/voiceclone/mlx/`.
Shipped model list: `helper/mlxaudio_model_test.json` (28 rows).

This range is mostly fixes to existing models. There are only two new model families: Maya-2 and ZONOS3-Mini.
**#360 (`MaterializedArray`) is a source-breaking API change.** Every `refAudio:` argument, every
`generate()` result and every `.audio` event is now an evaluated, immutable `MaterializedArray`, not a
`sending MLXArray`. The app needed a 3-line fix (§2). **Eighteen commits change the output of a model we
ship.** The biggest of these are Qwen3-TTS (#369 decoder sliding window, #371 ICL priming, #378 trim, #352
speaker mel), the Mimi transformer window (#357), which changes PocketTTS cloning and every Mimi-based decode,
and the shared mel frontend (#363) used by Qwen3-ASR, GLM-ASR and Voxtral Realtime.

---

## 1. Changes

Commits that change the output of a model we ship are marked **[SHIPPED OUTPUT]**. The P10 regression has to
watch those.

| Commit / PR | What changed | Path affected | Reaches the app? |
|---|---|---|---|
| `87d0359` (#360), `b13c8ab` (#359) | **Adopt `MaterializedArray` (mlx-swift 0.33.0).** `SpeechGenerationModel.generate/generateStream(refAudio:)` take `MaterializedArray?`. `generate` returns `MaterializedArray`. `AudioGeneration` is `Sendable` and `.audio` carries `MaterializedArray` (with a `.audio(MLXArray)` factory). `loadAudioArray` returns `(Int, MaterializedArray)`. `DiarizationOutput`, `DeepFilterNetStreamingChunk`, `Qwen3TTSReferenceConditioning` and Chatterbox `DefaultConditioning` are now `Sendable` snapshots. **`MLXAudioCore/SendingBox.swift` deleted.** No numeric change. | `MLXAudioCore/*`, `MLXAudioTTS/Generation.swift`, 55 model files, `Sortformer`, `VADOutput` | **Yes, compile break**: fixed in the app (§2). |
| `7def7fa` (#369) | Qwen3-TTS speech decoder now applies its 72-frame sliding window (`RotatingKVCache` + windowed mask). Before, it attended to the full history. Output changes past about 6 s of audio per call. **[SHIPPED OUTPUT]** | `Qwen3TTS/Qwen3TTSSpeechTokenizer.swift` | Yes: all 4 Qwen3-TTS rows, every voice |
| `1e9c8ee` (#371) | Qwen3-TTS ICL streaming: the decoder is primed with the last 25 reference codes, so streamed audio starts with the reference as left context (same as non-streaming). Applies to the `conditioning:` path too. **[SHIPPED OUTPUT]** (cloned voices) | `Qwen3TTS/Qwen3TTS.swift`, `Qwen3TTSSpeechTokenizer.swift` | Yes: Qwen3 Base clone builds (`VoiceCloneMLXArtifact.qwen3`), Voice Design anchor |
| `c8f59b8` (#378) | Qwen3-TTS keeps frames whose first codebook id is 0 when trimming decoded audio. Before, they were cut, which shortened the audio. **[SHIPPED OUTPUT]** | `Qwen3TTS.swift`, `Qwen3TTSSpeechTokenizer.swift` | Yes: all Qwen3-TTS rows |
| `ff1427a` (#352) | Qwen3-TTS: the speaker-encoder mel now matches the reference (magnitude, Slaney scale, natural log, reflect pad). Also, `refAudio` without `refText` now injects the x-vector instead of being dropped. **[SHIPPED OUTPUT]** for new clone builds: `prepareReferenceConditioning` computes a different `speaker_embedding`. Saved `conditioning.safetensors` keep the old embedding. | `Qwen3TTS.swift`, `Qwen3TTSSpeakerEncoder.swift` | Yes: Voice Clone Qwen3 builds and Voice Design anchors made after the upgrade. The x-vector-only path is not used (the app always passes a transcript). |
| `200ced1` (#357) | **Mimi transformer: attention is limited to its context window inside a multi-frame call.** Single-step calls keep `context` keys, not `context + 1`. Upstream measured PocketTTS voice-embedding error vs Python going from 31–52 % to about 2e-6 for 5–13 s clips. The dropped first word on cloning is fixed. **[SHIPPED OUTPUT]** | `MLXAudioCodecs/Mimi/Transformer.swift` | Yes: Pocket TTS (clone encode on every chunk, plus decode), Phonon 31m-en (Pocket decode), Marvis (Mimi codec), Qwen3-TTS encoder for references over 10 s |
| `df2a254` (#368), `b79cfe4` (#367) | Mimi `resetState` also resets the down/upsample convs, so the previous clip's tail no longer leaks into the next. `encodeStep` returns no codes when no frame completes. Marvis throws `invalidRefAudio` for a reference shorter than one Mimi frame. **[SHIPPED OUTPUT]** (Marvis, first frames after a reset) | `Mimi/Mimi.swift`, `Marvis/MarvisTTSModel.swift` | Yes: Marvis default and cloned |
| `4c363d4` (#363) | `computeMelSpectrogram` drops the last centred STFT frame, like `WhisperFeatureExtractor`. This changes the frame count and the max used by the −8 clamp. **[SHIPPED OUTPUT]** | `MLXAudioCore/DSP.swift` | Yes: Qwen3-ASR, GLM-ASR, Voxtral Realtime (also Qwen3 forced aligner). MOSS-Transcribe-Diarize does not use it. |
| `cc3b2c0` (#355) | `loadAudioArray` averages the channels to mono instead of keeping channel 0 | `MLXAudioCore/AudioUtils.swift` | `MLXAudioSpeaker.loadReferenceAudio`. App references are mono `reference.wav`, so no change is expected. |
| `afabf01` (#385) | Chatterbox Turbo keeps T3 in float16 (faster). `prepareInputEmbeddings` is split out. **[SHIPPED OUTPUT]** (numerics) | `Chatterbox/T3/T3GPT2Model.swift` | Yes: `mlx-audio-chatterbox-turbo-4bit` (`dtype: float16`) |
| `c521abf` (#380), `e845040` (#356) | VoxCPM2 / shared DiT runs bf16 checkpoints in bf16 (masks, RoPE, time embeddings cast). Multi-character Chinese tokens are fed one character at a time. **[SHIPPED OUTPUT]** | `VoxCPM2/MiniCPM.swift`, `VoxCPM2Model.swift`, `VoxCPMDiT.swift` | Yes: `voxcpm2-4bit`, `voxcpm15-4bit` (bf16 non-quantized layers); Chinese text |
| `1265304` (#384), `011287d` (#351) | Cohere Transcribe runs fp16 checkpoints in fp16. Streamed deltas now continue the text already sent, compared per Unicode scalar, so partial text no longer repeats on multi-byte or combining characters. **[SHIPPED OUTPUT]** (numerics; streamed partials) | `CohereTranscribe/*`, `STT/Shared/Tokenizer.swift` | Yes: `cohere-transcribe-4bit`. `.result` text unchanged except for numerics. |
| `3b7a255` (#383) | Voxtral Realtime casts float32 weights to the checkpoint's half dtype. **[SHIPPED OUTPUT]** (numerics) | `VoxtralRealtime/VoxtralRealtime.swift` | Yes: `voxtral-realtime-4b-4bit` |
| `2879cd5` (#377), `1edbcc1` (#358) | Soprano normalizer: decimals read as "N point M", ordinals read as ordinals, and the `\d[KMBT]` rule is word-bounded. **[SHIPPED OUTPUT]** for text with numbers | `Soprano/TextUtils.swift` | Yes: `soprano-1.1-80m-5bit` |
| `5fd737c` (#365) | GLM-ASR transcribes audio over 30 s in chunks. Before, longer audio was truncated. **[SHIPPED OUTPUT]** (long audio) | `GLMASR/GLMASR.swift` | Yes: `glm-asr-nano-4bit` file transcription |
| `3736445` (#374) | **Nemotron Unified** (`chunked_limited_with_rc`, buffered streaming, `NemotronASRBufferedEncoder`). `per_feature` normalisation uses valid frames only. A bare language code matches exactly one locale key (`de` → `de-DE`). `NemotronASRStreamSession.init` gains a defaulted `bufferedContext:`. | `NemotronASR/*` | Our row (`normalize: NA`, `chunked_limited`) takes the unchanged cache-aware path. Only the language resolution is new, and `MLXAudioASR` already passes dictionary codes. |
| `e0de9e7` (#396) | Parakeet: a dense HF `ParakeetForTDT` export is detected by `encoder_config` + `durations` (was `ternary_modules`). Ternary fields are optional. | `Parakeet/ParakeetModel.swift`, `ParakeetRedux.swift` | Routing checked on the mirror configs: `parakeet-redux` still has both keys and stays Redux, while `parakeet-tdt-0.6b-v3` and `Phonon-2` have neither and stay NeMo. No change. |
| `1c0520e` (#372) | **Maya-2 Native TTS** (Indic + English, Mimi codec + ECAPA speaker encoder from two other repos). `Qwen3Model.callAsFunction(inputEmbeddings:)` is split out (same graph). `EcapaSpeakerEncoder` is public. New resource `maya2_normalization_data.json`. | new `Maya2/*`, `Qwen3/Qwen3.swift`, `Codecs/EcapaTdnn/*`, `TTSModel.swift` | Qwen3.swift: Vyvo (`Qwen3Model`), identical output expected. Maya-2 is not catalogued (§6). |
| `cf095e0` (#399) | **ZONOS3-Mini TTS** (LM + codec + speaker encoder, single repo) | new `Zonos3/*`, `TTSModel.swift` | No (§6) |
| `8f371bd` (#366) | `MLXFFT` added to MLXAudioSTT's dependencies | `Package.swift` | Build only. Our fork `thirdparty/mlx-swift` exports `MLXFFT`. |
| `0dda61b`, `abd0135`, `fb829f5` (#364, #362, #361) | Canary and FireRedASR2 decode long audio in chunks. Canary accepts language names. | `Canary/*`, `FireRedASR2/*` | No (not catalogued) |
| `163fd1d`, `a699303`, `9d66d6f`, `6a898d6` (#379, #382, #381, #370) | Whisper fp16, MiMo-ASR embedding dtype, Voxtral TTS flow-head dtype, Voxtral TTS 25-frame streaming context | model folders | No |
| `c758bf2`, `e1b473e` (#354, #353) | Kokoro / StyleTTS2 English G2P: accents folded, numbers read like misaki | `StyleTTS2/G2P/*` | No (the app's Kokoro is sherpa-onnx / Core AI, not MLXAudio) |
| `9e26c1f`, `3518dd9`, `0e03fee`, `50f2edf`, `8e7bde5`, `4649f92`, `4b9ca01`, `5c4add0`, `927d4e3` | Confucius4 segments + `</s>`, IndexTTS / Zonos2 / MeloTTS / LongCat normalizers, Audio8 Realistic checkpoints, repo-move links | model folders | No |

---

## 2. Public API

| Symbol | Change | Our usage / fix |
|---|---|---|
| `SpeechGenerationModel.generate / generateStream / generateSamplesStream(refAudio:)` | `sending MLXArray?` → `MaterializedArray?` | **Broke** `MLXAudioSpeaker.chunkStream` `.reference` case (P5: `MLXAudioSpeaker.swift:574/578` "cannot convert value of type 'MLXArray' to expected argument type 'MaterializedArray'"). **Fix:** `MLXAudioSpeaker.loadReferenceAudio` returns `MaterializedArray?` (what `loadAudioArray` now returns), and `VoiceCloneMLXArtifact.reference(audio:)` holds a `MaterializedArray`. |
| `Qwen3TTSModel.prepareReferenceConditioning(refAudio:refText:language:)` | `refAudio: MaterializedArray` | **Broke** `VoiceCloneQwen3Builder.encode`. **Fix:** `MLXArray(resampled).materialized()`. Also fixed in the Mac prototype `testcases/audio/prototype/qwen3_voice_consistency/.../Probe.swift`, which is not an app target. |
| `generate(...)` return, `AudioGeneration.audio` | `MaterializedArray` (an `MLXArray` subclass) | `.asArray(Float.self)` unchanged. No fix needed. |
| `loadAudioArray(from:sampleRate:)` | Returns `(Int, MaterializedArray)`; averages channels | Via `loadReferenceAudio` (above) |
| `Qwen3TTSReferenceConditioning` | `Sendable`; stored fields are `MaterializedArray`; `init` still takes `MLXArray` | `readQwen3Conditioning` / `saveQwen3Conditioning` unchanged |
| `DiarizationOutput` (`speakerProbs: MaterializedArray?`), `DeepFilterNetStreamingChunk` | `Sendable` snapshots | `MLXAudioDiarizer` reads `segments` / `numSpeakers`; `MLXAudioSTS` calls `enhance(MLXArray)`. Unchanged. |
| `SendingBox` (package-internal) | Deleted | No app reference (grep of `libs/`, `apps/`, `testcases/`, `views/`, `ai/`: none) |
| `NemotronASRStreamSession.init(..., bufferedContext:)` | New defaulted parameter | `LiveSessionDecoder(nemotron:)`: source-compatible |
| `EcapaSpeakerEncoder`, `Qwen3Model.callAsFunction(inputEmbeddings:cache:)`, `Qwen3TTSModel.decodeChunk` (now internal, not private) | New / additive | Unused |

No STT, VAD or STS signature the app calls changed. `STTGenerationModel.generateStream(audio: MLXArray, ...)`
is unchanged. No app file imports both an `MLXAudio*` module and `MLXLMCommon` / `MLXLLM` / `MLXVLM`. The only
new top-level public names are `EcapaSpeakerEncoder` and the reshaped `AudioGeneration`, so there is no new
shared-name ambiguity. A side benefit: the app reuses one reference tensor across every chunk of an
utterance. Under the old `sending` contract that was technically a reuse-after-transfer. It is now sanctioned.

---

## 3. Conflicts and fork patches

| File | Resolution (P3) |
|---|---|
| `Audio8/Audio8Falcon.swift` | **Take upstream**, marker dropped. Upstream now writes `MLX.compile(shapeless:)` itself (#359), so the file is byte-identical to upstream. |
| `Marvis/MarvisTTSModel.swift` | **Take upstream** for `generateStream` (`let task = Task {…}`, `onTermination = { task.cancel() }`). Our 2026-03-21 `generationTask` + `SendingBox` patch is moot: upstream added the same cancel-on-termination and made `refAudio` a Sendable `MaterializedArray`. Kept: FR/DE voices, folder-first Mimi codec, cosF32 filter + `.noUnusedKeys`. |
| `MossTTSNano/MossTTSNanoModel.swift` | **Keep both.** Upstream still returns stereo and added `.materialized()`. Our stereo-to-mono mix is kept and `.materialized()` is applied to its result (marker 2026-05-04). |

Patch inventory: `d084891` 8 files / 12 markers → working tree **7 files / 10 markers**.
- Dropped (2): Audio8Falcon ×1, Marvis `generationTask` ×1.
- Kept, file untouched upstream: `ModelUtils` `?? true` (no manifest = complete), Phonon2Checkpoint ×2,
  Phonon2Model ×1 (fermion-only config is not a container), MossTTSNano `LocalizedError` ×1.
- Kept, file changed upstream: `TTSModel.swift` `gradium_phonon_tts` case (upstream still has only
  `"pocket_tts", "phonon"`), Marvis ×3, MossTTSNano stereo→mono ×1.

**Offline invariant:** the only new hub fetches are in Maya-2 (`Mimi.fromPretrained(config.codecRepo)` and
`EcapaSpeakerEncoder.fromPretrained(speakerEncoderRepo)`, two extra repos) and ZONOS3 (`resolveOrDownloadModel`,
single repo, cache-first). Neither ships. **Adding Maya-2 would need folder-first patches for both, like
Marvis.** No fetch was added to a shipped model.

`helper/docs/mlx-audio-swift.md` §6: remove the Audio8Falcon and Marvis `generationTask` / `SendingBox` rows,
and note that MossTTSNano now calls `.materialized()` after the mono mix.

---

## 4. Dependencies

- **mlx-swift floor 0.32.3 → 0.33.0** (`MaterializedArray`). Our fork `thirdparty/mlx-swift` is at 0.33.0 and
  exports `MLXFFT` (`Package.swift:369`). No dependency move was needed (P4).
- MLXAudioTTS resources gain `Maya2/maya2_normalization_data.json`. It goes into the existing
  `MLXAudio_MLXAudioTTS.bundle`.
- Floors unchanged: swift-transformers, swift-huggingface, zstd.

---

## 5. Risk assessment: 7 risks (0 high, 4 medium, 3 low)

### R1: Qwen3-TTS output changes (MEDIUM)
**Problem:** #369 (72-frame window) changes every Qwen3 voice past about 6 s per call. #378 keeps frames that
were trimmed. #371 changes the start of each streamed clone chunk. #357 changes the encoder for references over 10 s.
All are intended fixes toward the Python reference. **Verify:** P10
`./run_audio_tests.py --compare /Volumes/ssd2t/modeltests/reports/audio-20261113-163701.json`: the 4
`mlx-audio-qwen3-tts-*` rows pass default and cloned passes, with ASR similarity ≥ P1. Then a voice sweep of the
4 Qwen3 rows (§6.2). **Status:** P10.

### R2: Existing Qwen3 clone builds vs new builds (MEDIUM)
**Problem:** #352 changed the speaker-encoder mel. A `conditioning.safetensors` built before the upgrade keeps
its old x-vector and still loads (format unchanged), but a rebuild of the same clip now gives a different,
reference-faithful embedding. Voice Design anchors work the same way. There is no format break and no
migration. **Verify:** `VoiceCloneMLXArtifactTests` (round trip). P10 cloned pass for
`mlx-audio-qwen3-tts-0.6b-4bit` / `1.7b-base-4bit`. On device: one existing Qwen3 build still speaks after the
update. **Status:** P10 + device.

### R3: Mimi-based TTS output changes (MEDIUM)
**Problem:** #357 / #368 / #367 change Pocket TTS cloning (it re-encodes the clip through Mimi on every chunk,
and the first word is no longer dropped), the Pocket / Phonon decode (window `context`, not `context+1`), and
Marvis (reset leak; a reference under one frame now throws `invalidRefAudio`, which the app surfaces as an
error). **Verify:** P10 `mlx-audio-pocket-tts` (default + cloned), `mlx-audio-phonon-31m-en`,
`mlx-audio-marvis-tts-250m-4bit` (default + cloned) pass with similarity ≥ P1, plus a voice sweep of these 3
rows. **Status:** P10.

### R4: Shared mel frontend (MEDIUM)
**Problem:** #363 drops one STFT frame and so shifts the −8 dB clamp's max for Qwen3-ASR, GLM-ASR and Voxtral
Realtime. It matches Python and should not lower accuracy. **Verify:** P10 `mlx-audio-qwen3-asr-0.6b-4bit`,
`mlx-audio-glm-asr-nano-4bit`, `mlx-audio-voxtral-realtime-4b-4bit` similarity ≥ P1 (0.909 / 1.0 / 1.0).
**Status:** P10.

### R5: Half-precision numerics (LOW)
**Problem:** Chatterbox Turbo (#385), VoxCPM2 / 1.5 (#380), Cohere (#384) and Voxtral Realtime (#383) now run in
the checkpoint's half dtype. They are faster but not bit-equal to P1. **Verify:** P10 the 5 rows pass, with
cloned passes for Chatterbox and VoxCPM. **Status:** P10.

### R6: `MaterializedArray` evaluates eagerly (LOW)
**Problem:** every `.audio` chunk and every `refAudio` is now `eval`ed and snapshotted when it is created.
Streaming latency per chunk could rise slightly, and memory holds one more copy of each reference.
**Verify:** P10 timings for the TTS rows stay within the usual spread of P1. P5 macOS build and the iOS build
compile the §2 fix. **Status:** P5 rerun + P10.

### R7: Soprano / GLM-ASR / Cohere behaviour fixes (LOW)
**Problem:** Soprano reads decimals and ordinals differently (#377, #358). GLM-ASR no longer truncates audio over
30 s (#365). Cohere partials no longer repeat (#351). All are improvements, but they are user-visible.
**Verify:** P10 `mlx-audio-soprano-1.1-80m-5bit`, `mlx-audio-glm-asr-nano-4bit`, `mlx-audio-cohere-transcribe-4bit`
pass. `MLXAudioASRStreamingContractTests`. **Status:** P10.

Carried over, still open from 2026-10-30: STT streams run detached with no in-flight guard in
`MLXAudioASR.transcribe` (§7).

Shipped classes with **no** output change in this range: Parakeet TDT v3 / Redux / Phonon-2 (routing
re-checked), Nemotron 3.5 streaming (cache-aware path unchanged), SenseVoice, Granite, MOSS-Transcribe-Diarize,
Sortformer and DeepFilterNet (`MaterializedArray` only), Silero / Smart-Turn VAD, MOSS-TTS-Nano
(`.materialized()` only), and Vyvo (`Qwen3Model` refactor, same graph).

---

## 6. New models evaluated for the catalogue (2026-11-13)

Sizes are HF tree sums (HF API, 2026-11-13). Limits: TTS / STS / VAD / diarization ≤ 1000 MB, STT ≤ 700 MB.

| Model | Repo size | Outcome | Reason |
|---|---|---|---|
| **Nemotron Unified 0.6B 4-bit** (`mlx-community/nemotron-labs-unified-asr-0.6b-4bit`) | **473 MB** (`model.safetensors` 472, `config.json`, `tokenizer.model`, `vocab.txt`) | **Add** | STT, 25 European languages, `model_type: nemotron_asr` (existing dispatch, no new app code). Single repo, no second fetch. Licence OpenMDW-1.1 (permissive, redistribution allowed with the licence file, `LICENSE.md` is in the repo). Not a duplicate: a different base from the shipped Nemotron 3.5 streaming. It streams **buffered** (`chunked_limited_with_rc`): `MLXAudioASR` live mode reaches it through `LiveSessionDecoder` → `NemotronASRStreamSession`, which handles buffered checkpoints. P8 must prove both batch and live. 8-bit (744 MB) is over the limit. |
| Maya-2 Native (`maya-research/maya2-native`; community `aruntemme/maya2-native-mlx-8bit`) | 5056 MB bf16 / 3326 MB 8-bit; plus `kyutai/mimi` 385 MB and `speechbrain/spkrec-ecapa-voxceleb` 89 MB | **Reject** | Over 1000 MB, with no 4-bit MLX checkpoint. It would also need two folder-first offline patches (Mimi codec, ECAPA encoder). Apache-2.0. |
| ZONOS3-Mini (`mlx-community/ZONOS3-Mini`, `-swift-ptq`) | 1409 MB / 1276 MB | **Reject** | Over 1000 MB. Apache-2.0. Original `Zyphra/ZONOS3-Mini` is 30.7 GB. |
| Parakeet Ultra (`moondream/parakeet-ultra`; MLX fp16 `beshkenadze/parakeet-ultra-mlx-fp16`, `selcukkubur/parakeet-ultra-mlx`) | 1257 / 1255 MB | **Reject** | Over 700 MB (STT), with no quantized MLX checkpoint. CC-BY-4.0. Loadable after #396. |
| Orukeet (`oruk/orukeet`; `itsreidar/orukeet-mlx`) | 11837 / 1255 MB | **Reject** | Over 700 MB. CC-BY-SA-4.0. |
| Nemotron Unified 8-bit / bf16 | 744 / 1254 MB | **Reject** | Over 700 MB (the 4-bit is added) |

Families fixed in this range but already rejected on size on 2026-10-30, still over the limit: Confucius4,
Audio8 (incl. nativ-community Realistic), Voxtral TTS, IndexTTS, LongCat, MeloTTS (frontend mirroring),
Canary / FireRedASR2 / Whisper / MiMo-ASR (no rows).

### 6.1 Voices of Add candidates
Nemotron Unified is STT, so it has no voices.

### 6.2 Voice coverage (§4a): shipped TTS rows the range touched
`git diff --stat d084891..tag-20261113 -- Sources/MLXAudioTTS Sources/MLXAudioCodecs`: **no voice file, voice
enum, speaker table or prompt folder changed** (the Marvis `Voice` enum, Qwen3 `spk_id`, Pocket / Phonon voice
packs and Soprano are all untouched). Model code changed for these rows. Sweep them, because every voice's
audio moves:

| Row | Why |
|---|---|
| `mlx-audio-qwen3-tts-custom-0.6b`, `-0.6b-4bit`, `-1.7b-base-4bit`, `-1.7b-voicedesign-4bit` | #369 / #378 decoder, #371 / #352 clone |
| `mlx-audio-pocket-tts`, `mlx-audio-phonon-31m-en` | #357 Mimi transformer window |
| `mlx-audio-marvis-tts-250m-4bit` | #357 / #368 / #367 Mimi |
| `mlx-audio-chatterbox-turbo-4bit` | #385 fp16 T3 (no voice list; default + clone pass) |
| `mlx-audio-voxcpm2-4bit`, `mlx-audio-voxcpm15-4bit` | #380 bf16 (no voice list; default + clone pass) |
| `mlx-audio-soprano-1.1-80m-5bit` | #377 / #358 normalizer only (single voice) |

Not touched beyond `MaterializedArray`: `mlx-audio-vyvo-tts-en-4bit` (Qwen3.swift refactor, same graph),
`mlx-audio-moss-tts-nano-100m`.

---

## 7. Opportunities (not built)

| Upstream feature | Consumer that would use it | Value |
|---|---|---|
| Nemotron Unified buffered streaming (#374): `NemotronASRBufferedContext.forLatency` / `bufferedContext:` | `MLXAudioASR` live mode | A latency setting for the new row. Today it uses the model default (320 ms layout). |
| Nemotron bare-code language match (#374) | `MLXAudioASR` Nemotron language resolver | The app's own name → code mapping could rely on it for `de` → `de-DE`. Keep ours: it also maps English names. |
| `MaterializedArray` `Sendable` results (#360) | `MLXAudioSpeaker` | The reference tensor and Qwen3 conditioning can be cached across utterances and actors without copies. |
| Canary language names (#361), long-audio chunking for Canary / FireRed (#364, #362) | ASR catalogue | Only if a Canary / FireRed row under 700 MB appears |
| STT in-flight guard (open since 2026-10-30) | `MLXAudioASR.transcribe` | One generation per model; await the detached stream on cancel before `unloadModel` clears the cache |

---

## 8. Verification

- P1 baseline: `/Volumes/ssd2t/modeltests/reports/audio-20261113-163701.json`, 28 pass / 0 fail / 0 skip.
- P3: conflicts resolved; standalone `swift build` exit 0 (remote mlx-swift).
- P5: the first macOS app build **failed** on the §2 `MaterializedArray` break (`MLXAudioSpeaker.swift:574/578`).
  P7 fixed it in `MLXAudioSpeaker.swift` and `VoiceCloneMLXBuilders.swift`. **Pending: rerun the macOS build.**
- Pending: P10 `--compare` against the P1 report, voice sweep of the §6.2 rows, P8 Nemotron Unified 4-bit
  (batch + live), `MLXAudioCatalogueSchemaTests`, `MLXAudioVoiceCoverageTests`, `VoiceCloneEngineCatalogTests`,
  `MLXAudioReferenceAudioTests`, `VoiceCloneMLXArtifactTests`, `VoiceDesignAnchorWiringTests`,
  `MLXAudioASRStreamingContractTests`, and the iOS build.
- macOS cannot verify: R2 on device (an existing Qwen3 build after the update), and iOS memory for the new row.
