# mlx-audio-swift: What's New (tag-20261024 → tag-20261030)

Merged upstream `Blaizzy/mlx-audio-swift` `main` into our fork on 2026-10-30 (merge `50ba736`).
**31 upstream commits**, no conflicts. 15 fork markers were dropped as moot, one compile fix was folded
into the merge, and one fork patch was added on `tag-20261030` (Phonon `model_type`, §3).

| | tag-20261024 | tag-20261030 |
|---|---|---|
| Head | `4ef9820` (`116f2eb` with notes) | `50ba736` (merge of `d4fa7be`) |
| Diffstat, `Sources/` | | 173 files, +24051 / −903 |
| Diffstat, `Tests/` | | 38 files, +5412 / −3 |
| Dependency pins | mlx-swift ≥ 0.32.3 | mlx-swift ≥ 0.32.3, **+ facebook/zstd ≥ 1.5.7**, + `MLXFast` product |

Integration doc: `helper/docs/mlx-audio-swift.md`. Consumers: `libs/audio/mlxaudio/`, `libs/audio/voiceclone/mlx/`.
Shipped model list: `helper/mlxaudio_model_test.json`.

This range is mostly new model families: 20 new TTS types and 1 new STT type, all in new folders.
Four commits change the output of models we already ship. **#350 and #315 change Qwen3-TTS audio.** #350
fixes a click at every chunk join on the streaming path, which is the app's speak path. #315 makes the
cloned-voice (ICL) prefill run in the talker's dtype. #346 and #349 change how Parakeet-family and most
other STT `generateStream` calls deliver tokens. None of these is a ship-blocker. The only blocker found
was outside the code. On 2026-10-21 the HF repo `gradium/phonon-31m-en` renamed its `model_type`, and the
fork now covers that (§3).

---

## 1. Changes

Commits that change the output of a model we ship are marked **[SHIPPED OUTPUT]**. The P10 regression has
to watch those.

| Commit / PR | What changed | Path affected | Reaches the app? |
|---|---|---|---|
| `d4fa7be` (#350) | **Qwen3-TTS: the upsample conv bias was counted twice at every streaming chunk join.** Now the carried-over tail has the bias subtracted. `generateStream` joins chunks throughout, and `generate()` joins at 300-token chunks (clips over 24 s). Upstream measured SNR 24 dB → 111 dB against a full decode. **[SHIPPED OUTPUT]** | `Qwen3TTS/Qwen3TTSSpeechTokenizer.swift` (`DecoderBlockUpsample`) | Yes: all 4 Qwen3-TTS rows. `MLXAudioSpeaker.chunkStream` always streams. |
| `a7bd75b` (#315) | Qwen3-TTS ICL: the float32 speaker embedding is cast to the codec-prefix dtype before the concat. Before, it promoted prefill, KV cache and decode to float32 (upstream: 11.7 → 25.3 tok/s). **[SHIPPED OUTPUT]** (numerics; cloned voices only) | `Qwen3TTS/Qwen3TTS.swift` | Yes: Qwen3-TTS Base rows with a Voice Clone build (`VoiceCloneMLXArtifact.qwen3`) |
| `15d8223` (#346) | Parakeet `generateStream` now yields only "settled" tokens that a later chunk merge cannot revise. Before, a revised merge re-yielded the whole transcript, so appended text doubled. The final `.result` text is unchanged. **[SHIPPED OUTPUT]** for streamed partials | `Parakeet/ParakeetModel.swift` | Yes: Parakeet TDT v3, Parakeet Redux, Phonon-2 (Parakeet backbone). Affects `MLXAudioASR` `.token` accumulation; the app prefers `.result`. |
| `db17feb` (#349) | **STT `generateStream` no longer blocks the caller.** The new `makeSTTStream` runs the synchronous model on a `Task.detached`. Cancelling the consumer now cancels the producer, which checks `Task.checkCancellation()` per chunk. Applies to Canary, Cohere, FireRed, GLM-ASR, Granite, Lasr, Moonshine, Nemotron, Parakeet, SenseVoice, Voxtral Realtime, Wav2Vec2 and Whisper. Nemotron stream sessions break on `Task.isCancelled`. Behaviour, not numerics. | `MLXAudioSTT/Generation.swift` + 15 models | Yes: every shipped STT row except Qwen3-ASR / MOSS (already async). See R3. |
| `8fdde52` (#319), `7b431f3`, `66c3d24` | VoxCPM 1 / 1.5. **`loadAudioArray` now loops `AVAudioFile.read` until EOF**: before, it did one read, and AVFoundation can return a partial buffer for compressed files. VoxCPM2 DiT generalised (`VoxCPMLocDiTV2` → `VoxCPMLocDiT(conditioning: .tokens)`, `applyTransposeOutputPadding` default false; same graph for VoxCPM2). TTS loader honours the caller `HubCache` for local folders. | `MLXAudioCore/AudioUtils.swift`, `VoxCPM2/*`, `TTSModel.swift` | `loadAudioArray`: yes (`MLXAudioSpeaker.loadReferenceAudio`). Clone references are local WAVs, so output is expected unchanged; a compressed clip could now be read in full. VoxCPM2: no change expected. |
| `d60236d` (#314) | Phonon-2.5 STT: `.bps.tar.zst` / `.fermion` containers (zstd), extracted once into `<cache>/mlx-audio/phonon2/<digest>`. **`ModelUtils.resolveModelType` now reads only `config.json`** via the new `resolveModelConfig`, which is cache-first. `parakeet_tdt_five_value` dispatch. `Phonon2Checkpoint.prepare` passes plain Phonon-2 folders through. | `ModelUtils.swift`, `MLXAudioSTT.swift`, `Phonon2/*`, `Package.swift` | Loader only. Phonon-2 row unchanged (not a container). |
| `d0a133c` (#337) | LongCat TTS. **`SentencePieceTokenizer` lattice is now per Unicode scalar, not per grapheme**, and the vocabulary is keyed by bytes (no NFC/NFD aliasing). **[SHIPPED OUTPUT]** possible for text where grapheme ≠ scalar (combining marks, emoji ZWJ sequences, `\r\n`) | `MLXAudioCore/SentencePieceTokenizer.swift` | MOSS-TTS-Nano (encodes text); SenseVoice / Cohere (prompt ids; decode unaffected) |
| `e8dd23f` (#341) | Confucius4-TTS. `CAMPPlus` gains `batchnorm_` and `referenceCompatible` (default false) | `Codecs/S3Gen/CAMPPlus.swift` | Chatterbox: defaults unchanged, no output change expected |
| `0e55a69` (#348) | **Nemotron 3.5 Transcribe** (30B-A4B speech LM). `STT.loadModel(..., hfToken:)`. `"nemotron"`+`"transcribe"` repo names infer `nemotron_transcribe`. `Mamba2Scan` gains a time-step clip (default off). Shared `ExpertLinear` / `NemotronH`. Sortformer `ModulesConfig.init(...)` and `compressSpeakerCache` are public (additive). | `MLXAudioSTT/*`, `MLXAudioCore/LM/*`, `Sortformer/*` | Sortformer: additive. Our Nemotron row (`nemotron-3.5-asr-streaming`) resolves from `config.json` `model_type` and its name has no "transcribe". |
| `dc28d79` (#342) | Voxtral TTS. `TekkenTokenizer` moved from `MLXAudioSTT/Shared` to `MLXAudioCore` (rename, 89 % similar) | `MLXAudioCore/TekkenTokenizer.swift` | Voxtral Realtime uses it. The app does not reference the type. |
| `7eb6348`, `5935faf`, `fe2c182` | VibeVoice Realtime, Ming TTS, VibeVoice TTS. `MLXAudioCore` Qwen2 gains `includeNorm`, sliding-window and `projectToVocabulary` (defaults unchanged) | `MLXAudioCore/LM/Models/Qwen2.swift` | No (no shipped model uses Core Qwen2) |
| `7b431f3` (#320) | OuteTTS. `TTS.loadModel` now resolves the folder for `qwen3`/`qwen`/`llama`/`qwen2` types to tell OuteTTS / TADA apart (cache-first). `DescriptDACConfig` decodes HF-published keys. TTS gains a resource bundle (`outetts_default_speaker.json`). | `TTSModel.swift`, `Codecs/Descript/*` | Vyvo (`model_type: qwen3`) goes through the new check and stays `qwen3` (no OuteTTS tokens). Soprano is `soprano`. |
| `66c3d24` (#326) | Bark. Encodec transposed conv now uses native MLX instead of a scatter loop | `Codecs/Encodec/EncodecLayers.swift` | No (Encodec is used by Bark / Vocos only) |
| `f3b4829`, `a259232`, `09bea8a`, `3854579`, `1103198`, `7fdb657`, `8b41067`, `2896846`, `548acc8`, `20fef00`, `00b441a`, `84b7490` | New TTS: Higgs Audio v2/v3, rumik-oss, Audio8 (+ hybrid), KugelAudio, Ming-3.0 Tiny, ZONOS2, MeloTTS, TADA, Ming Omni. OmniVoice 8-bit + clone helpers. | new `Models/*` | No (not catalogued, §6) |
| `34ba52c` (#316) | IndexTTS mel positions | `IndexTTS/*` | No |
| `d085589` (#345) | Whisper: words lost at the 30 s cuts | `Whisper/*` | No (no Whisper row) |
| `9dd87cf`, `9762c57` | CI cache, README | | No |

---

## 2. Public API

| Symbol | Change | Our usage / fix |
|---|---|---|
| `STTGenerationModel.generateStream` (most models) | Returns at once; inference runs detached. Consumer cancellation cancels it. | `MLXAudioASR.transcribe` / `.bufferUntilStop` iterate with `for try await`, so no code change. Progress now updates per token, and `cancel()` now stops the work (R3). |
| `STT.loadModel(modelRepo:modelType:hfToken:cache:)` | New defaulted `hfToken:` | `MLXAudioASR.makeSTTModel` uses the 3-arg form (unchanged) |
| `ModelUtils.resolveModelConfig(repoID:hfToken:cache:)` | New, public; cache-first `mlx-audio/<repo_>/config.json`, else network | Indirect (via `resolveModelType`). Offline when the row is downloaded. |
| `loadAudioArray(from:sampleRate:)` | Same signature; reads until EOF | `MLXAudioSpeaker.loadReferenceAudio` |
| `TekkenTokenizer` | Module moved `MLXAudioSTT` → `MLXAudioCore` | Unused by the app |
| `ModulesConfig.init(numSpeakers:...)`, `SortformerModel.compressSpeakerCache` | New, additive | Unused (`MLXAudioDiarizer` loads via `fromModelDirectory`) |
| `OmniVoiceModel` voice-cloning helpers (#335) | New | Unused |
| `TTSModelSource.localDirectory(_, repoHint:, cache:)` | Internal enum gains `cache` | Internal |

No new required parameters on any API the app calls. No sample-rate changes for shipped classes. No new
`sending` / `Sendable` annotations on the app-facing APIs. `makeSTTStream` wraps the producer in
`UncheckedSendableBox`, which is internal. No app file imports both an `MLXAudio*` module and
`MLXLMCommon` / `MLXLLM`; 14 app files import `MLXAudio*`.

---

## 3. Conflicts and fork patches

No merge conflicts.

| File | Resolution |
|---|---|
| 12 files byte-identical to upstream (SAMAudioTextEncoder, LFMAudio/Processor, GLMASR, GraniteSpeech, Qwen3ForcedAligner, StreamingInferenceSession, Chatterbox, FishSpeechTokenizer, LlamaTTS, Qwen3, Qwen3TTS, Soprano) + comment-only markers in Qwen3ASR / MarvisTTSModel | **15 markers dropped.** The `Tokenizers.Tokenizer` disambiguation went moot when #287 removed MLXLMCommon. |
| `Audio8/Audio8Falcon.swift` (**new, merge**) | Bare `compile(shapeless:)` was ambiguous between `MLX.compile` and `MLXAudioCore.compile` → qualified `MLX.compile` (marked 2026-10-30) |
| `TTSModel.swift` (**new, P8**) | `gradium/phonon-31m-en` config.json switched to `model_type: "gradium_phonon_tts"` on 2026-10-21 (after #288), and upstream has no case for it. Added to `case "pocket_tts", "phonon", "gradium_phonon_tts":` (marked 2026-10-30). Upstream's own `phononGeneratesAudio` smoke test would hit the same throw. Drop the marker when upstream adds the case. |

Patch inventory: `116f2eb` 17 files / 21 markers → working tree **6 files / 8 markers**:
- `ModelUtils` `?? true` (2026-10-24): kept and still load-bearing. Phonon / Pocket pass extra patterns. Upstream left the line alone; it only replaced `resolveModelType` with a `config.json`-only read.
- Marvis ×3: codec folder-first, cosF32 + `.noUnusedKeys`, named `generationTask`.
- MossTTSNano ×2: stereo-to-mono, `LocalizedError`.
- `Audio8Falcon` ×1.
- `TTSModel` ×1.

`helper/docs/mlx-audio-swift.md` §6 needs its rows updated to match. In particular, all
`Tokenizers.Tokenizer` rows go. No new offline patch was needed: P3 read every new hub fetch in a shipped
type, and they are all cache-first. The one TTS addition is `resolveOrDownloadModel` for
qwen3/llama-typed repos (Vyvo).

---

## 4. Dependencies

- **New: `facebook/zstd` ≥ 1.5.7** (`libzstd`, MLXAudioSTT target), for Phonon-2.5 `.bps.tar.zst`. It is pinned in the app's `Package.resolved` (1.5.7).
- `MLXFast` product of mlx-swift is now a direct dependency of Core / Codecs / TTS / STT. Our fork `thirdparty/mlx-swift` (0.33.0) exports it.
- MLXAudioTTS now ships a resource bundle (`MLXAudio_MLXAudioTTS.bundle` with `outetts_default_speaker.json`), so the app embeds one more bundle.
- Floors unchanged: mlx-swift 0.32.3, swift-transformers 1.1.6, swift-huggingface 0.8.1.

No floor was raised, so there is no dependency move. The macOS app build is green.

---

## 5. Risk assessment — 6 risks (0 high, 3 medium, 3 low)

### R1 — Qwen3-TTS audio changes at chunk joins (MEDIUM)
**Problem:** #350 removes a doubled bias at every streaming chunk join. The app always streams Qwen3-TTS, so every Qwen3 row sounds different (intended: removes a click/offset at joins). #315 changes cloned-voice numerics (bf16 instead of float32 prefill), which is faster and close to, but not bit-equal to, P1. **Verify:** P10 `./run_audio_tests.py --compare /Volumes/ssd2t/modeltests/reports/audio-20261030-164254.json`: the 4 `mlx-audio-qwen3-tts-*` rows pass in default and cloned-voice passes, with the ASR round-trip similarity ≥ P1. **Status:** P10.

### R2 — Phonon 31m-en load needs the fork case (MEDIUM)
**Problem:** without the `gradium_phonon_tts` case, `TTS.loadModel` throws `unsupportedModelType`. Also check that the new-format config still decodes (`flow_lm` / `mimi` / `conditioners`) and that the `tokenizer.model` vocabulary = `n_bins`. **Verify:** P8 `./run_audio_tests.py --download --only mlx-audio-phonon-31m-en`, then `--only mlx-audio-phonon-31m-en` **Status:** verified 2026-10-30: the row loads as `PocketTTSModel` and passes the default-voice pass (`audio-20261030-172152`). The 31m-en checkpoint omits the speaker encoder ("use a bundled voice embedding"), so the row ships with `supports_voice_cloning: false`.

### R3 — STT streams now run detached and cancellable (MEDIUM)
**Problem:** #349 moves STT inference off the caller onto a detached task. Before, `generateStream` returned only after the whole transcription. Now:
- `MLXAudioASR.cancel()` really stops a running job (good).
- A second `transcribe` on the same `sttModel` while one is in flight would run two generations on one model at once. Before, the stream initializer serialised them.
- `unloadModel()` → `MLX.Memory.clearCache()` can overlap a producer that has not yet reached its next cancellation check.

The app has no in-flight guard in `MLXAudioASR.transcribe`. Callers today are user-serialised (dictation, file transcribe). **Verify:** P10 STT rows batch + live; one manual check on device: start a long file transcription, cancel, and start another at once. Check for no crash and no doubled text. **Status:** P10 + device.

### R4 — Parakeet-family streamed partials are shorter (LOW)
**Problem:** #346 holds back the last ~2 s of each chunk until the next chunk. Live partial text in `.bufferUntilStop` now lags but never repeats. The final text is unchanged. **Verify:** P10 `mlx-audio-parakeet-tdt-0.6b-v3`, `mlx-audio-parakeet-redux`, `mlx-audio-phonon-2` batch similarity equals P1. **Status:** P10.

### R5 — SentencePiece scalar lattice (LOW)
**Problem:** #337 tokenises per Unicode scalar. MOSS-TTS-Nano input with combining marks or emoji may tokenise differently, and SenseVoice / Cohere prompt ids could shift where vocab entries were NFC/NFD aliases. **Verify:** P10 `mlx-audio-moss-tts-nano-100m`, `mlx-audio-sensevoice-small`, `mlx-audio-cohere-transcribe-4bit` pass. **Status:** P10.

### R6 — `loadAudioArray` reads to EOF (LOW)
**Problem:** clone references longer than one AVFoundation buffer were previously truncated for compressed formats; now they are read in full. App references are `reference.wav` (PCM), so no change is expected. **Verify:** P10 cloned-voice passes (all clone-capable TTS rows) match P1. **Status:** P10.

Shipped classes with **no** source change in this range: PocketTTS, Marvis / Mimi, Chatterbox (defaults
only), Soprano, MOSS-TTS-Nano model code, Vyvo (`Qwen3Model`), VoxCPM2 (refactor, same graph),
Qwen3-ASR, Granite / GLM / SenseVoice / Cohere / Voxtral Realtime / Nemotron streaming (stream wrapper
only), Sortformer (additive), Silero / Smart-Turn VAD, DeepFilterNet STS, MOSS-Transcribe-Diarize.

---

## 6. New models evaluated for the catalogue (2026-10-30)

Sizes are HF tree sums (HF API, 2026-10-30). Limits: TTS / STS / VAD / diarization ≤ 1000 MB, STT ≤ 700 MB.

| Model | Repo size | Decision | Reason |
|---|---|---|---|
| **Phonon 31m-en** (`gradium/phonon-31m-en`) | 223 MB tree; **≈ 104 MB needed** (`config.json`, `model.safetensors` 102.8 MB, `tokenizer.model` 0.07 MB, `default-voice.safetensors` 0.3 MB) | **Accepted** | TTS, English, Apache-2.0, open (not gated). PocketTTS family, so no new wrapper (`MLXAudioSpeaker`'s Pocket voice branch applies). Needs the §3 fork case. Do not mirror the `*.gguf` files. No voice cloning: the checkpoint has no speaker encoder. |
| Phonon-2.5 English (`FermionResearch/Phonon-2.5-English`) | 321 MB; needed ≈ 164 MB (`config.json` + `phonon-2.5-en.bps.tar.zst`) | Candidate (verify) | STT, CC-BY-4.0, existing `phonon2` dispatch. **Caveat:** first load extracts into `<models>/mlx-audio/phonon2/<sha>`, a folder outside the row's cache folder, which Settings cannot size or delete. It would need a storage hook first. |
| Phonon-2.5 (multilingual, `FermionResearch/Phonon-2.5`) | 319 MB | Candidate (verify) | Same as above; `languages` from config |
| VoxCPM 1.5 4-bit (`mlx-community/VoxCPM1.5-4bit`) | 627 MB | Candidate | TTS, Apache-2.0, voice cloning, `voxcpm` dispatch, generic speaker path. Smaller sibling of the shipped VoxCPM2-4bit (2301 MB). 8-bit is 1028 MB (over). |
| MeloTTS English (`mlx-community/MeloTTS-English-MLX`) | 646 MB (`bert_weights.npz` 438 + `model.safetensors` 208) | Candidate (verify) | TTS, MIT, preset voices, no cloning. **Its English frontend (`vocab.txt`, `checkpoint20.npz`, `cmudict.zip`, NLTK tagger, `homographs.en`) downloads on first load.** Those files must be mirrored into the row's `frontend/` folder or it fails offline. English only. |
| OmniVoice 8-bit (`mlx-community/OmniVoice-8bit`) | 1449 MB (bf16 1639 MB, fp 3267 MB) | Rejected | Over 1000 MB |
| Audio8 TTS Preview 0.1B / 0.6B bf16 (+ hybrid #339) | 1696 / 2565 MB | Rejected | Over 1000 MB, no 4-bit; 0.1B licence "other" |
| OuteTTS 1.0 0.6B fp16 + DAC codec | 1221 + 296 MB | Rejected | Over 1000 MB |
| VibeVoice Realtime 0.5B 8-bit | 1221 MB | Rejected | Over 1000 MB |
| LongCat AudioDiT 1B 4-bit | 1418 MB | Rejected | Over 1000 MB; no licence tag |
| Bark small (+ Encodec 76 MB) | 1611 MB | Rejected | Over 1000 MB |
| Nemotron 3.5 Transcribe 30B-A4B 4-bit (`nativ-community/...-EA-4bit`) | 19276 MB | Rejected | Over 700 MB (STT). NVIDIA original 67.5 GB. |
| Voxtral 4B TTS 4-bit / rumik-oss-1 | 2543 / 7188 MB | Rejected | Over 1000 MB and CC-BY-NC-4.0 |

Over the limit, not evaluated further: Confucius4 int8 (2804 MB), Ming-3.0 TTS 0.5B bf16 (3003 MB),
Ming-3.0 TTS Tiny 4-bit (6663 MB), Ming Omni 16.8B-A3B 4-bit (10104 MB), Higgs Audio v2 3B q6 (4767 MB),
Higgs TTS 3 4B (9322 MB), ZONOS2 4-bit (4703 MB), TADA 1B (4588 MB, Llama 3.2 licence), VibeVoice 1.5B
(5408 MB), KugelAudio 0 open (18688 MB).

---

## 7. Opportunities (not built)

| Upstream feature | Consumer that would use it | Value |
|---|---|---|
| Cancellable STT streams (#349) | `MLXAudioASR` | Add an in-flight guard (one generation per model) and await the stream's end on cancel before `unloadModel` clears the cache (R3) |
| `STTGenerateParameters.task = "translate"` for Nemotron 3.5 Transcribe (#348) | ASR settings | Speech translation to `language`, but no model under the 700 MB limit supports it |
| VoxCPM 1.5 / MeloTTS / Phonon-2.5 rows | Catalogue (§6) | No new wrapper: all dispatch through `TTS.loadModel` / `STT.loadModel` and the generic speaker / ASR paths. MeloTTS needs frontend mirroring. Phonon-2.5 needs the extraction cache counted in Storage. |
| OmniVoice voice-cloning helpers (#335) | `VoiceCloneMLXBuilders` | Reusable clone prep if OmniVoice ever fits the size limit |
| `MLXAudioASR.cacheIgnoringFamilies` | — | Unchanged by this range. The `"voxtral"` substring still routes any Voxtral repo to `VoxtralRealtimeModel`. |

---

## 8. Verification

- P1 baseline: `/Volumes/ssd2t/modeltests/reports/audio-20261030-164254.json`, 26 pass / 0 fail / 0 skip.
- P3: `swift package resolve` rc=0. `swift build` was not run (cold build); the macOS app build covers it.
- P5: macOS app build (`AIAssistantMac`) green after the `Audio8Falcon` `MLX.compile` qualification.
- P7: consumer review found no break needing an app change (§2, R3).
- Done: P8 Phonon row load test (PASS, default voice). Pending: P10 regression `--compare` against the P1 report; `MLXAudioCatalogueSchemaTests`; iOS build.
- macOS cannot verify: R3 cancel/restart on device, and iOS memory for the new Phonon row.
