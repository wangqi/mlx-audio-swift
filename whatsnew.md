# mlx-audio-swift: What's New (tag-20260928 → tag-20261018)

Merged upstream `Blaizzy/mlx-audio-swift` `main` into our fork on 2026-10-18. **6 upstream
commits**, 3 files with conflicts (all resolved, see §3).

| | tag-20260928 | tag-20261018 |
|---|---|---|
| Head | `e74a08a` (our Marvis Mimi codec patch) | merge of `244458d` |
| Diffstat, `Sources/` | | 58 files, +4641 / −766 |
| Diffstat, `Tests/` | | 17 files, +1270 / −876 |
| Dependency pins | mlx-swift-lm 3.31.3 | mlx-swift-lm **3.32.3**, mlx-swift 0.32.3 (`Package.resolved`) |

The pins match our stack: upstream `3.32.3` (`3b339ad`) is an ancestor of our `thirdparty/mlx-swift-lm`
fork HEAD, and `thirdparty/mlx-swift` is on `prism-1bit-0.32.3`. The app overrides both with the local
packages anyway.

---

## 1. Changes

| Commit | What changed | Path affected |
|---|---|---|
| `b7c7971` (#269) | Qwen3-ASR batched inference pads more carefully. | `Qwen3ASR.swift` |
| `8d86630` (#276), `244458d` (#282) | Test fixes and a parallel test runner. | `Tests/` only |
| `dbe5eaa` (#281) | **Swift 6 concurrency pass for mlx-swift / mlx-swift-lm 3.32.3.** `refAudio` and returned audio become `sending`; `AudioGeneration` is no longer `Sendable`; new `SendingBox` moves tensors into generation tasks; new `TTSGenerationCache`. Sizeable rewrites of `Qwen3.swift` (Vyvo), `Soprano.swift` and `Sortformer.swift`; small edits across Chatterbox, Marvis, MOSS, Pocket, Qwen3-TTS, Parakeet, Silero. | Every TTS model, Sortformer, Parakeet |
| `2cb59ad` (#262) | **Dia 1.6B** dialogue TTS (encoder–decoder, 9 DAC codebooks, CFG). | new `Models/Dia/` |
| `a0bd39c` (#175) | **VoxCPM2** 2B TTS, 48 kHz, voice cloning. | new `Models/VoxCPM2/` |

---

## 2. Public API

| Symbol | Change | Our usage |
|---|---|---|
| `SpeechGenerationModel.generate/generateStream(refAudio:)` | `refAudio: sending MLXArray?`, returns `sending` | `MLXAudioSpeaker.chunkStream` reuses one reference clip for every chunk. The app builds in Swift 5 mode, so this compiles without diagnostics; sequential reuse of a read-only tensor is safe at runtime. It becomes an error if the app moves to Swift 6 mode. |
| `AudioGeneration` | no longer `Sendable` | `toSamplesStream()` — compiles unchanged |
| `loadAudioArray(from:sampleRate:)` | returns `sending (Int, MLXArray)` | `MLXAudioSpeaker.loadReferenceAudio` — unchanged |
| `SendingBox` | new, `package` access | not used by the app |

---

## 3. Conflicts and fork patches

| File | Resolution |
|---|---|
| `Parakeet/ParakeetModel.swift` ×2 | Took upstream. Our 2026-09-18 patch qualified the traces as `MLX.compile` because MLXLMCommon's `@Sendable` overloads shadowed it; upstream now writes `MLX.compile(inputs: [encoder])` / `MLX.compile(inputs: [decoder, joint])`, which is qualified and also declares the weights as compile inputs. 2 markers dropped. |
| `Marvis/CSMModel.swift` ×2 | Took upstream. Our 2026-08-31 `try? makePromptCache(...)` adaptation is moot: upstream builds `KVCacheSimple()` per layer directly. 2 markers dropped. |
| `Marvis/MarvisTTSModel.swift` | Kept both: upstream's `SendingBox(refAudio)` / `take()` and our 2026-03-21 named `generationTask` that `onTermination` cancels. |

Every other `wangqi modified` marker in `Sources/` survived unchanged.

No new fork patch. A codec-from-folder patch for Dia (like Marvis's Mimi one) was written and then
reverted when Dia was rejected for the catalogue (§5).

---

## 4. Risk assessment

### Low

| Area | Finding |
|---|---|
| Shipped models | All 19 catalogue rows passed `run_audio_tests.py` after the merge, identical to the tag-20260928 run (`audio-20261018-073808`). |
| Concurrency changes | Compile-time only for the app (Swift 5 mode). No behaviour change observed. |

### Medium

1. **`sending` reuse.** See §2 — a latent Swift 6 migration item in `MLXAudioSpeaker`.
2. **Dia ships in the package** although no catalogue row uses it. Its loader always fetches
   `mlx-community/descript-audio-codec-44khz` on first use and has no local-directory branch.

### High

None.

---

## 5. New models evaluated for the catalogue (2026-10-18)

| Model | Verdict | Evidence |
|---|---|---|
| VoxCPM2 4-bit (2301 MB) | **Added** | Gate PASS default + cloned; Whisper large-v3 transcribes both exactly; peak 3667 MiB (gate 4034 MiB → 8 GB devices); ~1x real time on an M-series Mac |
| MOSS Transcribe-Diarize (1833 MB) | **Added**, live streaming | Gate similarity 0.92 after the app's `[Speaker N]` rewrite; 51 live partials |
| Cohere Transcribe 4-bit (1506 MB) | **Added**, live streaming | Gate similarity 1.0 batch and live; no language auto-detect |
| Voxtral Realtime 4B 4-bit (3149 MB) | **Added**, live streaming | Gate similarity 1.0 batch and live, via the app's new `LiveSessionDecoder` |
| Dia 1.6B fp16 | **Rejected** | Near-silent output (Whisper hears nothing), no EOS within the cap, 5.36 GB peak, 3.5x slower than real time on Mac, no fixed voice |

Details: `helper/docs/mlx-audio-swift.md` §4.2 and §4.14.
