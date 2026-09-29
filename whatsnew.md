# mlx-audio-swift: What's New (tag-20260918 → tag-20260928)

Merged upstream `Blaizzy/mlx-audio-swift` `main` into our fork on 2026-09-29
(merge commit `eb6369f`). **3 upstream commits** in range, no local commits, no conflicts.

| | tag-20260918 | tag-20260928 |
|---|---|---|
| Head | `44979c9` (our Spark adaptation) | `eb6369f` (merge) |
| Diffstat, `Sources/` | | 5 files, +299 / −46 |
| Diffstat, `Tests/` | | 3 files, +375 / −8 |
| Files touched outside `VoxtralRealtime/` | | **none** |

The whole upgrade is one model family: **Voxtral Realtime** (`mistralai/Voxtral-Mini-4B-Realtime`
and its MLX conversions), and within it mostly the **streaming session**. Every other STT, TTS,
VAD, STS and codec source file is byte-identical to tag-20260918.

---

## 1. Changes

All three are memory/performance fixes for long-running Voxtral Realtime sessions, co-authored with
Lucas Newman. Each one replaces work proportional to *stream length* with work proportional to the
*step*.

| Commit | What changed | Effect | Path affected |
|---|---|---|---|
| `ad9c2c4` (#263) | Stream session drops conv-stem rows below `encState.consumed` and adapter rows below `decPos` once they are consumed; `...Dropped` counters keep frame/decode indices absolute. `feedIncremental` gains a `startIndex:` parameter. `step()` stops buffering samples once the stream finished on EOS / `maxTokens`. | Upstream measured a 20-minute Voxtral Mini 4B stream on an M-series MacBook Pro still **growing ~23 MB of active memory per minute** after the decoder window filled. Retained rows are now bounded independently of stream length. | `VoxtralRealtimeStreamSession` only |
| `0b55010` (#264) | `VoxtralRealtimeDecoderKVCache` goes from a value type holding exactly the sliding window (concatenate every token's K/V onto the whole window, then slice the oldest row off — in every layer) to a **class with preallocated storage grown in 256-row blocks**. New rows go in by slice update; out-of-window rows are compacted to row 0 in one move once a block has piled up. Index math lives in `VoxtralRealtimeDecoderKVCacheAppendPlan`, testable without Metal. | Removes a **full-window copy per token per layer** (8 192 rows/layer once the window fills on Mini 4B) and the transient doubling of cache memory that copy caused. | Decoder — **both** offline `generate` and streaming |
| `01dec7c` (#265) | New `VoxtralRealtimeTranscriptText` appends each token's UTF-8 bytes incrementally (holding at most 3 bytes of an unfinished character) instead of re-decoding the whole token list and diffing the whole text every step. `decodeStreaming(_:)` → `streamingTokenBytes(_:)` (internal); tokenizer's `tokenBytes(for:)` goes from private to internal. | Per-step text cost is O(1) instead of O(transcript). Mattered for hour-long streams. `Delta.text` semantics unchanged. | `VoxtralRealtimeStreamSession` only |

Test coverage added upstream: 64 random token sequences splitting multi-byte characters, combining
marks, emoji and invalid bytes match a one-pass decode scalar-for-scalar; 2 000 plan appends and
1 500 real-array appends match the old concatenate-then-trim exactly; a one-minute stream crossing
the encoder window 47 times matches offline `generate` exactly.

### What matters for iOS devices

The pattern is the right one for iOS — on a phone the jetsam limit, not speed, ends a long session,
and all three commits turn unbounded growth into a bounded working set. But **none of it reaches a
Privacy AI user today**:

- No Voxtral model is in `helper/mlxaudio_model*.json`, so nothing in the catalogue loads it.
- Our live-mic path (`MLXAudioASR.startStreaming`) builds a `StreamingInferenceSession` for **Qwen3
  only**; `VoxtralRealtimeStreamSession` is never constructed by the app.
- The one code path that names Voxtral is `MLXAudioASR.loadCacheIgnoringModel` →
  `VoxtralRealtimeModel.fromDirectory(_:)`, which is unchanged and only reached if a Voxtral repo is
  ever added.

So the value of this upgrade is **readiness**: if Voxtral Realtime is ever added as a live-dictation
model, the streaming session is now viable for long sessions on a 6–8 GB device, where previously
per-minute memory growth would have made a long dictation a jetsam candidate.

---

## 2. Public API

| Symbol | Change | Our usage |
|---|---|---|
| `VoxtralRealtimeStreamSession.text` | Same signature; now backed by `transcript.text`. Doc now notes reading it costs O(length) — prefer `Delta.text`. | Not used |
| `VoxtralRealtimeStreamSession.step(_:)` | Returns an empty `Delta` after the stream has finished (previously appended samples and advanced to an empty result). | Not used |
| `VoxtralRealtimeModel.decodeStreaming(_:)` | Removed → `streamingTokenBytes(_:)`. Both **internal**. | Not used |
| `VoxtralRealtimeDecoderKVCache` | `struct` → `final class`. **Internal**. | Not used |
| `VoxtralRealtimeModel.fromDirectory(_:)` | Unchanged | `MLXAudioASR.swift:264` |

No `public` declaration was added, removed or changed in signature. No new `#if os(...)` and no
new platform-only API.

---

## 3. Our fork's patches

All **26** `wangqi modified` markers in `Sources/` survive — none of them are in
`VoxtralRealtime/`, so the merge could not touch them. The previous local commit (`44979c9`, adapt
`SparkModel` to the throwing `encode` / `newCache` APIs) is the base of this range and is intact.

---

## 4. Risk assessment

### Low — verified by inspection

| Area | Finding |
|---|---|
| **Blast radius** | 5 source files, all under `Sources/MLXAudioSTT/Models/VoxtralRealtime/`. All 17 rows in `helper/mlxaudio_model.json` — TTS (Soprano, Pocket TTS, Qwen3-TTS ×2, VyvoTTS, Marvis, Chatterbox Turbo, MOSS-TTS-Nano), STT (Qwen3-ASR, Parakeet TDT v3, GLM-ASR-Nano, SenseVoice, Granite Speech, Nemotron streaming), Sortformer diarization, Smart Turn v3 and DeepFilterNet v3 — compile from byte-identical sources. **Zero output drift is possible on shipped models.** |
| **Package compilation** | `swift build --target MLXAudioSTT` on the merged package (macOS host): **passed**, exit 0. |
| **App compilation** | Our only Voxtral call site, `VoxtralRealtimeModel.fromDirectory`, is unchanged. No public API moved, so `libs/audio/mlxaudio/*.swift` needs no edit. |
| **Binary size** | +1 file (`VoxtralRealtimeTranscriptText.swift`, 84 lines). Negligible. |
| **Platform safety** | No platform conditionals added; only `MLXArray.zeros` / slice updates, available everywhere MLX is. |

### Medium — only if Voxtral is ever enabled

1. **KV cache aliasing.** The decoder cache is now a reference type mutated in place by `append`.
   Upstream's decoder passes each cache forward and never keeps an older one, but any future caller
   that snapshots `[VoxtralRealtimeDecoderKVCache?]` to rewind or branch a decode (as prompt
   caching does in the LLM stack) would silently share state. The type is internal, so this can
   only bite from inside the package.

2. **In-place update relies on MLX buffer uniqueness.** The doc comment warns that holding `keys` /
   `values` across an `append` makes MLX copy the whole storage. That is a performance cliff, not
   a correctness bug — but it is invisible, and it would undo #264's win without failing a test.

3. **Storage capacity steps.** Capacity grows in 256-row blocks and compacts when a block of
   out-of-window rows piles up, so resident cache memory is roughly `slidingWindow + 256` rows per
   layer rather than exactly `slidingWindow`. Up to ~3% extra at an 8 192-row window — in exchange
   for no longer transiently holding two full copies during every append.

### High

None.

---

## 5. Verification — every catalogue model, through the app's own wrappers

`helper/scripts/model_regression/run_audio_tests.py` on macOS, 2026-09-29, against weights
downloaded from the mirror `flyingfishinwater/mlxaudiomodels` (all 171 files verified byte-identical
to the mirror by sha256 / git blob hash). Swift suite:
`testcases/engines/mlxaudio/MLXAudioCatalogueRegressionTests.swift`.
Report: `/Volumes/ssd2t/modeltests/reports/audio-20260929-115552.md`.

**17 passed, 0 failed, 0 skipped.**

| Type | Model | Engine class | Result |
|---|---|---|---|
| stt | Qwen3-ASR 0.6B 4bit | `Qwen3ASRModel` | similarity 0.909 (= baseline) |
| stt | Nemotron ASR streaming 0.6B 8bit | `NemotronASRModel` | 0.818 (= baseline) |
| stt | SenseVoice Small | `SenseVoiceModel` | 0.909 (= baseline) |
| stt | GLM-ASR Nano 4bit | `GLMASRModel` | 1.0 (= baseline) |
| stt | Granite Speech 1B 4bit | `GraniteSpeechModel` | 1.0 (= baseline) |
| stt | Parakeet TDT 0.6B v3 | `ParakeetModel` | 1.0 (= baseline) |
| tts | Soprano 1.1 80M | `SopranoModel` | audio produced |
| tts | Pocket TTS | `PocketTTSModel` | audio produced |
| tts | MOSS-TTS-Nano 100M | `MossTTSNanoModel` | audio produced (cloned voice) |
| tts | Marvis TTS 250M | `MarvisTTSModel` | audio produced |
| tts | Chatterbox Turbo 4bit | `ChatterboxModel` | audio produced |
| tts | VyvoTTS EN 4bit | `Qwen3Model` | audio produced |
| tts | Qwen3-TTS 0.6B Base 4bit | `Qwen3TTSModel` | audio produced |
| tts | Qwen3-TTS 0.6B CustomVoice | `Qwen3TTSModel` | audio produced |
| diarization | Sortformer 4spk | `SortformerModel` | 1 segment, 3.84 s speech |
| vad | Smart Turn v3 | `SmartTurnModel` | endpoint detected |
| sts | DeepFilterNet v3 | `DeepFilterNetModel` | length preserved |

The 2026-09-18 baseline covered only the six STT rows (every other row skipped on the development
Mac's iCloud Drive storage), so the other eleven had never been exercised by the gate before this
run. Getting them green found three problems **unrelated to this upgrade**, recorded in
`helper/docs/mlx-audio-swift.md` §4.14:

1. **DeepFilterNet could never load in the app** — `ModelUtils.resolveOrDownloadModel` only looks
   for top-level weights, deleted the install and re-downloaded from upstream. Fixed in
   `MLXAudioSTS.loadModel` (loads through `DeepFilterNetModel.fromLocal`).
2. **The bundle-fallback catalogue decode dropped every snake_case field** (`loader_repo`,
   `model_type`, …) because of `.convertFromSnakeCase`. Fixed in both bundle loaders; covered by
   `MLXAudioCatalogueSchemaTests`.
3. **Marvis and MOSS-TTS-Nano fetched a second repository from upstream on first use** (the Mimi
   codec, 385 MB; the MOSS audio tokenizer, 44 MB), which the mirror did not carry, so they needed
   the network on first use and failed offline. Resolved: both now ship byte-identical inside their
   model folders on the mirror (`codec/`, `audio_tokenizer/`), and Marvis loads the codec from
   there through the new `Mimi.fromWeightsFile` (fork patch #4, `helper/docs/mlx-audio-swift.md` §6).

Issues 1 and 2 were also reworked after review (DeepFilterNet loads by item through the shared
`ensure_file_ready` gate; FluidAudio's two bundle loaders had the same decode bug and are fixed too).
The gate now FAILs any row that writes outside its own folder and runs a cloned-voice pass for every
voice-cloning row. Re-run the same day: report `/Volumes/ssd2t/modeltests/reports/audio-20260929-143301.md`,
**17 passed, 0 failed, 0 skipped**, identical to the run above under `--compare`; Pocket TTS, Marvis,
Chatterbox and both Qwen3-TTS pass both the default and the cloned voice, MOSS-TTS-Nano its cloned
voice. With `codec/` moved out of the staged Marvis folder the row FAILs "fetches
`kyutai_moshiko-pytorch-bf16` from outside the mirror at first use", as intended.

---

## 6. Integration opportunities for `libs/audio/mlxaudio/`

**No changes are required, and none are worth making now.** This range adds no new API that our
shipped models can use, and it does not touch any code path the app runs.

If Voxtral Realtime is added to the catalogue later:

- It already loads through `loadCacheIgnoringModel` (`fromDirectory`, app storage, no second
  download) for batch transcription, and offline `generate` benefits from #264 automatically.
- Live dictation would need a second branch in `startStreaming`: today it downcasts to
  `Qwen3ASRModel` for `StreamingInferenceSession`. Voxtral's streaming entry point is its own
  `VoxtralRealtimeStreamSession(model:…)` driven by `step(_:)` / `finish()`; consume `Delta.text`
  per step rather than reading `.text`, which is now documented as O(length).
- Run it through `helper/scripts/model_regression/run_audio_tests.py` before shipping, as for any
  new row.
