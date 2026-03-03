# MLX-Audio-Swift Update: tag-20260224 → tag-20260302

**Update Date:** March 2, 2026
**Previous Version:** tag-20260224
**Current Version:** tag-20260302
**New Commits (upstream):** 2 functional commits

---

## Executive Summary

This is a **low-risk maintenance update** with two focused improvements: a unified MLX-native audio resampling API that eliminates AVFoundation dependency, and a more robust model download validation system that prevents silent failures from incomplete or corrupted downloads.

### Risk Assessment: **LOW** (1/5)

| Risk Area | Level | Reason |
|-----------|-------|--------|
| MLXArray resampling overload | **LOW** | Additive API; existing `[Float]` variant unchanged |
| AVFoundation removal from tool apps | **LOW** | Only CLI tool binaries affected, not the library |
| Download validation & cache clearing | **LOW** | More defensive; adds failure path that was previously silent |
| New `ModelUtilsError` type | **LOW** | Additive; only callers that do not handle the new error are affected |
| API breaking changes | **NONE** | No existing public API removed or changed |

---

## Commits Included

| Commit | Author | Date | Description |
|--------|--------|------|-------------|
| `1e49dcf` | Prince Canuma | 2026-02-25 | Enhance model download validation and cache management in ModelUtils (#76) |
| `b76c81f` | Lucas Newman | 2026-02-25 | Add an MLXArray-based audio resampling variant (#77) |

---

## Change Details

### 1. MLXArray-Based Audio Resampling API (#77)

**Files changed:**
- `Sources/MLXAudioCore/AudioUtils.swift` (+15 lines)
- `Sources/Tools/mlx-audio-swift-sts/App.swift` (-68 lines)
- `Sources/Tools/mlx-audio-swift-stt/App.swift` (-84 lines)
- `Sources/Tools/mlx-audio-swift-tts/App.swift` (-14 / +5 lines)
- `Tests/MLXAudioSmokeTests.swift` (minor update)

**What changed:**

A new `resampleAudio(_:from:to:)` overload has been added to `MLXAudioCore.AudioUtils`:

```swift
/// Resample audio to a target sample rate.
public func resampleAudio(
    _ samples: MLXArray,
    from sourceSampleRate: Int,
    to targetSampleRate: Int
) throws -> MLXArray
```

This variant accepts and returns `MLXArray` directly, wrapping the existing `[Float]` implementation with a zero-copy bridge. Previously each tool app (STS, STT, TTS) had its own private copy of `AVAudioConverter`-based resampling code (~80–95 lines each), duplicated three times. All three tool apps now call `MLXAudioCore.resampleAudio()` instead.

**What was removed:**
- Private `resampleAudio(_:from:to:)` implementations using `AVAudioConverter` from all three tool apps
- `AppError.audioResampleFailed` error case from STS and STT tool apps
- `@preconcurrency import AVFoundation` from the STT tool app

**Net result:** ~180 lines deleted across tool apps, 15 lines added to the shared library. The public library surface gains a more ergonomic `MLXArray → MLXArray` resampling path.

**iOS Device Impact:**
- The library-level change (`AudioUtils.swift`) is purely additive — existing callers are unaffected
- Tool app changes are CLI-only, not shipped on device
- The new `MLXArray`-based overload avoids manual `asArray(Float.self)` / `MLXArray(...)` round-trips, reducing intermediate allocations when resampling audio in hot paths

**Risk Level: LOW** — additive API; all existing call sites unchanged

---

### 2. Model Download Validation and Cache Management (#76)

**File changed:** `Sources/MLXAudioCore/ModelUtils.swift` (+38 lines, -2 lines)

**What changed:**

**Before:** When a model download was interrupted or produced zero-byte weight files, `ModelUtils` silently left the corrupted directory in place. On the next launch the app would attempt to load the broken model and fail with a cryptic error.

**After:** Three improvements:

#### a) Post-download validation
After `Hub.snapshot()` completes, the code now verifies that at least one file with the required extension (e.g. `.safetensors`, `.npz`) exists and has a non-zero file size. If not, the cache is cleared and `ModelUtilsError.incompleteDownload` is thrown.

#### b) Atomic dual-directory cache clearing
The new private `clearCaches(modelDir:repoID:hubCache:)` helper removes both the model directory **and** the Hub metadata cache directory together. Previously only the model directory was deleted, leaving stale Hub metadata that could prevent a clean re-download.

#### c) User-friendly error type
New public `ModelUtilsError.incompleteDownload` error with a clear English message:

```
"Downloaded model 'repo/name' has missing or zero-byte weight files.
 The cache has been cleared — please try again."
```

**iOS Device Impact:**
- On-device model downloads now self-heal: a failed or interrupted download clears its own broken cache automatically
- Users see a meaningful error message instead of a cryptic Swift runtime crash
- Eliminates the previous failure mode where users had to manually clear app data to recover from a bad download
- Particularly impactful on iOS where network interruptions during large model downloads (1–8 GB) are common

**Risk Level: LOW** — defensive improvement; the only new failure path covers downloads that were already silently broken

---

## Files Changed Summary

| File | Change Type | Net Lines |
|------|-------------|-----------|
| `Sources/MLXAudioCore/AudioUtils.swift` | New API | +15 |
| `Sources/MLXAudioCore/ModelUtils.swift` | Enhancement | +38 / -2 |
| `Sources/Tools/mlx-audio-swift-sts/App.swift` | Refactor (tool only) | -68 |
| `Sources/Tools/mlx-audio-swift-stt/App.swift` | Refactor (tool only) | -84 |
| `Sources/Tools/mlx-audio-swift-tts/App.swift` | Refactor (tool only) | -9 |
| `Tests/MLXAudioSmokeTests.swift` | Minor update | ±4 |

**Total:** 6 files changed, ~53 insertions, ~165 deletions (net -112 lines)

---

## iOS Device Impact Summary

| Change | iOS Benefit | Action Required |
|--------|-------------|-----------------|
| MLXArray resampling overload | Avoids `[Float]` round-trips in hot paths | None (additive API) |
| Post-download validation | Detects zero-byte weight files immediately after download | None |
| Dual-directory cache clearing | Prevents stale Hub metadata after failed download | None |
| `ModelUtilsError.incompleteDownload` | User-visible error instead of silent crash | Consider surfacing in download UI |

---

## Our Integration — Action Items

### Required: **NONE**

This is a drop-in replacement. No code changes required in AIAssistant.

### Recommended — Surface Download Error

`ModelUtilsError.incompleteDownload` is now thrown where previously a silent failure occurred. If AIAssistant catches errors from `ModelUtils.loadModel()` or the download path, consider adding a case for this new error to show users a "Download was incomplete — please retry" message.

---

## Testing Checklist

- [ ] Project builds successfully with updated submodule
- [ ] MLX Audio TTS generation works (basic speech synthesis)
- [ ] MLX Audio STT transcription works (basic transcription)
- [ ] Model download from HuggingFace completes successfully
- [ ] (Optional) Simulate interrupted download — verify cache is cleared and error message appears
- [ ] No runtime warnings or crashes

---

## Overall Risk Rating

**LOW — safe to upgrade**

Both changes are defensive improvements: one adds a more ergonomic `MLXArray`-based resampling overload (purely additive), the other makes download failure handling more robust. No model architectures changed, no inference code changed, no public API removed.

---

**Generated:** 2026-03-02
**Covers commits:** 2 upstream functional commits (tag-20260224 → tag-20260302)
