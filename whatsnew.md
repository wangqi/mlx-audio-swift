# mlx-audio-swift What's New

## tag-20260403 (2026-04-03)

### Changes from tag-20260328

**Upstream commits merged from Blaizzy:main:**

- `2fd4145` — Add missing MLXFast dependency to MLXAudioSTT (#132) by Hurryitup

### Details

#### MLXFast Dependency Added to MLXAudioSTT

**Package.swift** now declares `MLXFast` as a dependency of the `MLXAudioSTT` target:

```swift
// Before (tag-20260328)
dependencies: [
    "MLXAudioCore",
    "MLXAudioCodecs",
    .product(name: "MLX", package: "mlx-swift"),
    .product(name: "MLXNN", package: "mlx-swift"),
    ...
]

// After (tag-20260403)
dependencies: [
    "MLXAudioCore",
    "MLXAudioCodecs",
    .product(name: "MLX", package: "mlx-swift"),
    .product(name: "MLXFast", package: "mlx-swift"),   // <-- added
    .product(name: "MLXNN", package: "mlx-swift"),
    ...
]
```

`MLXFast` exposes Metal-accelerated primitives (RoPE, scaled dot-product attention, RMS norm)
that can be used inside STT model inference graphs. Previously `MLXFast` was only a dependency
of `MLXAudioCodecs` (added in #128 / tag-20260328); it was accidentally omitted from `MLXAudioSTT`.

**Effect on iOS devices:** STT models (Qwen3-ASR, GLM-ASR, Granite Speech, Parakeet, FireRed,
SenseVoice) can now link against MLXFast operations directly. This closes a potential
compile-time symbol resolution gap; future STT model implementations can use fast attention
and norm ops without a Package.swift change.

### Upgrade Risk Assessment

| Risk Area | Level | Notes |
|-----------|-------|-------|
| API compatibility | None | No public API changed — pure dependency addition |
| Runtime behavior | Minimal | Existing STT models that do not call MLXFast ops are unaffected |
| iOS device | None | MLXFast ships in the existing mlx-swift xcframework |
| Build stability | None | Resolves a missing-symbol risk rather than introducing one |
| Memory | None | MLXFast already loaded by MLXAudioCodecs in the same process |

**Overall risk: Low.** This is a build-correctness fix. No code changes to STT model
implementations, no new models, no API changes.

---

## tag-20260328 (2026-03-28)

### Changes from tag-20260321

**Upstream commits merged from Blaizzy:main:**

- `be7b8f6` — Add MLXFast dependency for MLXAudioCodecs (#128) by Noor Bhatia
- `4aaf7cd` — feat: add Kokoro TTS with multilingual support (#124) by Aleksandr Beshkenadze

### Details

#### MLXFast Dependency Added to MLXAudioCodecs
Enables hardware-accelerated codec operations via Metal GPU for codec targets.

#### Kokoro TTS + KittenTTS (StyleTTS2 family)

New `Sources/MLXAudioTTS/Models/StyleTTS2/` directory with:

- **MLXAudioG2P** — new standalone module for grapheme-to-phoneme conversion
  - `NeuralPhonemizer`: multilingual ByT5-based G2P covering 100+ languages
  - `MisakiTextProcessor`: English G2P using Misaki (BART fallback)
- **KittenTTS** — StyleTTS2-based English TTS (80M/40M/15M param variants)
  - Voices: Bella, Alex, and others; 24kHz output, non-autoregressive
- **Kokoro** — lightweight multilingual TTS (82M params, 24kHz)
  - 9 languages: en-US, en-GB, es, fr, hi, it, ja, pt, zh
  - 54 voices with automatic language detection from voice prefix
  - Backed by `KokoroMultilingualProcessor` + `MLXAudioG2P`

New TTSModel factory cases: `kitten_tts`, `kitten`, `kokoro`, `kokoro_tts`

**Effect on iOS devices:** Adds two new lightweight TTS options well-suited to on-device
inference. Kokoro-82M runs on iPhone 15 Pro+ and all M-chip devices. KittenTTS nano (15M)
targets devices with tighter memory budgets.
