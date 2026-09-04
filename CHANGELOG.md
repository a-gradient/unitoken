# Changelog

## 0.1.10 — 2026-09-04

### Changed

- Accelerated the built-in GPT-2, r50k, cl100k, and o200k PAT pretokenizers
  with portable SWAR scanning and a faster Unicode path.
- Added the optional `simd` Cargo feature for ASCII-led PAT inputs: NEON on
  AArch64 and runtime-detected AVX2 on x86_64. Unsupported CPUs and
  Unicode-led inputs retain scalar scanning and identical token boundaries.
- Batched trusted SIMD-derived PAT boundaries before invoking downstream
  callbacks, reducing callback overhead while preserving ordered, fallible
  emission.
- Expanded reproducible PAT benchmark reporting to cover scalar and optional
  SIMD runs, including backend availability.

## 0.1.9 — 2026-08-13

### Added

- Publishable `@tokn-ai/ffbpe` WebAssembly package with Node.js and browser
  adapters that mirror the in-memory Python API.
- `@tokn-ai/ffbpe-inspect` for framework-neutral pretoken and token inspection.
- `@tokn-ai/ffbpe-presets` with verified, lazy-loaded definitions for common
  tiktoken encodings.
- Interactive browser inspector for comparing pretokenizer boundaries with BPE
  output.

### Changed

- Specialized common pretokenizer patterns and extracted their scanners into
  the reusable `ffbpe-pat` crate.
- Added low-level pretoken spans and token-byte access used by inspection tools.

### Fixed

- Preserve browser runtime registration when JavaScript bundlers tree-shake the
  environment-specific package entry.
- Ensure the static inspector and preset loader use one configured FFBPE runtime.

## 0.1.8 — 2026-07-27

- First release under the FFBPE name, replacing `unitoken` while retaining
  compatibility with existing model directories and serialized formats.
