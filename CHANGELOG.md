# Changelog

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
