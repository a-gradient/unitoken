import type { BpeEncoderOptions, FileFormat, ModelConfig, Unit } from "./types.js"

export function resolveFormat(unit: Unit, format?: FileFormat | null): FileFormat {
  const resolved_format = format ?? (unit === "unicode" ? "unitoken" : "gpt2")
  if (unit === "unicode" && resolved_format === "gpt2") {
    throw new Error('format="gpt2" is not compatible with unit="unicode"')
  }
  return resolved_format
}

export function validateModelConfig(value: unknown): ModelConfig {
  if (typeof value !== "object" || value === null) {
    throw new Error("Invalid FFBPE model config: expected an object")
  }
  const config = value as Partial<ModelConfig>
  if (config.version !== 1) {
    throw new Error(`Unsupported FFBPE model config version ${String(config.version)}; expected 1`)
  }
  if (config.unit !== "byte" && config.unit !== "unicode") {
    throw new Error(`Invalid FFBPE model unit ${String(config.unit)}`)
  }
  if (config.format !== "gpt2" && config.format !== "unitoken") {
    throw new Error(`Invalid FFBPE model format ${String(config.format)}`)
  }
  if (config.unit === "unicode" && config.format === "gpt2") {
    throw new Error('format="gpt2" is not compatible with unit="unicode"')
  }
  if (typeof config.vocab_file !== "string" || !isSafeRelativePath(config.vocab_file)) {
    throw new Error("Invalid FFBPE model config field 'vocab_file'")
  }
  if (typeof config.merges_file !== "string" || !isSafeRelativePath(config.merges_file)) {
    throw new Error("Invalid FFBPE model config field 'merges_file'")
  }
  if (!Array.isArray(config.special_tokens) || !config.special_tokens.every(value => typeof value === "string")) {
    throw new Error("Invalid FFBPE model config field 'special_tokens'")
  }
  if (config.pat_str !== null && config.pat_str !== undefined && typeof config.pat_str !== "string") {
    throw new Error("Invalid FFBPE model config field 'pat_str'")
  }
  if (config.unicode_bigrams !== null && config.unicode_bigrams !== undefined && (
    !Array.isArray(config.unicode_bigrams)
    || !config.unicode_bigrams.every(value => typeof value === "string")
  )) {
    throw new Error("Invalid FFBPE model config field 'unicode_bigrams'")
  }
  if (
    config.unicode_bigram_mixed_boundary !== "keep"
    && config.unicode_bigram_mixed_boundary !== "split"
  ) {
    throw new Error("Invalid FFBPE model config field 'unicode_bigram_mixed_boundary'")
  }
  if (config.split_on_vocab_bigrams !== undefined && typeof config.split_on_vocab_bigrams !== "boolean") {
    throw new Error("Invalid FFBPE model config field 'split_on_vocab_bigrams'")
  }
  return {
    version: 1,
    unit: config.unit,
    format: config.format,
    vocab_file: config.vocab_file,
    merges_file: config.merges_file,
    special_tokens: [...config.special_tokens],
    pat_str: config.pat_str ?? null,
    unicode_bigrams: config.unicode_bigrams === undefined ? null : config.unicode_bigrams,
    unicode_bigram_mixed_boundary: config.unicode_bigram_mixed_boundary,
    split_on_vocab_bigrams: config.split_on_vocab_bigrams ?? true,
  }
}

export function modelOptions(config: ModelConfig): BpeEncoderOptions {
  return {
    unit: config.unit,
    format: config.format,
    special_tokens: config.special_tokens,
    pat_str: config.pat_str,
    unicode_bigrams: config.unicode_bigrams,
    unicode_bigram_mixed_boundary: config.unicode_bigram_mixed_boundary,
    split_on_vocab_bigrams: config.split_on_vocab_bigrams,
  }
}

function isSafeRelativePath(path: string): boolean {
  if (path.length === 0 || path.startsWith("/") || path.startsWith("\\")) return false
  if (/^[A-Za-z]:[\\/]/.test(path)) return false
  return !path.split(/[\\/]/).includes("..")
}
