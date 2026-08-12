import { BpeEncoder, type TiktokenSpecialToken } from "@tokn-ai/ffbpe"

const R50K_PATTERN = String.raw`'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s`
const CL100K_PATTERN = String.raw`'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s`
const O200K_PATTERN = [
  String.raw`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
  String.raw`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
  String.raw`\p{N}{1,3}`,
  String.raw` ?[^\s\p{L}\p{N}]+[\r\n/]*`,
  String.raw`\s*[\r\n]+`,
  String.raw`\s+(?!\S)`,
  String.raw`\s+`,
].join("|")

const ENDOFTEXT = "<|endoftext|>"
const FIM_PREFIX = "<|fim_prefix|>"
const FIM_MIDDLE = "<|fim_middle|>"
const FIM_SUFFIX = "<|fim_suffix|>"
const ENDOFPROMPT = "<|endofprompt|>"

export type TokenizerPreset =
  | "gpt2"
  | "r50k_base"
  | "p50k_base"
  | "cl100k_base"
  | "o200k_base"

export type PatternFamily = "r50k" | "cl100k" | "o200k"

export interface PresetSpecialToken extends TiktokenSpecialToken {}

export interface TokenizerPresetDefinition {
  name: TokenizerPreset
  display_name: string
  description: string
  pattern_family: PatternFamily
  model_url: string
  model_sha256: string
  vocab_size: number
  mergeable_token_count: number
  special_tokens: readonly PresetSpecialToken[]
}

export interface LoadPresetOptions {
  fetch?: typeof globalThis.fetch
  model_url?: string | URL
  verify_hash?: boolean
}

const R50K_MODEL_URL = "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken"
const R50K_MODEL_SHA256 = "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930"

function specialToken(text: string, id: number): PresetSpecialToken {
  return Object.freeze({ text, id })
}

function definePreset(
  preset: Omit<TokenizerPresetDefinition, "special_tokens"> & {
    special_tokens: PresetSpecialToken[]
  },
): TokenizerPresetDefinition {
  return Object.freeze({
    ...preset,
    special_tokens: Object.freeze(preset.special_tokens),
  })
}

const PRESETS_BY_NAME: Readonly<Record<TokenizerPreset, TokenizerPresetDefinition>> = Object.freeze({
  gpt2: definePreset({
    name: "gpt2",
    display_name: "GPT-2",
    description: "The original 50k GPT-2 byte-pair encoding.",
    pattern_family: "r50k",
    model_url: R50K_MODEL_URL,
    model_sha256: R50K_MODEL_SHA256,
    vocab_size: 50_257,
    mergeable_token_count: 50_256,
    special_tokens: [specialToken(ENDOFTEXT, 50_256)],
  }),
  r50k_base: definePreset({
    name: "r50k_base",
    display_name: "r50k_base",
    description: "The GPT-3 and Codex-era 50k base encoding.",
    pattern_family: "r50k",
    model_url: R50K_MODEL_URL,
    model_sha256: R50K_MODEL_SHA256,
    vocab_size: 50_257,
    mergeable_token_count: 50_256,
    special_tokens: [specialToken(ENDOFTEXT, 50_256)],
  }),
  p50k_base: definePreset({
    name: "p50k_base",
    display_name: "p50k_base",
    description: "A 50k encoding tuned for code and natural language.",
    pattern_family: "r50k",
    model_url: "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
    model_sha256: "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
    vocab_size: 50_281,
    mergeable_token_count: 50_280,
    special_tokens: [specialToken(ENDOFTEXT, 50_256)],
  }),
  cl100k_base: definePreset({
    name: "cl100k_base",
    display_name: "cl100k_base",
    description: "The 100k encoding used by GPT-4 and text embeddings.",
    pattern_family: "cl100k",
    model_url: "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    model_sha256: "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    vocab_size: 100_277,
    mergeable_token_count: 100_256,
    special_tokens: [
      specialToken(ENDOFTEXT, 100_257),
      specialToken(FIM_PREFIX, 100_258),
      specialToken(FIM_MIDDLE, 100_259),
      specialToken(FIM_SUFFIX, 100_260),
      specialToken(ENDOFPROMPT, 100_276),
    ],
  }),
  o200k_base: definePreset({
    name: "o200k_base",
    display_name: "o200k_base",
    description: "The multilingual 200k encoding introduced with GPT-4o.",
    pattern_family: "o200k",
    model_url: "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    model_sha256: "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    vocab_size: 200_019,
    mergeable_token_count: 199_998,
    special_tokens: [
      specialToken(ENDOFTEXT, 199_999),
      specialToken(ENDOFPROMPT, 200_018),
    ],
  }),
})

export const TOKENIZER_PRESETS: readonly TokenizerPresetDefinition[] = Object.freeze([
  PRESETS_BY_NAME.gpt2,
  PRESETS_BY_NAME.r50k_base,
  PRESETS_BY_NAME.p50k_base,
  PRESETS_BY_NAME.cl100k_base,
  PRESETS_BY_NAME.o200k_base,
])

function presetPattern(family: PatternFamily): string {
  switch (family) {
    case "r50k": return R50K_PATTERN
    case "cl100k": return CL100K_PATTERN
    case "o200k": return O200K_PATTERN
  }
}

async function sha256(data: Uint8Array): Promise<string> {
  if (globalThis.crypto?.subtle === undefined) {
    throw new Error("SHA-256 verification requires the Web Crypto API")
  }
  const copy = new Uint8Array(data.byteLength)
  copy.set(data)
  const digest = await globalThis.crypto.subtle.digest("SHA-256", copy.buffer)
  return [...new Uint8Array(digest)]
    .map(byte => byte.toString(16).padStart(2, "0"))
    .join("")
}

export function getPreset(name: TokenizerPreset): TokenizerPresetDefinition {
  const preset = PRESETS_BY_NAME[name]
  if (preset === undefined) throw new RangeError(`Unknown tokenizer preset: ${String(name)}`)
  return preset
}

/** Build a preset encoder from an already-loaded `.tiktoken` asset. */
export function createPresetEncoder(
  name: TokenizerPreset,
  model_data: Uint8Array,
): BpeEncoder {
  const preset = getPreset(name)
  return BpeEncoder.fromTiktoken(
    new TextDecoder().decode(model_data),
    preset.special_tokens,
    {
      unit: "byte",
      format: "gpt2",
      pat_str: presetPattern(preset.pattern_family),
    },
  )
}

export async function loadPreset(
  name: TokenizerPreset,
  options: LoadPresetOptions = {},
): Promise<BpeEncoder> {
  const preset = getPreset(name)
  const fetch_model = options.fetch ?? globalThis.fetch
  if (fetch_model === undefined) throw new Error("Loading a tokenizer preset requires fetch")
  const model_url = options.model_url ?? preset.model_url
  const response = await fetch_model(model_url)
  if (!response.ok) {
    throw new Error(`Cannot download ${name} from ${String(model_url)}: ${response.status} ${response.statusText}`)
  }
  const model_data = new Uint8Array(await response.arrayBuffer())
  if (options.verify_hash ?? true) {
    const actual_hash = await sha256(model_data)
    if (actual_hash !== preset.model_sha256) {
      throw new Error(`SHA-256 mismatch for ${name}: expected ${preset.model_sha256}, received ${actual_hash}`)
    }
  }
  return createPresetEncoder(name, model_data)
}
