import type { BpeEncoder, PreTokenKind } from "@tokn-ai/ffbpe"

export interface InspectedPreToken {
  kind: PreTokenKind
  text: string
  start_byte: number
  end_byte: number
  token_start: number
  token_end: number
}

export interface InspectedToken {
  id: number
  bytes: number[]
  byte_hex: string
  text: string | null
  start_byte: number
  end_byte: number
  pretoken_index: number
}

export interface Inspection {
  text: string
  byte_count: number
  token_count: number
  pretoken_count: number
  pretokens: InspectedPreToken[]
  tokens: InspectedToken[]
}

const text_encoder = new TextEncoder()
const strict_text_decoder = new TextDecoder("utf-8", { fatal: true })

function decodeToken(bytes: Uint8Array): string | null {
  try {
    return strict_text_decoder.decode(bytes)
  } catch {
    return null
  }
}

function toHex(bytes: Uint8Array): string {
  return [...bytes].map(byte => byte.toString(16).padStart(2, "0")).join(" ")
}

/** Inspect the complete pretokenizer and BPE output for one string. */
export function inspect(encoder: BpeEncoder, text: string): Inspection {
  const pretokens: InspectedPreToken[] = []
  const tokens: InspectedToken[] = []

  for (const [pretoken_index, pretoken] of encoder.preTokenizer().split(text).entries()) {
    const ids = pretoken.kind === "special"
      ? encoder.encode(pretoken.text)
      : encoder.encodeWord(pretoken.text)
    const token_start = tokens.length
    let start_byte = pretoken.start_byte

    for (const id of ids) {
      const token_bytes = encoder.tokenBytes(id)
      const end_byte = start_byte + token_bytes.byteLength
      tokens.push({
        id,
        bytes: [...token_bytes],
        byte_hex: toHex(token_bytes),
        text: decodeToken(token_bytes),
        start_byte,
        end_byte,
        pretoken_index,
      })
      start_byte = end_byte
    }

    if (start_byte !== pretoken.end_byte) {
      throw new Error(
        `Tokenizer output covers ${start_byte - pretoken.start_byte} bytes for a ${pretoken.end_byte - pretoken.start_byte}-byte pretoken`,
      )
    }
    pretokens.push({
      ...pretoken,
      token_start,
      token_end: tokens.length,
    })
  }

  return {
    text,
    byte_count: text_encoder.encode(text).byteLength,
    token_count: tokens.length,
    pretoken_count: pretokens.length,
    pretokens,
    tokens,
  }
}
