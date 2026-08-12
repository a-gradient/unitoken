import assert from "node:assert/strict"
import test from "node:test"

import { FFBPE, trainBpe } from "@tokn-ai/ffbpe"

import { inspect } from "../dist/index.js"

await FFBPE.init()

test("inspection links ordered pretokens and tokens by byte range", () => {
  const encoder = trainBpe(
    "hello, tokenizer! hello, world!",
    { vocab_size: 270 },
  ).encoder()
  const result = inspect(encoder, "hello, world!")

  assert.equal(result.byte_count, 13)
  assert.equal(result.pretoken_count, result.pretokens.length)
  assert.equal(result.token_count, result.tokens.length)
  assert.equal(result.pretokens.map(pretoken => pretoken.text).join(""), result.text)
  assert.deepEqual([...encoder.encode(result.text)], result.tokens.map(token => token.id))

  for (const [pretoken_index, pretoken] of result.pretokens.entries()) {
    const tokens = result.tokens.slice(pretoken.token_start, pretoken.token_end)
    assert.equal(tokens.every(token => token.pretoken_index === pretoken_index), true)
    assert.equal(tokens.at(0)?.start_byte, pretoken.start_byte)
    assert.equal(tokens.at(-1)?.end_byte, pretoken.end_byte)
  }
})

test("inspection preserves special tokens and UTF-8 byte offsets", () => {
  const special_token = "<|endoftext|>"
  const encoder = trainBpe(
    `你好${special_token}世界`,
    {
      vocab_size: 275,
      special_tokens: [special_token],
    },
  ).encoder()
  const result = inspect(encoder, `你${special_token}好`)
  const special = result.pretokens.find(pretoken => pretoken.kind === "special")

  assert.equal(result.byte_count, 19)
  assert.equal(special?.text, special_token)
  assert.equal(special?.start_byte, 3)
  assert.equal(special?.end_byte, 16)
  assert.deepEqual(
    result.tokens.flatMap(token => token.bytes),
    [...new TextEncoder().encode(result.text)],
  )
})

test("inspection handles empty input", () => {
  const encoder = trainBpe("seed text", { vocab_size: 260 }).encoder()

  assert.deepEqual(inspect(encoder, ""), {
    text: "",
    byte_count: 0,
    token_count: 0,
    pretoken_count: 0,
    pretokens: [],
    tokens: [],
  })
})
