import assert from "node:assert/strict"
import test from "node:test"

import { FFBPE } from "@tokn-ai/ffbpe"

import {
  TOKENIZER_PRESETS,
  createPresetEncoder,
  getPreset,
  loadPreset,
} from "../dist/index.js"

await FFBPE.init()

function tinyTiktokenModel() {
  const entries = Array.from({ length: 256 }, (_, rank) => [Uint8Array.of(rank), rank])
  entries.push(
    [new TextEncoder().encode("ab"), 256],
    [new TextEncoder().encode("abc"), 257],
    [new TextEncoder().encode(" abc"), 258],
  )
  return entries
    .map(([bytes, rank]) => `${Buffer.from(bytes).toString("base64")} ${rank}`)
    .join("\n")
}

function modelResponse(contents, status = 200) {
  return new Response(contents, {
    status,
    statusText: status === 200 ? "OK" : "Not Found",
  })
}

test("registry exposes the five common tokenizer presets", () => {
  assert.deepEqual(
    TOKENIZER_PRESETS.map(preset => preset.name),
    ["gpt2", "r50k_base", "p50k_base", "cl100k_base", "o200k_base"],
  )
  assert.equal(getPreset("cl100k_base").vocab_size, 100_277)
  assert.equal(getPreset("o200k_base").pattern_family, "o200k")
  assert.throws(() => getPreset("missing"), /Unknown tokenizer preset/)
})

test("loader converts ranked tiktoken tokens into exact FFBPE merges", async () => {
  const encoder = await loadPreset("gpt2", {
    fetch: async () => modelResponse(tinyTiktokenModel()),
    verify_hash: false,
  })

  assert.deepEqual([...encoder.encode("abc abc")], [257, 258])
  assert.equal(encoder.decode([257, 258]), "abc abc")
  assert.deepEqual([...encoder.encode("<|endoftext|>")], [50_256])
})

test("in-memory model APIs support self-hosted assets", () => {
  const data = new TextEncoder().encode(tinyTiktokenModel())
  const encoder = createPresetEncoder("r50k_base", data)
  assert.deepEqual([...encoder.encode("abc")], [257])
})

test("loader verifies model hashes by default", async () => {
  await assert.rejects(
    loadPreset("gpt2", {
      fetch: async () => modelResponse(tinyTiktokenModel()),
    }),
    /SHA-256 mismatch for gpt2/,
  )
})

test("loader reports download failures with preset context", async () => {
  await assert.rejects(
    loadPreset("cl100k_base", {
      fetch: async () => modelResponse("missing", 404),
    }),
    /Cannot download cl100k_base.*404 Not Found/,
  )
})
