import assert from "node:assert/strict"
import { readFile } from "node:fs/promises"
import { fileURLToPath } from "node:url"
import test from "node:test"

const native_fetch = globalThis.fetch
globalThis.fetch = async input => {
  const url = input instanceof Request ? new URL(input.url) : new URL(input)
  if (url.protocol !== "file:") return native_fetch(input)
  const body = await readFile(fileURLToPath(url))
  const headers = url.pathname.endsWith(".wasm")
    ? { "Content-Type": "application/wasm" }
    : undefined
  return new Response(body, { headers })
}

const {
  BpeEncoder,
  FFBPE,
  PreTokenizer,
  trainBpe,
} = await import("../dist/browser.js")

await FFBPE.init()

test("browser file helpers accept Blob inputs", async () => {
  const corpus = new Blob(["ab a"])
  const pretokenizer = new PreTokenizer([], { pat_str: "[^\\s]" })

  assert.deepEqual(await pretokenizer.getWordsFromFile(corpus), { a: 2, b: 1 })

  const selection = await pretokenizer.selectUnicodeBigramsFromFile(
    new Blob(["你好你好"]),
    1,
    1,
  )
  assert.equal(selection.bigrams.length, 1)

  const model = trainBpe("hello browser", { vocab_size: 270 })
  const files = model.toPretrainedFiles()
  const encoder = await BpeEncoder.load(
    new Blob([files["vocab.json"]]),
    new Blob([files["merges.txt"]]),
  )
  const ids = await encoder.encodeFile(new Blob(["hello browser"]))
  assert.equal(encoder.decode(ids), "hello browser")
})

test("browser save output stays in memory", () => {
  const files = trainBpe("small model", { vocab_size: 268 }).toPretrainedFiles()

  assert.equal(JSON.parse(files["ffbpe.json"]).version, 1)
  assert.match(files["vocab.json"], /^\{/)
  assert.ok(files["merges.txt"].length > 0)
})
