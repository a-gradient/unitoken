import assert from "node:assert/strict"
import { mkdtemp, readFile, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath, pathToFileURL } from "node:url"
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
  WordCounter,
  trainBpe,
} = await import("../dist/browser.js")

await FFBPE.init()

test("browser initialization is idempotent", async () => {
  await FFBPE.init()
  assert.deepEqual(new PreTokenizer([]).getWords("browser"), { browser: 1 })
})

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

test("browser pretrained loading fetches a hosted model directory", async () => {
  const files = trainBpe("hosted browser model", { vocab_size: 275 }).toPretrainedFiles()
  const directory = await mkdtemp(join(tmpdir(), "ffbpe-browser-model-"))
  await Promise.all(Object.entries(files).map(([file_name, content]) => (
    writeFile(join(directory, file_name), content, "utf8")
  )))

  const base_url = pathToFileURL(`${directory}/`)
  const encoder = await BpeEncoder.fromPretrained(base_url)
  const ids = encoder.encode("hosted browser model")

  assert.equal(encoder.decode(ids), "hosted browser model")
})

test("browser file APIs reject paths and filesystem writes", async () => {
  const pretokenizer = new PreTokenizer([])
  const counter = new WordCounter(pretokenizer)
  const model = trainBpe("browser only", { vocab_size: 268 })

  await assert.rejects(
    pretokenizer.getWordsFromFile("corpus.txt"),
    /expect a File or Blob/,
  )
  await assert.rejects(
    counter.save("words.json"),
    /only available in the Node runtime/,
  )
  await assert.rejects(
    model.savePretrained("model"),
    /only available in the Node runtime/,
  )
  await assert.rejects(
    model.saveVocabJson("vocab.json"),
    /only available in the Node runtime/,
  )
})
