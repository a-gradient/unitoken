import assert from "node:assert/strict"
import { mkdtemp, readFile, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import test from "node:test"

import {
  BpeEncoder,
  BpeTrainer,
  FFBPE,
  PreTokenizer,
  WordCounter,
  trainBpe,
} from "../dist/node.js"

await FFBPE.init()

test("pretokenizer and counters match the Python-style API", () => {
  const pretokenizer = new PreTokenizer([], { pat_str: "[^\\s]" })
  assert.deepEqual(pretokenizer.getWords("ab a"), { a: 2, b: 1 })

  const counter = pretokenizer.wordCounter()
  counter.addBatch(["ab", "a"])
  assert.equal(counter.len, 2)
  assert.deepEqual(counter.words(), { a: 2, b: 1 })

  const restored = pretokenizer.loadWordCounterData(counter.serialize())
  assert.deepEqual(restored.words(), { a: 2, b: 1 })
})

test("Node file helpers accept paths and preserve counters", async () => {
  const directory = await mkdtemp(join(tmpdir(), "ffbpe-files-"))
  const text_file = join(directory, "corpus.txt")
  const counter_file = join(directory, "words.json")
  await writeFile(text_file, "ab a", "utf8")

  const pretokenizer = new PreTokenizer([], { pat_str: "[^\\s]" })
  assert.deepEqual(await pretokenizer.getWordsFromFile(text_file), { a: 2, b: 1 })

  const counter = new WordCounter(pretokenizer)
  counter.addSource(["ab", "a"], { max_records: 1, max_bytes: 2 })
  await counter.save(counter_file)
  assert.deepEqual((await pretokenizer.loadWordCounter(counter_file)).words(), { a: 2, b: 1 })
})

test("serialized data produces deterministic merge ids", () => {
  const vocabulary = Array.from({ length: 256 }, (_, id) => [Uint8Array.of(id), id])
  vocabulary.push([Uint8Array.of(97, 98), 256])
  const encoder = BpeEncoder.fromData(
    vocabulary,
    [[Uint8Array.of(97), Uint8Array.of(98)]],
    { unit: "byte", format: "gpt2" },
  )

  assert.deepEqual([...encoder.encode("ab")], [256])
  assert.deepEqual([...encoder.encodeWord("ab")], [256])
  assert.equal(encoder.decode([256]), "ab")
})

test("training and encoding round trip", () => {
  const model = trainBpe(
    ["hello world", "hello tokenizer"],
    { vocab_size: 280, special_tokens: ["<|endoftext|>"] },
  )
  const ids = model.encode("hello world")
  assert.equal(model.decode(ids), "hello world")
  assert.equal(model.unit, "byte")
})

test("Unicode training and serialization round trip", () => {
  const model = trainBpe(
    ["你好世界", "你好 tokenizer"],
    { unit: "unicode", vocab_size: 270 },
  )
  const files = model.toPretrainedFiles()
  const encoder = BpeEncoder.fromSerialized(files["vocab.json"], files["merges.txt"], {
    unit: "unicode",
    format: "unitoken",
  })

  assert.equal(encoder.decode(encoder.encode("你好世界")), "你好世界")
  assert.equal(JSON.parse(files["ffbpe.json"]).unit, "unicode")
})

test("numeric arguments reject lossy JavaScript values", () => {
  const trainer = new BpeTrainer([], { unit: "byte" })
  assert.throws(() => trainer.addWords({ hello: 1.5 }), /safe integer/)
  assert.throws(() => trainer.train(-1), /between 0/)

  const counter = new PreTokenizer([]).bigramCounter()
  assert.throws(() => counter.select(1.5, 1), /safe integer/)
})

test("pretrained Node directory round trip", async () => {
  const trainer = new BpeTrainer(["<|endoftext|>"], { unit: "byte" })
  trainer.addWords({ hello: 3 })
  trainer.train(262)
  const model = trainer.validateModel()
  const directory = await mkdtemp(join(tmpdir(), "ffbpe-wasm-"))

  await model.savePretrained(directory)
  await model.saveFiles(join(directory, "explicit-vocab.json"), join(directory, "explicit-merges.txt"))
  const encoder = await BpeEncoder.fromPretrained(directory)
  const ids = encoder.encode("hello")

  assert.equal(encoder.decode(ids), "hello")
  assert.equal(JSON.parse(await readFile(join(directory, "ffbpe.json"), "utf8")).version, 1)
  assert.match(await readFile(join(directory, "explicit-vocab.json"), "utf8"), /^\{/)
})
