import assert from "node:assert/strict"
import test from "node:test"

import {
  BigramCounter,
  BpeEncoder,
  BpeTrainer,
  FFBPE,
  PreTokenizer,
  WordCounter,
  trainBpe,
} from "../dist/node.js"

await FFBPE.init()

test("initialization is idempotent", async () => {
  await FFBPE.init()
  await FFBPE.init()

  assert.deepEqual(new PreTokenizer([]).getWords("hello"), { hello: 1 })
})

test("pretokenizer keeps special tokens indivisible and out of word counts", () => {
  const pretokenizer = new PreTokenizer(["<|endoftext|>"])

  assert.deepEqual(
    pretokenizer.getWords("hello<|endoftext|>hello"),
    { hello: 2 },
  )
  assert.deepEqual(
    pretokenizer.withUnicodeBigrams(["你好"]).getWords("你好世界"),
    { "世": 1, "你好": 1, "界": 1 },
  )
})

test("bigram counters add, batch, merge, and retain cutoff ties", () => {
  const pretokenizer = new PreTokenizer([])
  const left = new BigramCounter(pretokenizer)
  const right = pretokenizer.bigramCounter()

  left.addSource(["你好你好"], { max_records: 1, max_bytes: 12 })
  right.addBatch(["你好世界"])
  left.merge(right)

  assert.deepEqual(left.items(), [
    ["世界", 1],
    ["你好", 3],
    ["好世", 1],
    ["好你", 1],
  ])
  assert.deepEqual(left.selected(1, 1), ["你好"])
  assert.deepEqual(left.select(2, 1), {
    bigrams: ["世界", "你好", "好世", "好你"],
    cutoff_freq: 1,
    max_excluded_freq: null,
  })
})

test("word counters merge, serialize, clear, and report emptiness", () => {
  const pretokenizer = new PreTokenizer([], { pat_str: "[^\\s]" })
  const left = new WordCounter(pretokenizer)
  const right = pretokenizer.wordCounter()

  assert.equal(left.isEmpty, true)
  left.addBatch(["ab", "a"])
  right.addText("bc")
  left.merge(right)

  assert.equal(left.isEmpty, false)
  assert.equal(left.len, 3)
  assert.deepEqual(left.words(), { a: 2, b: 2, c: 1 })
  assert.deepEqual(
    pretokenizer.loadWordCounterData(left.serialize()).words(),
    left.words(),
  )

  left.clear()
  assert.equal(left.isEmpty, true)
  assert.equal(left.len, 0)
  assert.deepEqual(left.words(), {})
})

test("source batching validates limits and yielded values", () => {
  const counter = new PreTokenizer([]).wordCounter()

  assert.throws(
    () => counter.addSource(["hello"], { max_records: 0 }),
    /max_records must be positive/,
  )
  assert.throws(
    () => counter.addSource(["hello"], { max_bytes: 1.5 }),
    /max_bytes must be a safe integer/,
  )
  assert.throws(
    () => counter.addSource(["hello", 42]),
    /source must yield strings/,
  )
})

test("trainer consumes counters, steps deterministically, and exposes diagnostics", () => {
  const counter = new PreTokenizer([]).wordCounter()
  counter.addSource(["aba", "aba", "abc"])

  const trainer = new BpeTrainer([], {
    unit: "byte",
    hot_pair_window_size: 2,
  })
  trainer.addWordCounter(counter)

  assert.equal(counter.isEmpty, true)
  assert.equal(trainer.vocabSize, 256)
  assert.equal(trainer.lastMergeFreq, null)
  assert.equal(trainer.memoryUsage.vocab_entries, 256)

  trainer.initTraining()
  const size = trainer.step()

  assert.equal(size, 257)
  assert.equal(trainer.vocabSize, 257)
  assert.equal(trainer.lastMergeFreq, 3)
  assert.equal(trainer.hotPairWindowStats.resident_pairs <= 2, true)
  assert.equal(trainer.vocab.length, trainer.vocabSize)
})

test("models cache their default encoder and route special tokens", () => {
  const model = trainBpe(
    "hello<|endoftext|>world",
    {
      vocab_size: 270,
      special_tokens: ["<|endoftext|>"],
    },
  )
  const encoder = model.encoder()

  assert.equal(model.encoder(), encoder)
  assert.deepEqual(model.specialTokens, ["<|endoftext|>"])
  assert.deepEqual([...encoder.encode("<|endoftext|>")], [0])
  assert.equal(
    encoder.decode(encoder.encode("hello<|endoftext|>world")),
    "hello<|endoftext|>world",
  )
  assert.equal(encoder.preTokenizer().getWords("hello").hello, 1)
})

test("serialized models restore inferred special tokens and batch encoding", () => {
  const model = trainBpe(
    ["hello world", "hello tokenizer"],
    {
      vocab_size: 280,
      special_tokens: ["<|endoftext|>"],
    },
  )
  const files = model.toPretrainedFiles()
  const encoder = BpeEncoder.fromSerialized(
    files["vocab.json"],
    files["merges.txt"],
    { unit: "byte", format: "gpt2" },
  )
  const encoded = encoder.encodeWords(["hello", "world"])

  assert.deepEqual([...encoder.encode("<|endoftext|>")], [0])
  assert.equal(encoder.decode(encoded[0]), "hello")
  assert.equal(encoder.decode(encoded[1]), "world")
  assert.equal(JSON.parse(files["ffbpe.json"]).special_tokens[0], "<|endoftext|>")
})

test("public APIs reject incompatible formats and invalid token ids", () => {
  const unicode = trainBpe("你好世界", {
    unit: "unicode",
    vocab_size: 265,
  })

  assert.throws(
    () => unicode.serializeVocab("gpt2"),
    /not compatible/,
  )
  assert.throws(
    () => new BpeTrainer([], { hot_pair_window_size: 0 }),
    /hot_pair_window_size must be positive/,
  )
  assert.throws(
    () => unicode.decode([0xffff_ffff]),
    /Out of vocabulary idx/,
  )
})
