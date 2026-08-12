# @tokn-ai/ffbpe

FFBPE's Rust tokenizer and trainer for JavaScript, compiled to WebAssembly.

The API follows the Python package's in-memory classes. Filesystem operations
are asynchronous and use separate Node and browser adapters.

## Node

```ts
import { BpeEncoder, FFBPE } from "@tokn-ai/ffbpe"

await FFBPE.init()

const encoder = await BpeEncoder.fromPretrained("./my-tokenizer")
const ids = encoder.encode("hello world")
console.log(encoder.decode(ids))
```

Node resolves a string passed to `fromPretrained` as a directory containing
`ffbpe.json`, `vocab.json`, and `merges.txt`. File methods such as
`getWordsFromFile`, `BpeEncoder.load`, `encodeFile`, and `WordCounter.save`
accept filesystem paths or `file:` URLs. `BpeModel.savePretrained` writes a
model directory.

## Browser

```ts
import { BpeEncoder, FFBPE } from "@tokn-ai/ffbpe/browser"

await FFBPE.init()

const encoder = await BpeEncoder.fromPretrained(
  new URL("./models/my-tokenizer/", location.href),
)
const ids = encoder.encode("你好, tokenizer!")
```

Browsers fetch pretrained files relative to the supplied URL. File methods
accept a `File` or `Blob`; they never receive local path strings. Browser model
saving returns an in-memory file set:

```ts
const files = model.toPretrainedFiles()
```

You can also load files selected by the user without uploading them:

```ts
const encoder = await BpeEncoder.load(vocabFile, mergesFile)
const ids = await encoder.encodeFile(textFile)
```

The browser adapter currently reads each `File`/`Blob` into memory. Native
Python file chunking and segment offsets are intentionally not simulated in
the first WASM package.

## Train and encode

```ts
import { FFBPE, trainBpe } from "@tokn-ai/ffbpe"

await FFBPE.init()

const model = trainBpe(
  ["hello world", "hello tokenizer"],
  {
    vocab_size: 280,
    special_tokens: ["<|endoftext|>"],
  },
)

const ids = model.encode("hello world")
if (model.decode(ids) !== "hello world") throw new Error("round trip failed")
```

Pure tokenizer operations are synchronous after `FFBPE.init()`. Operations
which read or write files return promises.

The initial package includes `PreTokenizer`, `BigramCounter`, `WordCounter`,
`BpeTrainer`, `BpeModel`, `BpeEncoder`, and `trainBpe`. Python APIs tied to
NumPy or native streaming iterators are not exposed in the browser package.
`PreTokenizer.split` returns ordered logical pretokens with UTF-8 byte offsets,
and `BpeEncoder.tokenBytes` returns the exact bytes for one vocabulary id. These
low-level methods support companion tooling such as `@tokn-ai/ffbpe-inspect`.

## Build from source

Publishing requires Rust, the `wasm32-unknown-unknown` target, `wasm-pack`, and
pnpm:

```sh
rustup target add wasm32-unknown-unknown
pnpm install
pnpm build
pnpm test
```
