# @tokn-ai/ffbpe-inspect

Framework-neutral inspection data for visualizing FFBPE pretokenization and
tokenization in Node.js or a browser.

```ts
import { BpeEncoder, FFBPE } from "@tokn-ai/ffbpe"
import { inspect } from "@tokn-ai/ffbpe-inspect"

await FFBPE.init()

const encoder = await BpeEncoder.fromPretrained("./my-tokenizer")
const result = inspect(encoder, "hello, tokenizer!")

for (const pretoken of result.pretokens) {
  console.log(pretoken.text, pretoken.start_byte, pretoken.end_byte)
}

for (const token of result.tokens) {
  console.log(token.id, token.byte_hex, token.pretoken_index)
}
```

Offsets are UTF-8 byte offsets, matching the tokenizer's native model. A byte
token can contain an incomplete UTF-8 fragment; in that case its `text` is
`null` and `bytes`/`byte_hex` remain exact.
