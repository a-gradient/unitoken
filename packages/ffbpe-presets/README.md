# @tokn-ai/ffbpe-presets

Lazy-loaded common tokenizer presets for `@tokn-ai/ffbpe`.

```ts
import { FFBPE } from "@tokn-ai/ffbpe"
import { loadPreset } from "@tokn-ai/ffbpe-presets"

await FFBPE.init()

const encoder = await loadPreset("cl100k_base")
console.log([...encoder.encode("hello tokenizer")])
```

The package includes metadata—not model weights—for `gpt2`, `r50k_base`,
`p50k_base`, `cl100k_base`, and `o200k_base`. `loadPreset` downloads only the
selected official OpenAI `.tiktoken` asset, verifies its SHA-256 digest, and
converts it into an in-memory FFBPE encoder. Text sent to `encode` remains local.

Use `TOKENIZER_PRESETS` to build a selector or inspect download metadata:

```ts
import { TOKENIZER_PRESETS } from "@tokn-ai/ffbpe-presets"

for (const preset of TOKENIZER_PRESETS) {
  console.log(preset.name, preset.vocab_size, preset.model_url)
}
```

Node.js 20 and newer provide the required `fetch` and Web Crypto APIs. Browsers
must allow requests to the model host. To self-host an unchanged asset, pass
`model_url`; hash verification remains enabled by default:

```ts
const encoder = await loadPreset("o200k_base", {
  model_url: new URL("/models/o200k_base.tiktoken", location.href),
})
```

If your application loads bytes through its own cache or asset pipeline, use
`createPresetEncoder(name, model_data)` instead.

Special-token strings are recognized as special tokens by FFBPE. This differs
from tiktoken's default safety policy, which requires callers to explicitly
allow special tokens during each encode call.
