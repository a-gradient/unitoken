# tiktoken compatibility

FFBPE includes a tiktoken-shaped [`Encoding`][ffbpe.tiktoken_compat.Encoding]
for code that expects common encoding and decoding methods.

```python
from ffbpe import Encoding

encoding = Encoding.from_files(
  "my-model",
  vocab_file="vocab.json",
  merges_file="merges.txt",
  special_tokens={"<|endoftext|>": 0},
)

ids = encoding.encode(
  "hello",
  disallowed_special=(),
)
assert encoding.decode(ids) == "hello"
```

The compatibility layer supports ordinary and batch encoding, special-token
checks, byte decoding, and local model files. It does not bundle OpenAI model
vocabularies or a network registry.

Use [`BpeEncoder`][ffbpe.encoder.BpeEncoder] directly for new FFBPE code unless
an existing integration benefits from the tiktoken-shaped surface.
