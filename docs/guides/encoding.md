# Encode and save models

[`BpeModel`][ffbpe.model.BpeModel] is an immutable validated training result.
It can encode directly, create reusable encoders, or write model files.

## Direct use

```python
ids = model.encode("hello world")
text = model.decode(ids)
```

`model.encoder()` caches its default encoder. Pass explicit options when the
saved model needs a custom regex, retained Unicode bigrams, or a different
vocabulary-bigram splitting policy.

## Self-describing directories

Prefer a self-describing directory for application use:

```python
from ffbpe import BpeEncoder

model.save_pretrained(
  "my-tokenizer",
  split_on_vocab_bigrams=True,
)

encoder = BpeEncoder.from_pretrained("my-tokenizer")
```

The directory contains `vocab.json`, `merges.txt`, and `ffbpe.json`. FFBPE also
reads the legacy `unitoken.json` metadata name.

## Raw model files

Byte models default to GPT-2 serialization. Unicode models use the lossless
`unitoken` format.

```python
model.save_files(
  "vocab.json",
  "merges.txt",
  format="gpt2",
)

encoder = BpeEncoder.load(
  unit="byte",
  format="gpt2",
  vocab_file="vocab.json",
  merges_file="merges.txt",
)
```

The GPT-2 format cannot represent Unicode-unit models.

## Vocabulary-bigram splitting

Encoders can partition long pretokenized words using bigrams already present in
the model vocabulary. This changes the amount of work, not token IDs.

```python
encoder = model.encoder(split_on_vocab_bigrams=False)
```

Disable the optimization only when profiling representative byte-model
workloads shows that its extra scan is unprofitable. Save the same setting with
the model to keep behavior consistent after loading.
