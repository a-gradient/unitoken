# Installation and quickstart

FFBPE requires Python 3.11 or newer.

```bash
pip install ffbpe
```

## Train from text

Use [`train_bpe()`][ffbpe.training.train_bpe] when the corpus fits a one-pass
workflow. The input may be one string or any iterable of independent text
records.

```python
from ffbpe import train_bpe

model = train_bpe(
  [
    "A tokenizer learns repeated byte sequences.",
    "The tokenizer can then encode new text.",
  ],
  vocab_size=320,
  special_tokens=["<|endoftext|>"],
)
```

Training can finish below the requested size when no eligible pair remains. A
target smaller than the initial vocabulary is rejected.

## Encode and decode

The validated model is immediately usable:

```python
text = "A tokenizer learns."
ids = model.encode(text)

assert model.decode(ids) == text
```

For repeated or lower-level encoding, build an encoder explicitly:

```python
encoder = model.encoder()
ids = encoder.encode("hello tokenizer")
array = encoder.encode_to_numpy("hello tokenizer")
```

## Save a self-describing model

```python
from ffbpe import BpeEncoder

model.save_pretrained("my-tokenizer")
encoder = BpeEncoder.from_pretrained("my-tokenizer")

assert encoder.decode(encoder.encode("hello")) == "hello"
```

The directory records the unit, file format, pretokenizer configuration, and
encoding optimization settings needed to reproduce the encoder.

## Choose the next guide

- Use [Train a tokenizer](guides/training.md) for explicit counters, partitions,
  or bounded-memory training.
- Use [Multilingual corpora](guides/multilingual.md) before enabling Unicode
  bigram shaping or byte fallback.
- Use [Encode and save models](guides/encoding.md) for file formats and loading.
