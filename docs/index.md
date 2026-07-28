# Fast and faithful BPE

FFBPE trains exact byte-pair encoding tokenizers for large, multilingual
corpora. Its Python API handles the common workflow; the Rust crate exposes the
lower-level training, encoding, and streaming primitives.

```bash
pip install ffbpe
```

```python
from ffbpe import train_bpe

model = train_bpe(
  ["hello world", "hello tokenizer"],
  vocab_size=280,
  special_tokens=["<|endoftext|>"],
)

ids = model.encode("hello world")
assert model.decode(ids) == "hello world"
```

[Get started](getting-started.md){ .md-button .md-button--primary }
[Python API](api/index.md){ .md-button }
[Rust API](https://docs.rs/ffbpe){ .md-button }

## Why FFBPE?

### Exact at corpus scale

Bound persistent pair postings with an optional hot-pair window without
approximating global frequencies, winner selection, or deterministic
tie-breaking.

### Built for multilingual text

Shape Unicode-heavy inventories with measured bigram boundaries, and reserve
learned vocabulary for byte fallback inside rare Unicode scalars.

### Stream instead of materializing

Feed replayable Python iterables through bounded native batches. Merge counters
from independent partitions and transfer them directly into the trainer.

### Familiar formats and APIs

Use GPT-2 files for byte models, lossless `unitoken` files for Unicode models,
self-describing model directories, and an optional tiktoken-shaped interface.

## Where should I begin?

| Goal | Start here |
|---|---|
| Train from strings or an iterable | [Installation and quickstart](getting-started.md) |
| Control counting and training separately | [Train a tokenizer](guides/training.md) |
| Train on CJK or other Unicode-heavy corpora | [Multilingual corpora](guides/multilingual.md) |
| Save, reload, encode, and decode | [Encode and save models](guides/encoding.md) |
| Port code written for tiktoken | [tiktoken compatibility](guides/tiktoken.md) |
| Use native Rust primitives | [Rust API on docs.rs](https://docs.rs/ffbpe) |

!!! note "Package rename"

    FFBPE 0.1.8 is the first release under the `ffbpe` package name. Existing
    model directories and the serialized `unitoken` format remain compatible.
