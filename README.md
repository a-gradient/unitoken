# FFBPE

[![CI](https://github.com/tokn-ai/ffbpe/actions/workflows/ci.yml/badge.svg)](https://github.com/tokn-ai/ffbpe/actions/workflows/ci.yml)
[![Website](https://img.shields.io/badge/web-tokn--ai.dev%2Fffbpe-ff5c35.svg)](https://tokn-ai.dev/ffbpe/)
[![Documentation](https://img.shields.io/badge/docs-tokn--ai.dev%2Fffbpe%2Fdocs-blue.svg)](https://tokn-ai.dev/ffbpe/docs/)
[![PyPI](https://img.shields.io/pypi/v/ffbpe.svg)](https://pypi.org/project/ffbpe/)
[![crates.io](https://img.shields.io/crates/v/ffbpe.svg)](https://crates.io/crates/ffbpe)
[![docs.rs](https://docs.rs/ffbpe/badge.svg)](https://docs.rs/ffbpe)

**FFBPE is fast and faithful byte-pair encoding for large, multilingual corpora.**

Exact BPE at corpus scale.

It combines Unicode-aware inventory shaping, frequency-safe merge cutoffs,
exact bounded-memory training, and tiktoken-compatible encoding.

Python is the easiest way to train and use a tokenizer. Rust exposes the lower-level
training, encoding, and streaming primitives.

## Why FFBPE

- **Shape Unicode-heavy inventories before training.** A two-pass Unicode-bigram
  pipeline retains frequent adjacent pairs and splits unproductive boundaries,
  reducing the nearly unique word inventories common in CJK corpora.
- **Carry measured boundaries into training.** Bigram selection reports its inclusive
  frequency cutoff, and BPE training can stop before learning a pair below that
  boundary. Selection includes all ties at the cutoff.
- **Keep rare Unicode scalars encodable without bloating the alphabet.** Unicode
  training can reserve part of its learned vocabulary for byte-level fallback merges
  inside scalars that were not materialized directly.
- **Accelerate long-word encoding when the vocabulary supports it.** Encoders can
  partition PAT words using bigrams already present in the model vocabulary, with an
  explicit opt-out for workloads where the extra scan is not profitable.
- **Bound memory without approximating the model.** The optional hot-pair window
  bounds persistent occurrence postings while preserving global frequencies,
  winner selection, and deterministic tie-breaking.
- **Stream corpora through bounded native batches.** Replayable sources, prefetch,
  mergeable counters, and native counter-to-trainer transfer avoid materializing a
  complete Python dictionary.
- **Use familiar model formats and APIs.** Byte models support GPT-2 serialization,
  Unicode models use the lossless legacy `unitoken` format, and Python includes a
  tiktoken-shaped API.

## Measured impact

One release run on the 64 MiB FineWeb2 Chinese fixture, training a 10,000-token
vocabulary, measured:

| Pipeline | Unique words | BPE training |
|---|---:|---:|
| Regular Unicode inventory | 1,803,009 | 26.681 s |
| Retained Unicode bigrams | 606,153 | 3.702 s |

For this workload, Unicode-bigram shaping produced an approximately 3× smaller
inventory and 7× faster BPE training. It changes corpus segmentation and should be
benchmarked on representative text rather than treated as a universal speedup.

On a 1 GiB FineWeb2 Chinese Unicode-bigram inventory, the exact bounded-memory mode
reduced observed training peak RSS from 1,797 MiB to 1,649 MiB while producing the
same model; training changed from 5.58 s to 5.85 s and required two hydration scans.

See [BENCHMARKS.md](BENCHMARKS.md) for benchmark contracts, qualifications, and
reproduction commands.

## Install

Python 3.11 or newer:

```bash
pip install ffbpe
```

Rust:

```bash
cargo add ffbpe
```

### Renamed from unitoken

FFBPE 0.1.8 is the first release under the new package name. Install and import
`ffbpe` for new projects. Existing model directories remain compatible:

- FFBPE writes model metadata to `ffbpe.json`.
- FFBPE also reads legacy `unitoken.json` metadata.
- The serialized `unitoken` model format name remains stable, so existing vocab
  and merge files do not need conversion.

The old packages will receive a separate deprecation-only release after FFBPE is
available.

## Five-minute Python quickstart

Train directly from strings, encode and decode without an intermediate file round
trip, then save a self-describing model directory:

```python
from ffbpe import BpeEncoder, train_bpe

model = train_bpe(
  ["hello world", "hello tokenizer"],
  vocab_size=280,
  special_tokens=["<|endoftext|>"],
)

ids = model.encode("hello world")
assert model.decode(ids) == "hello world"

model.save_pretrained("my-tokenizer")
encoder = BpeEncoder.from_pretrained("my-tokenizer")
assert encoder.decode(encoder.encode("hello world")) == "hello world"
```

`train_bpe` accepts a single string or a one-pass iterable of independent text
records. Counting is batched in Rust. Training can finish below the requested size
when the corpus has no eligible pairs left. A target below the initial vocabulary
size is rejected.

Use `BpeTrainer` and `PreTokenizer` directly when you need compressed word counts,
Unicode-bigram selection, a custom regex, or manual merge steps.

Encoding partitions long PAT words using model-vocabulary bigrams by default. This
does not change token ids. If profiling shows that the scan is slower for your byte
model, disable it consistently when creating or saving the encoder:

```python
encoder = model.encoder(split_on_vocab_bigrams=False)
model.save_pretrained("my-tokenizer", split_on_vocab_bigrams=False)
```

## Unicode-bigram training with a safe cutoff

Unicode-bigram shaping is an explicit two-pass workflow:

1. Count adjacent Unicode pairs.
2. Retain the most frequent pairs, including every tie at the boundary.
3. Count words using the retained pairs.
4. Carry the measured cutoff into BPE training.

```python
from ffbpe import BpeTrainer, PreTokenizer


class Corpus:
  def scan(self):
    yield "你好世界"
    yield "你好，tokenizer"


corpus = Corpus()
pretokenizer = PreTokenizer([])

bigram_counter = pretokenizer.bigram_counter()
bigram_counter.add_source(corpus.scan())
selection = bigram_counter.select(top_k=100_000, min_freq=2)

word_counter = (
  pretokenizer
  .with_unicode_bigrams(selection.bigrams)
  .word_counter()
)
word_counter.add_source(corpus.scan())

trainer = BpeTrainer(
  [],
  unit="unicode",
  bigram_cutoff_freq=selection.cutoff_freq,
)
trainer.add_word_counter(word_counter)
trainer.train(vocab_size=10_000)
model = trainer.validate_model()

encoder = model.encoder(unicode_bigrams=selection.bigrams)
model.save_pretrained(
  "my-unicode-tokenizer",
  unicode_bigrams=selection.bigrams,
)
```

`train()` stops before a new merge below `bigram_cutoff_freq`. Equality is valid
because selection retains all ties at the cutoff. Manual `step()` calls remain
available, while model validation rejects a final merge below the configured cutoff.
Pass the selected bigrams to `model.encoder()` and `model.save_pretrained()` as shown;
the self-describing directory then restores the same pretokenizer configuration.

`unicode_bigram_mixed_boundary="keep"` is the conservative default: it preserves
mixed or unmeasured edges and splits unretained script-to-script edges. Use `"split"`
only when the more aggressive segmentation matches your intended tokenizer.

## Unicode BBPE fallback

Unicode models can spend a configurable share of learned vocabulary slots on byte
merges inside rare Unicode scalars:

```python
trainer = BpeTrainer([], unit="unicode")
trainer.add_word_counter(word_counter)
trainer.train_with_bbpe_fallback(
  vocab_size=10_000,
  primary_vocab_ratio=0.9,
)
model = trainer.validate_model()
```

The ratio applies to learned slots after special tokens and the mandatory 256-byte
alphabet. Primary Unicode training runs first; the fallback pass then learns only
inside omitted scalars and never across scalar boundaries. Unused fallback slots
return to primary training. The resulting model needs no special loading option
because the behavior is encoded in its merge rules.

Fallback is a target-aware, finalizing operation and must run before ordinary
vocabulary growth. Use `primary_vocab_ratio=1.0` when no fallback slots should be
reserved; that delegates to ordinary training and leaves the trainer extendable.

## Streaming and partitioned counting

`add_source` pulls at most 4,096 records or 64 MiB per batch by default. It overlaps
Python iteration with Rust processing using one bounded look-ahead batch. Pass
`prefetch=0` for synchronous processing or override `max_records` and `max_bytes`
for the record sizes and worker memory available.

Counters can be merged after independently counting corpus partitions:

```python
left = pretokenizer.word_counter()
left.add_source(left_partition)

right = pretokenizer.word_counter()
right.add_source(right_partition)

left.merge(right)
trainer.add_word_counter(left)
```

`add_word_counter` consumes the native inventory without constructing a Python
dictionary; the counter is empty and reusable afterward. `word_counter.words()`
remains available for small inventories but copies the complete result into Python.

## Exact bounded-memory training

By default, the trainer retains occurrence postings for every discovered pair.
Set `hot_pair_window_size` to keep an exact top-K candidate window:

```python
trainer = BpeTrainer(
  [],
  unit="unicode",
  hot_pair_window_size=4096,
)
trainer.add_word_counter(word_counter)
trainer.train(vocab_size=10_000)
```

Smaller windows use less persistent posting memory but may require more full
inventory scans when a cold pair wins. Larger windows retain more postings and
usually reduce hydration. The setting affects resource use, not pair frequencies or
the resulting model. Inspect `trainer.hot_pair_window_stats` for hydration, pruning,
resident-pair, and occurrence-capacity diagnostics.

The hot window is not a total process-memory bound: the full word inventory and a
compact global pair table remain resident to preserve exact selection. Inspect
`trainer.memory_usage` for capacity-backed word, pair-table, occurrence, heap, and
model storage. Its `estimated_persistent_bytes` intentionally excludes allocator
retention, stacks, and temporary parallel work.

## Tiktoken-compatible API

Use the familiar `Encoding` surface with an existing model:

```python
from ffbpe import Encoding

encoding = Encoding.from_files(
  "my-tokenizer",
  vocab_file="my-tokenizer/vocab.json",
  merges_file="my-tokenizer/merges.txt",
  special_tokens={"<|endoftext|>": 0},
)

ids = encoding.encode("hello world")
text = encoding.decode(ids)
```

The `ffbpe.tiktoken` namespace exports `Encoding`, `get_encoding`,
`encoding_for_model`, `encoding_name_for_model`, and `list_encoding_names` with
signatures checked against upstream tiktoken. Built-in registry names are currently
limited to fixture models; load trained models explicitly.

## Rust quickstart

```rust
use ffbpe::{
  bpe::{BpeTrainer, Idx},
  traits::Encode,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let mut trainer = BpeTrainer::<u8, Idx>::from_words(
    [("hello", 10), ("world", 7)],
    &[],
  );
  trainer.train_until(260)?;

  let model = trainer.validate_model()?;
  let encoder = model.to_encoder()?;
  let ids = encoder.encode_string("hello")?;

  assert_eq!(encoder.decode(&ids)?, "hello");
  Ok(())
}
```

The same program is available as `examples/quickstart.rs` and compiled in CI.

Known GPT-2, r50k, cl100k, and o200k PAT expressions are implemented by the
workspace's [`ffbpe-pat`](crates/ffbpe-pat) crate. It can also be used directly
when an application needs only zero-copy pretoken ranges or borrowed tokens;
arbitrary patterns continue through FFBPE's regex fallback.

## CLI

The Rust CLI is feature-gated:

```bash
cargo run --release --features cli -- train \
  --vocab-size 10000 \
  --out out/models \
  corpus.txt
```

Run `cargo run --features cli -- train --help` for chunking, boundary, character-unit,
special-token, and output-format options.

## Development

Build the Python extension in a local environment:

```bash
uv sync --dev
maturin develop --release --features py
uv run pytest
```

Useful entry points:

- `python examples/quickstart.py`
- `cargo run --example quickstart`
- `cargo test -p ffbpe-pat`
- `cargo bench -p ffbpe-pat --bench scan`
- `python benchmarks/compare_tiktoken.py`
- `python benchmarks/compare_hf_training.py`
- `cargo bench --bench regression -- suite smoke`
- `cargo bench --bench regression -- suite 64mib --check`

FFBPE is licensed under the [MIT License](LICENSE).
