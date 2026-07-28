# Multilingual corpora

Unicode-heavy corpora can produce nearly unique pretokenized words. FFBPE offers
two separate tools for this problem:

1. Unicode-bigram inventory shaping changes pretokenizer boundaries using
   measured corpus frequencies.
2. Unicode byte fallback spends part of the learned vocabulary on UTF-8 byte
   merges inside rare scalars.

Benchmark both choices on representative text; neither is a universal default.

## Select Unicode bigrams

Selection is a two-pass workflow, so the corpus must be replayable.

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
```

Selection includes every tie at `cutoff_freq`. Carry that boundary into the
trainer so automatic training stops before learning a pair below the measured
selection boundary:

```python
trainer = BpeTrainer(
  [],
  unit="unicode",
  bigram_cutoff_freq=selection.cutoff_freq,
)
trainer.add_word_counter(word_counter)
trainer.train(vocab_size=10_000)
model = trainer.validate_model()
```

Pass the same bigrams when creating or saving the encoder:

```python
model.save_pretrained(
  "my-unicode-tokenizer",
  unicode_bigrams=selection.bigrams,
)
```

## Add byte fallback for rare scalars

```python
trainer = BpeTrainer([], unit="unicode")
trainer.add_word_counter(word_counter)
trainer.train_with_bbpe_fallback(
  vocab_size=10_000,
  primary_vocab_ratio=0.9,
)
model = trainer.validate_model()
```

The ratio applies to learned slots after special tokens and the mandatory
256-byte alphabet. The fallback phase learns only within omitted Unicode
scalars and never across scalar boundaries.

!!! warning

    Byte fallback is a finalizing operation and must start before ordinary
    vocabulary growth. Create a new trainer if you need to train further.
