# Train a tokenizer

The high-level [`train_bpe()`][ffbpe.training.train_bpe] shortcut is appropriate
for most byte-level models. Use [`PreTokenizer`][ffbpe.pretokenizer.PreTokenizer]
and [`BpeTrainer`][ffbpe.trainer.BpeTrainer] separately when you need control
over counting, partitioning, or training.

## Stream a replayable corpus

```python
from ffbpe import BpeTrainer, PreTokenizer


class Corpus:
  def scan(self):
    yield "first document"
    yield "second document"


pretokenizer = PreTokenizer(["<|endoftext|>"])
counter = pretokenizer.word_counter()
counter.add_source(Corpus().scan())

trainer = BpeTrainer(["<|endoftext|>"], unit="byte")
trainer.add_word_counter(counter)
trainer.train(vocab_size=10_000)
model = trainer.validate_model()
```

`add_source()` pulls at most 4,096 records or 64 MiB per native batch by
default. One bounded batch can be prefetched while Rust processes the current
batch. Set `prefetch=0` for synchronous iteration.

## Merge partitioned counters

Counters can be populated independently and merged before training:

```python
left = pretokenizer.word_counter()
left.add_source(left_partition)

right = pretokenizer.word_counter()
right.add_source(right_partition)

left.merge(right)
trainer.add_word_counter(left)
```

`add_word_counter()` consumes the native inventory without copying it into a
Python dictionary. The counter is empty and reusable afterward.

## Bound pair-posting memory

```python
trainer = BpeTrainer(
  [],
  unit="byte",
  hot_pair_window_size=4096,
)
trainer.add_word_counter(counter)
trainer.train(vocab_size=10_000)
```

The window bounds persistent occurrence postings, not the frequency table.
FFBPE still selects the exact global winner and preserves deterministic
tie-breaking, but a small window can require additional inventory scans.

Inspect [`hot_pair_window_stats`][ffbpe.trainer.BpeTrainer.hot_pair_window_stats]
after training to see whether hydration scans occurred.
