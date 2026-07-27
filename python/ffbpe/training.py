from collections.abc import Iterable, Sequence

from .model import BpeModel
from .pretokenizer import PreTokenizer
from .trainer import BpeTrainer, Unit


def train_bpe(
  texts: str | Iterable[str],
  *,
  vocab_size: int,
  special_tokens: Sequence[str] = (),
  unit: Unit = "byte",
  hot_pair_window_size: int | None = None,
  max_records: int = 4096,
  max_bytes: int = 64 * 1024 * 1024,
  prefetch: int = 1,
) -> BpeModel:
  """Train a BPE model from text records using bounded native word counting.

  `texts` may be one string or a one-pass iterable of independent text records.
  For Unicode-bigram selection and other two-pass workflows, use
  :class:`PreTokenizer` and :class:`BpeTrainer` directly.
  """
  tokens = list(special_tokens)
  pretokenizer = PreTokenizer(tokens)
  counter = pretokenizer.word_counter()
  source = [texts] if isinstance(texts, str) else texts
  counter.add_source(
    source,
    max_records=max_records,
    max_bytes=max_bytes,
    prefetch=prefetch,
  )

  trainer = BpeTrainer(
    tokens,
    unit=unit,
    hot_pair_window_size=hot_pair_window_size,
  )
  trainer.add_word_counter(counter)
  trainer.train(vocab_size)
  return trainer.validate_model()
