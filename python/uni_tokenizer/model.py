from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ._lib import BpeModelBase
from ._serialization import (
  MODEL_CONFIG_FILENAME,
  MODEL_CONFIG_VERSION,
  ModelConfig,
  write_model_config,
)
from .trainer import FileFormat, Unit, _resolve_format

if TYPE_CHECKING:
  from .encoder import BpeEncoder


class BpeModel:
  """An immutable BPE model produced by :meth:`BpeTrainer.validate_model`."""

  def __init__(self, model: BpeModelBase) -> None:
    self._model = model
    self._encoder_cache: BpeEncoder | None = None

  @property
  def unit(self) -> Unit:
    """Atomic BPE unit used by this model."""
    return cast(Unit, self._model.unit)

  @property
  def vocab(self) -> dict[bytes, int]:
    """Return a snapshot of the validated token-to-id vocabulary."""
    return dict(self._model.get_vocab().items())

  @property
  def last_merge_freq(self) -> int | None:
    """Frequency of the final pair merge, if the model contains one."""
    return self._model.last_merge_freq

  @property
  def special_tokens(self) -> list[str]:
    """Reserved special tokens in vocabulary order."""
    return list(self._model.special_tokens)

  def encoder(
    self,
    *,
    pat_str: str | None = None,
    unicode_bigrams: Sequence[str] | None = None,
    unicode_bigram_mixed_boundary: str = "keep",
  ) -> "BpeEncoder":
    """Build an encoder directly from this model."""
    from .encoder import BpeEncoder
    use_cache = (
      pat_str is None
      and unicode_bigrams is None
      and unicode_bigram_mixed_boundary == "keep"
    )
    if use_cache and self._encoder_cache is not None:
      return self._encoder_cache
    encoder = BpeEncoder._from_encoder(
      self.unit,
      self._model.encoder(
        pat_str=pat_str,
        unicode_bigrams=unicode_bigrams,
        unicode_bigram_mixed_boundary=unicode_bigram_mixed_boundary,
      ),
    )
    if use_cache:
      self._encoder_cache = encoder
    return encoder

  def encode(self, text: str) -> list[int]:
    """Encode text with the model's default pretokenizer."""
    return self.encoder().encode(text)

  def decode(self, ids: Sequence[int]) -> str:
    """Decode token ids into text."""
    return self.encoder().decode(ids)

  def save_vocab_json(
    self,
    path: str | PathLike,
    *,
    format: FileFormat | None = None,
  ) -> None:
    """Save the validated vocabulary to a JSON file."""
    self._model.save_vocab(path, _resolve_format(self.unit, format))

  def save_merges_txt(
    self,
    path: str | PathLike,
    *,
    format: FileFormat | None = None,
  ) -> None:
    """Save the validated merge list to a text file."""
    self._model.save_merges_txt(path, _resolve_format(self.unit, format))

  def save(self, name: str, *, outdir: str | PathLike = ".", format: FileFormat | None = None) -> None:
    """Save `vocab.{name}[{unit}].json` and `merges.{name}[{unit}].txt` into `outdir`."""
    vocab_path = Path(outdir) / f"vocab.{name}[{self.unit}].json"
    merges_path = Path(outdir) / f"merges.{name}[{self.unit}].txt"
    self.save_files(vocab_path, merges_path, format=format)

  def save_files(
    self,
    vocab_path: str | PathLike,
    merges_path: str | PathLike,
    *,
    format: FileFormat | None = None,
  ) -> None:
    """Save the validated vocabulary and merge list to explicit paths."""
    resolved_format = _resolve_format(self.unit, format)
    self._model.save_vocab(vocab_path, resolved_format)
    self._model.save_merges_txt(merges_path, resolved_format)

  def save_pretrained(
    self,
    directory: str | PathLike,
    *,
    format: FileFormat | None = None,
    pat_str: str | None = None,
    unicode_bigrams: Sequence[str] | None = None,
    unicode_bigram_mixed_boundary: str = "keep",
  ) -> None:
    """Save a self-describing model directory loadable by `BpeEncoder.from_pretrained`."""
    # Validate the complete encoding configuration before creating partial output.
    self.encoder(
      pat_str=pat_str,
      unicode_bigrams=unicode_bigrams,
      unicode_bigram_mixed_boundary=unicode_bigram_mixed_boundary,
    )
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_format = _resolve_format(self.unit, format)
    vocab_file = "vocab.json"
    merges_file = "merges.txt"
    self.save_files(
      output_dir / vocab_file,
      output_dir / merges_file,
      format=resolved_format,
    )
    config: ModelConfig = {
      "version": MODEL_CONFIG_VERSION,
      "unit": self.unit,
      "format": resolved_format,
      "vocab_file": vocab_file,
      "merges_file": merges_file,
      "special_tokens": self.special_tokens,
      "pat_str": pat_str,
      "unicode_bigrams": list(unicode_bigrams) if unicode_bigrams is not None else None,
      "unicode_bigram_mixed_boundary": unicode_bigram_mixed_boundary,
    }
    write_model_config(output_dir / MODEL_CONFIG_FILENAME, config)
