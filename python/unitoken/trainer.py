from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Literal
from ._lib import BpeTrainer_Character_CharIdx, BpeTrainer_u8_Idx

CharLevel = Literal["char", "u8"]
OutputFormat = Literal["uni", "bpe"]

class BpeTrainer:
  def __init__(self, special_tokens: Sequence[str], *, ch: CharLevel = "u8", output_format: OutputFormat | None = None) -> None:
    # super().__init__()
    self._ch: CharLevel = ch
    self.output_format = "uni"
    if ch == "char":
      self._trainer = BpeTrainer_Character_CharIdx(special_tokens=special_tokens)
    else:
      self.output_format = output_format or "bpe"
      self._trainer = BpeTrainer_u8_Idx(special_tokens=special_tokens)

  @property
  def vocab_size(self) -> int:
    return self._trainer.vocab_size()

  @property
  def char_level(self) -> CharLevel:
    return self._ch

  @property
  def vocabs(self):
    return self._trainer.get_vocabs()

  def add_words(self, words: Sequence[tuple[str, int]]) -> None:
    self._trainer.add_words(words)

  def init_training(self) -> None:
    self._trainer.init_training()

  def step(self) -> None:
    self._trainer.step()

  def save(self, name: str, *, outdir: str | PathLike = ".", output_format: OutputFormat | None = None) -> None:
    vocab_path = Path(outdir) / f"vocab.{name}[{self.char_level}].json"
    merges_path = Path(outdir) / f"merges.{name}[{self.char_level}].txt"
    spec = output_format or self.output_format
    self._trainer.save_vocab(vocab_path, spec)
    self._trainer.save_merges_txt(merges_path, spec)
