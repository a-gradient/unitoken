from collections.abc import Sequence
from os import PathLike
from typing import Literal
from ._lib import BpeTrainer_Character_CharIdx, BpeTrainer_u8_Idx

CharLevel = Literal["char", "u8"]

class BpeTrainer:
  def __init__(self, special_tokens: Sequence[str], *, ch: CharLevel = "u8") -> None:
    # super().__init__()
    self._ch: CharLevel = ch
    if ch == "char":
      self._trainer = BpeTrainer_Character_CharIdx(special_tokens=special_tokens)
    else:
      self._trainer = BpeTrainer_u8_Idx(special_tokens=special_tokens)

  @property
  def vocab_size(self) -> int:
    return self._trainer.vocab_size()

  @property
  def char_level(self) -> CharLevel:
    return self._ch

  def add_words(self, words: Sequence[tuple[str, int]]) -> None:
    self._trainer.add_words(words)

  def init_training(self) -> None:
    self._trainer.init_training()

  def step(self) -> None:
    self._trainer.step()

  def save_merges_txt(self, path: str | PathLike, spec: str) -> None:
    self._trainer.save_merges_txt(path, spec)

  def save_vocab(self, path: str | PathLike, spec: str) -> None:
    self._trainer.save_vocab(path, spec)
