from pathlib import Path
from tempfile import TemporaryDirectory

from ffbpe import BpeEncoder, train_bpe


model = train_bpe(
  ["hello world", "hello tokenizer"],
  vocab_size=280,
  special_tokens=["<|endoftext|>"],
)

ids = model.encode("hello world")
assert model.decode(ids) == "hello world"

with TemporaryDirectory() as directory:
  model_dir = Path(directory) / "my-tokenizer"
  model.save_pretrained(model_dir)
  restored = BpeEncoder.from_pretrained(model_dir)
  assert restored.decode(restored.encode("hello world")) == "hello world"

print(f"trained {len(model.vocab)} tokens; encoded 'hello world' as {ids}")
