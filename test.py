# %%
from unitoken import BpeTrainer

bpe = BpeTrainer(["<|endoftext|>"], ch="char")
print(bpe.vocab_size)
bpe.add_words([("我", 1), ("是", 2), ("一个", 1), ("测试", 1), ("文本", 1)])
bpe.init_training()

# %%
for i in range(100):
  try:
    bpe.step()
  except Exception as e:
    print(f"Error at step {i}: {e}")
    break

# %%
print(bpe.vocab_size)
# %%
bpe.save_vocab("vocab.json", "uni")
bpe.save_merges_txt("merges.txt", "uni")

# %%
