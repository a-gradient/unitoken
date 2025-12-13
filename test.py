# %%
from unitoken import BpeTrainer, PreTokenizer
pre = PreTokenizer(["<|endoftext|>"], None)
words = pre.get_words_from_file("fixtures/tinystories_sample_5M.txt", 100)

# %%

bpe = BpeTrainer(["<|endoftext|>"], ch="char")
print(bpe.vocab_size)
bpe.add_words(words)
bpe.init_training()

# %%
for i in range(bpe.vocab_size, 1000):
  try:
    bpe.step()
  except Exception as e:
    print(f"Error at step {i}: {e}")
    break

# %%
print(bpe.vocab_size)

# %%
vocabs = dict(bpe.vocabs.items())
vocabs

# %%
bpe.save("test")

# %%
