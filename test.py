# %%
from unitoken import BpeTrainer, PreTokenizer
pre = PreTokenizer(["<|endoftext|>"], None)
words = pre.get_words_from_file("fixtures/tinystories_sample_5M.txt", 100)

# %%

bpe = BpeTrainer(["<|endoftext|>"], ch="char")
print(bpe.vocab_size)
bpe.add_words(words)
bpe.train(500)
print(bpe.vocab_size)

# %%
vocabs = dict(bpe.vocabs.items())
vocabs

# %%
bpe.save("test")

# %%
