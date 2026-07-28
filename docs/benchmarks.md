# Benchmarks

FFBPE keeps correctness gates and timing measurements separate. Treat a timing
result as comparable only when the input, model, tokenizer configuration,
environment, and output fingerprints match.

## Unicode-bigram inventory shaping

One release run used a 64 MiB FineWeb2 Chinese fixture and a target vocabulary
of 10,000:

| Pipeline | Unique words | Total BPE training |
|---|---:|---:|
| Regular Unicode inventory | 1,803,009 | 26.681 s |
| Retained Unicode bigrams | 606,153 | 3.702 s |

For this workload, shaping produced an approximately 3× smaller inventory and
7× faster training. It changes corpus segmentation and is not a model-parity
claim.

## Exact bounded-memory training

On a 1 GiB FineWeb2 Chinese Unicode-bigram inventory:

| Occurrence mode | Observed peak RSS | Total training | Hydration scans |
|---|---:|---:|---:|
| Exact, default | 1,797 MiB | 5.58 s | — |
| Window size 4,096 | 1,649 MiB | 5.85 s | 2 |

Both modes produced the same model. Peak RSS and timings are
environment-dependent.

See the
[benchmark methodology](https://github.com/tokn-ai/ffbpe/blob/master/BENCHMARKS.md)
for contracts, qualifications, and reproduction commands.
