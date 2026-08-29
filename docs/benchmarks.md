# Benchmarks

FFBPE keeps correctness gates and timing measurements separate. Treat a timing
result as comparable only when the input, model, tokenizer configuration,
environment, and output fingerprints match.

## PAT scanning

On 2026-08-28, the first GigaToken-inspired PAT optimization pass was measured
on an Apple M2 (aarch64 macOS), using Rust 1.97.0 and the default bench profile.
The baseline scanner was from commit `a4329516a75f2d4d86a93dccda2d3deef58a48bc`.
Each measurement below is the median of nine samples scanning the checked-in
English and Chinese TinyStories fixtures (approximately 5 MiB each).

| Pattern | English before → after (MiB/s) | Speedup | Chinese before → after (MiB/s) | Speedup |
|---|---:|---:|---:|---:|
| GPT-2 | 222 → 422 | 1.90× | 517 → 587 | 1.14× |
| cl100k | 169 → 262 | 1.55× | 541 → 623 | 1.15× |
| o200k | 132 → 342 | 2.60× | 381 → 384 | 1.01× |

These are single-thread PAT scan results, not end-to-end BPE encoding or a
comparison against GigaToken itself. The o200k Chinese result is effectively
unchanged. All input hashes and full token-stream fingerprints matched between
baseline, candidate, and the regex reference. The unknown-pattern fallback was
unchanged; its observed timing varied by about 2–4%.

The changes use portable SWAR for ASCII letter runs, direct ASCII/contraction
dispatch, a phase-based o200k word scan, and shared whitespace scanning. They
retain the public iterator API and `regex-syntax` Unicode properties. This pass
also fixes a pre-existing cl100k mismatch: trailing `" \n \t"` must remain one
whitespace token because the end-of-input branch precedes the newline branch.
See the [PAT crate](https://github.com/tokn-ai/ffbpe/blob/master/crates/ffbpe-pat/README.md) for upstream attribution and
the optimization candidates not yet adopted.

Reproduce on each revision with:

```bash
cargo bench --bench regression -- pretokenizer-scan \
  --repeats 9 --max-bytes 5242880 --output out/benchmarks/pat-scan.json
```

The standalone crate benchmark additionally covers generated mixed-script
text, long ASCII words, code, and case transitions. It consumes token slices
and byte lengths rather than measuring a special count-only path.

### Unicode scanner follow-up

On 2026-08-29, the Unicode hot-path follow-up compared merged commit `a385e28`
with the candidate in alternating nine-sample runs on the same Apple M2 and
fixtures. Both builds used the same `Cargo.lock`; complete token-stream
fingerprints matched across all 72 specialized samples.

| Pattern | English before → after (MiB/s) | Speedup | Chinese before → after (MiB/s) | Speedup |
|---|---:|---:|---:|---:|
| GPT-2 | 412 → 426 | 1.03× | 550 → 638 | 1.16× |
| cl100k | 252 → 331 | 1.31× | 629 → 673 | 1.07× |
| o200k | 336 → 350 | 1.04× | 391 → 602 | 1.54× |

Dense non-ASCII letter runs use direct UTF-8 decoding, while short script
transitions retain Rust's standard UTF-8 iterator. The o200k case-state loop
also resolves its Unicode table handle once per word. The unknown-pattern
fallback is unchanged; its timing continued to vary with the regex reference
timing during the alternating runs.

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
