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

The automated PR report runs this default build on both revisions, then repeats
the scan with `--features simd` as a separate report. Its SIMD section compares
the feature builds across revisions and the PR's scalar and feature builds on
the same inputs. The report records whether AVX2, NEON, or only the scalar
fallback was available. GPT-2 and r50k are reported separately; Chinese rows
are controls because their leading non-ASCII window disables the optional
ASCII SIMD backend.

The standalone crate benchmark additionally covers generated mixed-script
text, long ASCII words, code, and case transitions. It consumes token slices
and byte lengths rather than measuring a special count-only path.

### Unicode scanner follow-up

On 2026-08-29, the Unicode hot-path follow-up compared merged commit `a385e28`
with the candidate in nine-sample runs on the same Apple M2 and fixtures. Both
builds used the same `Cargo.lock`; complete token-stream fingerprints matched
across all 72 samples.

| Pattern | English before → after (MiB/s) | Speedup | Chinese before → after (MiB/s) | Speedup |
|---|---:|---:|---:|---:|
| GPT-2 | 412 → 438 | 1.06× | 550 → 632 | 1.15× |
| cl100k | 252 → 393 | 1.56× | 629 → 694 | 1.10× |
| o200k | 336 → 361 | 1.07× | 391 → 613 | 1.57× |

The ASCII SWAR scanner remains a small inline wrapper; only non-ASCII
continuations enter the out-of-line Unicode scanner. This keeps Unicode code
out of the frequent ASCII call sites. Dense non-ASCII letter runs use direct
UTF-8 decoding, while short script transitions retain Rust's standard UTF-8
iterator. The o200k case-state loop also resolves its Unicode table handle once
per word.

The cl100k English fixture produced 1,242,910 pretokens (4.2 bytes per token),
while the same-size Chinese fixture produced 270,425 (19.4 bytes per token).
Consequently, English pays call-site and dispatch costs about 4.6 times as
often per input byte and benefits more from the smaller inline path. This is a
compiler- and workload-sensitive result, not a general claim that English text
scans 1.56× faster on every target. The unknown-pattern fallback is unchanged;
its observed throughput varied by less than 2%.

### Optional AArch64 NEON scanner

On 2026-08-31, a feature-gated NEON follow-up was measured on the same Apple M2
with Rust 1.97.0. The `simd` feature was disabled for the scalar baseline and
enabled for the candidate. The checked-in 5 MiB English fixture used 15 timing
samples; complete token-stream fingerprints matched:

| Harness and pattern | Scalar → NEON (MiB/s) | Speedup |
|---|---:|---:|
| Regression, GPT-2 English | 424 → 676 | 1.59× |
| Criterion, GPT-2 English | 475 → 657 | 1.38× |
| Criterion, r50k English | 538 → 675 | 1.25× |

The backend activates only when the first 64-byte input window is ASCII. NEON
classifies letter, digit, space, whitespace, and apostrophe lanes into `u64`
masks; scalar bit algebra derives token starts and fixes contraction endings.
Only boundaries through byte 60 are cached so contractions near the batch edge
return to the scalar scanner. Unicode-led fixtures stay on the scalar backend;
their timings are therefore controls rather than SIMD speedup claims.

These measurements cover the non-default AArch64 backend. Reproduce the two
modes with the regression benchmark, adding `--features simd` for the candidate.
The standalone cases use `cargo bench -p ffbpe-pat --bench scan` with and
without `--features simd`.

An x86_64 follow-up adds an AVX2 classifier behind the same feature. It uses the
same boundary cache and scalar algebra, but selects AVX2 once at iterator
construction because AVX2 is not an x86_64 baseline feature. Correctness is
executed on x86_64 CI; throughput will be reported only after a controlled run
on native x86_64 hardware.

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
