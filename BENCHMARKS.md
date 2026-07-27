# Benchmark methodology

FFBPE keeps correctness gates and timing measurements separate. Timing results are
informational unless the input, model, configuration, environment, and output
fingerprints all match.

## What the benchmark suite checks

The Rust regression suites record:

- input and model file identities;
- tokenizer unit, model format, regex, chunking, and Unicode-bigram configuration;
- Unicode BBPE allocation and model-vocabulary bigram splitting configuration;
- token counts and SHA-256 fingerprints;
- model vocabulary and merge fingerprints;
- deterministic repeats;
- exact versus bounded-memory model parity;
- timing and available RSS measurements.

CI runs smoke, pretokenizer, byte codec, and Unicode codec regression cases. Reports
are uploaded as workflow artifacts.

## Unicode-bigram inventory shaping

The README comparison uses release runs with a 64 MiB FineWeb2 Chinese fixture and a
target vocabulary size of 10,000:

| Pipeline | Unique words | Occurrences | Total training | Merge steps |
|---|---:|---:|---:|---:|
| Regular Unicode inventory | 1,803,009 | 5,774,521 | 26.681 s | 20.416 s |
| Retained Unicode bigrams | 606,153 | 15,901,831 | 3.702 s | 3.034 s |

The occurrence counts differ because bigram shaping changes segmentation. This
comparison measures the effect of the FFBPE pipeline on inventory shape and
training cost; it is not a model-parity claim.

Corpus language, bigram selection parameters, mixed-boundary mode, and vocabulary
target can materially change the result.

## Exact bounded-memory training

One release run used a 1 GiB FineWeb2 Chinese Unicode-bigram inventory with 3,855,974
unique words and a target vocabulary size of 10,000:

| Occurrence mode | Observed training peak RSS | Total training | Hydration scans |
|---|---:|---:|---:|
| Exact, default | 1,797 MiB | 5.58 s | — |
| K=4096 | 1,649 MiB | 5.85 s | 2 |

Both modes produced the same final merge frequency and model. Peak RSS is
process-level and environment-dependent; benchmark representative inventories
before selecting a production window size.

## Comparison with tiktoken

Run:

```bash
python benchmarks/compare_tiktoken.py
```

The script checks token equality before reporting FFBPE and upstream tiktoken
encode/decode medians. Do not compare results produced from different vocabularies,
regexes, special-token policies, or text slices.

Install the optional comparison dependency with:

```bash
uv pip install "tiktoken>=0.12.0"
```

## Comparison with Hugging Face tokenizers

For a fixed compressed word inventory:

```bash
python benchmarks/compare_hf_training.py
```

FFBPE receives `(word, frequency)` pairs directly. Hugging Face receives an
expanded iterator because its Python trainer API does not accept compressed counts.
This benchmark demonstrates the value of FFBPE's compressed-inventory training
contract; it is not a pure trainer-algorithm comparison.

For an end-to-end raw-text comparison:

```bash
python benchmarks/compare_hf_training.py \
  --text /path/to/corpus.txt \
  --chunk-size 1048576 \
  --boundary line \
  --repeats 3
```

Raw mode reports FFBPE pretokenization and BPE training separately, then compares
their total with Hugging Face raw training. By default, Hugging Face receives the
same chunk boundaries. Treat a speed result as comparable only when the report also
confirms vocabulary parity; a report with `same_vocab: false` is diagnostic, not a
parity benchmark.

Install the optional dependency with:

```bash
uv pip install "tokenizers>=0.22.1"
```

## Reproducible regression suites

Named profiles keep trainer, pretokenizer, and codec cases in one reviewable
configuration:

```bash
cargo bench --bench regression --no-run
cargo bench --bench regression -- suite smoke
cargo bench --bench regression -- suite 64mib
cargo bench --bench regression -- suite 1gib
```

The checked-in smoke profile uses repository fixtures and includes Unicode BBPE
training plus a codec case for its pinned model. The larger profiles expect prepared
FineWeb2 Chinese inventories under `out/data/`. Validate configuration and artifact
dependencies without running benchmark cases:

```bash
cargo bench --bench regression -- suite 64mib --check
```

Codec cases can set `split_on_vocab_bigrams: false` for measured opt-out
comparisons. Reports and encoder fingerprints record that choice. The harness also
supports expected input, vocabulary, merge, token-count, token-stream, and model
fingerprints. See:

```bash
cargo bench --bench regression -- --help
```

## Reporting new headline numbers

Before adding a number to the README:

1. Use a release build.
2. Record the exact corpus slice and its hash.
3. Record tokenizer and chunking configuration.
4. Run enough repeats to report a median.
5. State whether vocabularies or token streams match.
6. Record hardware, OS, thread count, package versions, and git revision.
7. Preserve the machine-readable report or publish it as a CI artifact.
8. Describe asymmetrical input contracts next to the result.
