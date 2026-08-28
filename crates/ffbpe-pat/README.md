# ffbpe-pat

`ffbpe-pat` provides zero-copy scanners for the GPT-2, r50k, cl100k,
and o200k tokenizer pretokenization patterns.

The crate recognizes exact known PAT expressions. It deliberately does not
compile arbitrary regular expressions; callers remain responsible for their
fallback behavior.

```rust
use ffbpe_pat::{Pattern, O200K_PATTERN};

let pattern = Pattern::recognize(O200K_PATTERN).unwrap();
let tokens = pattern.split("Hello, 世界!").collect::<Vec<_>>();
```

Use `Pattern::offsets` when byte ranges are more convenient than borrowed
string slices.

## Scanning strategy

ASCII letter runs use portable SWAR (eight bytes at a time using integer
arithmetic), with scalar Unicode continuation. o200k tracks upper/shared and
lower/shared phases in one forward scan, remembering the last shared-character
boundary needed by the regex's greedy alternatives. Whitespace runs are scanned
once; contraction
matching dispatches directly on ASCII bytes and retains Unicode case folding.

The public iterators still yield borrowed strings or byte ranges in order.
There are no unsafe reads, padding requirements, architecture-specific
instructions, or new dependencies. Unicode properties continue to come from
`regex-syntax`, so they match the reference regex engine's Unicode version.

The SWAR arithmetic, byte-dispatch strategy, and o200k phase approach are
adapted from [GigaToken](https://github.com/marcelroed/gigatoken) at commit
`fac0114b37120ec8a76362e9ee8e1c742aaafaef`, especially
[`fast/mod.rs`](https://github.com/marcelroed/gigatoken/blob/fac0114b37120ec8a76362e9ee8e1c742aaafaef/src/pretokenize/fast/mod.rs),
[`fast/cl100k.rs`](https://github.com/marcelroed/gigatoken/blob/fac0114b37120ec8a76362e9ee8e1c742aaafaef/src/pretokenize/fast/cl100k.rs), and
[`fast/o200k_family.rs`](https://github.com/marcelroed/gigatoken/blob/fac0114b37120ec8a76362e9ee8e1c742aaafaef/src/pretokenize/fast/o200k_family.rs).
Its copyright and MIT license are retained in [LICENSE-GIGATOKEN](LICENSE-GIGATOKEN).

GigaToken's 64-byte SIMD boundary masks, packed Unicode tables, and buffered
span emission are separate optimization candidates. Dual-cursor counting and
BPE caches do not directly accelerate this crate's ordered streaming iterator.

## Validation

```bash
cargo test -p ffbpe-pat
cargo bench -p ffbpe-pat --bench scan -- specialized
```

Tests compare complete token streams and offsets against `fancy-regex`,
including exhaustive short class combinations, ASCII block boundaries and
tails, Unicode mixtures, marks, contractions, and trailing whitespace. The
benchmark consumes actual tokens and their byte lengths, covering English,
Chinese, mixed scripts, long ASCII runs, code, and case transitions.
