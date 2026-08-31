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
arithmetic) in a small inline wrapper. Only non-ASCII continuations enter the
out-of-line Unicode scanner, keeping its decoder and class-table state out of
ASCII call sites. o200k resolves the class-table handle once per non-ASCII word
and decodes case runs directly from the UTF-8 byte cursor while tracking
upper/shared and lower/shared phases in one forward scan. It remembers the
last shared-character boundary needed by the regex's greedy alternatives.
Whitespace runs are scanned once; contraction matching dispatches directly on
ASCII bytes and retains Unicode case folding.

The public iterators still yield borrowed strings or byte ranges in order. The
internal UTF-8 decoder and class-table lookup use unchecked indexing to remove
redundant bounds checks; their cursors originate from `str`, stay on character
boundaries, and the decoder is tested against every Unicode scalar. There are
no overreads, padding requirements, architecture-specific instructions, or new
dependencies. Unicode properties continue to come from `regex-syntax`, so they
match the reference regex engine's Unicode version.

## Optional SIMD

The non-default `simd` feature enables boundary-mask backends for each
supported architecture:

- AArch64 uses its baseline NEON instruction set.
- x86_64 selects AVX2 once at iterator construction when the CPU supports it.

```bash
cargo bench -p ffbpe-pat --features simd --bench scan -- specialized
```

For GPT-2 and r50k inputs whose first 64-byte window is ASCII, the selected
backend classifies 64 bytes into letter, digit, space, whitespace, and
apostrophe masks. Scalar `u64` algebra derives trustworthy token starts, and the
iterator pops cached boundaries instead of dispatching and scanning every token
independently. Contraction endings are corrected before the boundaries are
exposed.

The implementation is intentionally conservative. Short inputs and inputs whose
first window contains non-ASCII use the scalar scanner for their full lifetime.
Later Unicode-containing windows temporarily return to the scalar scanner. The
feature has no effect outside AArch64 and x86_64, and disabling it removes all
SIMD code. The AArch64 targets supported by this workspace include NEON as a
baseline feature. AVX2 is not part of the x86_64 baseline, so its classifier is
never called without a successful runtime feature check.

The SWAR arithmetic, byte-dispatch strategy, and o200k phase approach are
adapted from [GigaToken](https://github.com/marcelroed/gigatoken) at commit
`fac0114b37120ec8a76362e9ee8e1c742aaafaef`, especially
[`fast/mod.rs`](https://github.com/marcelroed/gigatoken/blob/fac0114b37120ec8a76362e9ee8e1c742aaafaef/src/pretokenize/fast/mod.rs),
[`fast/cl100k.rs`](https://github.com/marcelroed/gigatoken/blob/fac0114b37120ec8a76362e9ee8e1c742aaafaef/src/pretokenize/fast/cl100k.rs), and
[`fast/o200k_family.rs`](https://github.com/marcelroed/gigatoken/blob/fac0114b37120ec8a76362e9ee8e1c742aaafaef/src/pretokenize/fast/o200k_family.rs), plus its
[`unicode.rs`](https://github.com/marcelroed/gigatoken/blob/fac0114b37120ec8a76362e9ee8e1c742aaafaef/src/pretokenize/unicode.rs) table-access strategy.
Its copyright and MIT license are retained in [LICENSE-GIGATOKEN](LICENSE-GIGATOKEN).

GigaToken's SIMD mask construction and boundary algebra informed the optional
GPT-2/r50k backends. The cl100k/o200k mask schemes and buffered span emission
remain separate optimization candidates. A packed Unicode table was benchmarked
but not adopted: halving the table size did not offset the added nibble decoding
cost. Dual-cursor counting and BPE caches do not directly accelerate this
crate's ordered streaming iterator.

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
