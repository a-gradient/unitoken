# tokn-pat

`tokn-pat` provides zero-copy scalar scanners for the GPT-2, r50k, cl100k,
and o200k tokenizer pretokenization patterns.

The crate recognizes exact known PAT expressions. It deliberately does not
compile arbitrary regular expressions; callers remain responsible for their
fallback behavior.

```rust
use tokn_pat::{Pattern, O200K_PATTERN};

let pattern = Pattern::recognize(O200K_PATTERN).unwrap();
let tokens = pattern.split("Hello, 世界!").collect::<Vec<_>>();
```

Use `Pattern::offsets` when byte ranges are more convenient than borrowed
string slices.
