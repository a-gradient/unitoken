use ffbpe::pretokenizer::{
  CL100K_PAT_STR, DEFAULT_PAT_STR, O200K_PAT_STR, R50K_PAT_STR,
};

pub const UNKNOWN_PAT_STR: &str =
  r"\p{L}+|\p{N}{1,4}|[^\s\p{L}\p{N}]+|\s+";

pub const PATTERNS: [(&str, &str); 5] = [
  ("gpt2", DEFAULT_PAT_STR),
  ("r50k", R50K_PAT_STR),
  ("cl100k", CL100K_PAT_STR),
  ("o200k", O200K_PAT_STR),
  ("unknown", UNKNOWN_PAT_STR),
];

pub const DATASETS: [(&str, &str); 2] = [
  ("english", "fixtures/tinystories_sample_5M.txt"),
  (
    "chinese",
    "fixtures/TinyStories_all_data_zh_1M-sample.txt",
  ),
];
