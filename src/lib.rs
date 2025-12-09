#[macro_use]
extern crate tracing;

pub mod bpe;
pub mod spec;
pub mod pretokenizer;

#[derive(thiserror::Error, Debug)]
pub enum MyError {
  #[error("IO error: {0}")]
  Io(#[from] std::io::Error),
  #[error("Regex error: {0}")]
  Regex(#[from] fancy_regex::Error),
  #[error("Json error: {0}")]
  Json(#[from] serde_json::Error),
  #[error("Merge txt error: {0} at line {1}")]
  MergeTxt(&'static str, usize),
  #[error("UTF-8 error: {0}")]
  Utf8(#[from] std::str::Utf8Error),
  #[error("Character not in printable set: {0}")]
  InvalidPrintableChar(char),
  #[error("Character not in printable set: {0}")]
  InvalidPrintableEscape(String),
  #[error("Out of vocabulary: {0}")]
  Oov(String),
  #[error("Out of vocabulary idx: {0}")]
  OovIdx(u64),
  #[error("Out of vocabulary bytes: {0}")]
  OovBytes(String),
  #[error("Arrow error: {0}")]
  Arrow(#[from] arrow::error::ArrowError),
  #[error("Parquet error: {0}")]
  Parquet(#[from] parquet::errors::ParquetError),
}

pub type MyResult<T> = Result<T, MyError>;

#[cfg(test)]
#[allow(dead_code)]
mod tests {
  pub enum TestData {
    /// fixtures/tinystories_sample_5M.txt
    TinyStories5M,
    /// fixtures/_words.TinyStoriesV2-GPT4-train.json
    TinyStroies,
    /// https://huggingface.co/datasets/52AI/TinyStoriesZh
    TinyStoriesZh
  }
}
