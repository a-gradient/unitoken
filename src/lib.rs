pub mod bpe;
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
  #[error("Out of vocabulary: {0}")]
  Oov(String),
  #[error("Out of vocabulary bytes: {0}")]
  OovBytes(String),
}

pub type MyResult<T> = Result<T, MyError>;
