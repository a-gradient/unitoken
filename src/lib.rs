pub mod bpe;
pub mod pretokenizer;

#[derive(thiserror::Error, Debug)]
pub enum MyError {
  #[error("IO error: {0}")]
  Io(#[from] std::io::Error),
  #[error("Regex error: {0}")]
  Regex(#[from] fancy_regex::Error),
  #[error("UTF-8 error: {0}")]
  Utf8(#[from] std::str::Utf8Error),
  #[error("Out of vocabulary bytes: {0}")]
  OovBytes(String),
}

pub type MyResult<T> = Result<T, MyError>;
