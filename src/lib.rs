pub mod bpe;
pub mod pretokenizer;

#[derive(thiserror::Error, Debug)]
pub enum MyError {
  #[error("IO error: {0}")]
  Io(#[from] std::io::Error),
  #[error("Regex error: {0}")]
  Regex(#[from] fancy_regex::Error),
}

pub type MyResult<T> = Result<T, MyError>;
