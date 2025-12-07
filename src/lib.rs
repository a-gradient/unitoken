pub mod bpe;
pub mod pretokenizer;

#[derive(Debug)]
pub struct Error {
  pub msg: String,
  pub loc: (&'static str, u32),
}
