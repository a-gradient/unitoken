use std::collections::BTreeMap;

use crate::{MyResult, bpe::{Merge, Word}};

pub mod gpt2;

pub trait Spec<Idx, Char> {
  fn encode_vocab<W: std::io::Write>(&self, w: W, vocab: &BTreeMap<Idx, Word<Char>>) -> MyResult<()>;
  fn decode_vocab<R: std::io::Read>(&self, r: R) -> MyResult<BTreeMap<Idx, Word<Char>>>;

  fn encode_merges<W: std::io::Write>(&self, w: W, merges: &Vec<Merge<Char, Idx>>) -> MyResult<()>;
  fn decode_merges<R: std::io::Read>(&self, r: R, vocab: &BTreeMap<Idx, Word<Char>>) -> MyResult<Vec<Merge<Char, Idx>>>;
}
