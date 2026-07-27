use std::collections::BTreeMap;

use crate::{
  MyError, MyResult,
  spec::Spec,
};

use super::{BpeEncoder, CharIdx, Character, Freq, Idx, IdxLike, Merge, Word};

/// An immutable BPE model produced by validating a trainer snapshot.
#[derive(Debug)]
pub struct BpeModel<C, I> {
  special_tokens: Vec<String>,
  vocab: BTreeMap<I, Word<C>>,
  merges: Vec<Merge<C, I>>,
}

impl<C, I> BpeModel<C, I> {
  pub(crate) fn new(
    special_tokens: Vec<String>,
    vocab: BTreeMap<I, Word<C>>,
    merges: Vec<Merge<C, I>>,
  ) -> Self {
    Self {
      special_tokens,
      vocab,
      merges,
    }
  }

  /// Reserved special tokens in vocabulary order.
  pub fn special_tokens(&self) -> &[String] {
    &self.special_tokens
  }

  /// Validated token-id vocabulary.
  pub fn vocab(&self) -> &BTreeMap<I, Word<C>> {
    &self.vocab
  }

  /// Validated merge rules in rank order.
  pub fn merges(&self) -> &[Merge<C, I>] {
    &self.merges
  }

  /// Frequency of the final pair merge, if the model contains one.
  pub fn last_merge_freq(&self) -> Option<Freq> {
    self.merges.last().map(|merge| merge.data.freq)
  }

  /// Serialize the vocabulary to JSON using `spec`.
  pub fn save_vocab_json<W: std::io::Write>(&self, spec: &dyn Spec<C, I>, mut writer: W) -> MyResult<()> {
    spec.encode_vocab(&mut writer, &self.vocab)
  }

  /// Serialize the merge list to text using `spec`.
  pub fn save_merges_txt<W: std::io::Write>(&self, spec: &dyn Spec<C, I>, mut writer: W) -> MyResult<()> {
    spec.encode_merges(&mut writer, &self.merges)
  }
}

fn required_target<C, I: Copy>(merge: &Merge<C, I>) -> MyResult<I> {
  merge.target.ok_or_else(|| {
    MyError::InvalidBpeModel("validated merge is missing its target token id".to_string())
  })
}

impl BpeModel<u8, Idx> {
  /// Build an encoder directly from this validated byte model.
  pub fn to_encoder(&self) -> MyResult<BpeEncoder<u8>> {
    let merges = self.merges.iter()
      .map(|merge| Ok((merge.tp, required_target(merge)?)))
      .collect::<MyResult<Vec<_>>>()?;
    BpeEncoder::new(self.vocab.clone(), merges, self.special_tokens.clone())
  }
}

impl BpeModel<Character, CharIdx> {
  /// Build an encoder directly from this validated Unicode model.
  pub fn to_encoder(&self) -> MyResult<BpeEncoder<Character>> {
    let vocab = self.vocab.iter()
      .map(|(token_id, token)| (token_id.to_u64() as Idx, token.clone()))
      .collect();
    let merges = self.merges.iter()
      .map(|merge| {
        let target = required_target(merge)?;
        Ok((
          (
            merge.tp.0.to_u64() as Idx,
            merge.tp.1.to_u64() as Idx,
          ),
          target.to_u64() as Idx,
        ))
      })
      .collect::<MyResult<Vec<_>>>()?;
    BpeEncoder::new(vocab, merges, self.special_tokens.clone())
  }
}
