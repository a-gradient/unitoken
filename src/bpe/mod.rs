use std::{collections::{BTreeMap, BTreeSet}, sync::Arc};

pub mod trainer;
pub mod encoder;
pub mod utils;

pub use trainer::BpeTrainer;
pub use encoder::BpeEncoder;
use utils::*;

use ordermap::OrderMap;

pub type Idx = u32;
pub type Word<C> = Arc<[C]>;
pub type Freq = i64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Character {
  Unicode(char),
  Byte(u8),
}

#[derive(Debug)]
pub struct PreToken<C, I> {
  pub src: Word<C>,
  pub idxs: Vec<I>,
  pub freq: Freq,
}

impl<C, I> PreToken<C, I> {
  pub fn display(&self) -> String where Word<C>: WordExt {
    format!("<{:?} => {}>", self.src.display(), self.freq)
  }

  pub fn display_split(&self, vocabs: &BTreeMap<I, Word<C>>) -> String where I: Ord, C: Clone, Word<C>: WordExt {
    let parts = self
      .idxs
      .iter()
      .map(|i| vocabs.get(i).unwrap().display())
      .collect::<Vec<_>>()
      .join(" ");
    format!("<{} => {}>", parts, self.freq)
  }
}

#[derive(Debug, Clone)]
pub struct Merge<C, I> {
  pub tp: (I, I),
  pub content: (Word<C>, Word<C>),
  pub target: Option<I>,
  pub data: MergeData,
}

impl<C, I> Merge<C, I> {
  pub fn merged_content(&self) -> Word<C> where C: Clone {
    let mut v = Vec::with_capacity(self.content.0.len() + self.content.1.len());
    v.extend_from_slice(&self.content.0);
    v.extend_from_slice(&self.content.1);
    Arc::<[C]>::from(v.into_boxed_slice())
  }

  pub fn with_target(mut self, target: I) -> Self {
    self.target = Some(target);
    self
  }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct MergeData {
  pub occurs_in: BTreeSet<u64>,
  pub freq: Freq,
}

impl MergeData {
  pub fn new(freq: Freq) -> Self {
    Self {
      occurs_in: BTreeSet::new(),
      freq,
    }
  }

  #[must_use]
  pub fn add_occurs_in<I: IntoIterator<Item = u64>>(self, iter: I) -> Self {
    Self {
      occurs_in: iter.into_iter().collect(),
      freq: self.freq,
    }
  }

  pub fn occurs_in_vec(&self) -> Vec<u64> {
    self.occurs_in.iter().copied().collect::<Vec<u64>>()
  }
}

impl<C, I> Merge<C, I> {
  pub fn new(tp: (I, I), content: (Word<C>, Word<C>)) -> Self {
    Self {
      tp,
      content,
      target: None,
      data: MergeData::default(),
    }
  }

  pub fn add(&mut self, doc_id: u64, freq: Freq) {
    self.data.occurs_in.insert(doc_id);
    self.data.freq += freq;
  }

  pub fn remove(&mut self, doc_id: &u64, freq: Freq) {
    self.data.freq -= freq;
    self.data.occurs_in.remove(doc_id);
  }
}
