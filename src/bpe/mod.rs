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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CharIdx {
  Idx(Idx),
  Char(char),
}

#[derive(Debug)]
pub struct PreToken<C, I> {
  pub src: Word<C>,
  pub idxs: Vec<I>,
  pub freq: Freq,
}

impl<C, I> PreToken<C, I> {
  pub fn display(&self) -> String where Word<C>: WordDebugExt {
    format!("<{:?} => {}>", self.src.debug_display(), self.freq)
  }

  pub fn display_split(&self, vocabs: &BTreeMap<I, Word<C>>) -> String where I: Ord, C: Clone, Word<C>: WordDebugExt {
    let parts = self
      .idxs
      .iter()
      .map(|i| vocabs.get(i).unwrap().debug_display())
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

pub trait Cachable: std::hash::Hash + Send + Sync + 'static { }
impl<C: std::hash::Hash + Send + Sync + 'static> Cachable for C { }

pub trait IdxLike: Ord + std::hash::Hash + Eq + Copy + Send + Sync + 'static {
  fn from_u64(v: u64) -> Self;
  fn to_u64(self) -> u64;
  fn decode_from_u64(v: u64, start: u64) -> Option<Self> {
    Some(Self::from_u64(v - start))
  }
  fn encode_to_u64(&self, start: u64) -> u64 {
    self.to_u64() + start
  }
}
impl IdxLike for Idx {
  fn from_u64(v: u64) -> Self {
    v as Self
  }
  fn to_u64(self) -> u64 {
    self as u64
  }
}
impl IdxLike for CharIdx {
  fn from_u64(v: u64) -> Self {
    CharIdx::Idx(v as Idx)
  }
  fn to_u64(self) -> u64 {
    match self {
      CharIdx::Idx(i) => i as u64,
      CharIdx::Char(c) => unimplemented!("Cannot convert CharIdx::Char to u64: {:?} [u{:04x}]", c, c as u32),
    }
  }
}

pub trait CharToIdx<I: IdxLike> {
  fn char_to_idx(&self, start: u64) -> I;
}

pub trait HasChar<C>: Sized {
  fn get_char(self) -> Option<char>;
  fn from_char(_c: char) -> Option<Self> { None }
  fn idx_to_word(self) -> Option<Word<C>> where for<'a> &'a str: ToWord<C>{
    self.get_char().map(|i| i.to_string().to_word())
  }
}
impl CharToIdx<Idx> for u8 {
  fn char_to_idx(&self, start: u64) -> Idx {
    (*self as u64 + start) as Idx
  }
}
impl<C> HasChar<C> for Idx {
  fn get_char(self) -> Option<char> {
    None
  }
}
impl CharToIdx<CharIdx> for char {
  fn char_to_idx(&self, start: u64) -> CharIdx {
    if self.is_ascii() {
      CharIdx::Idx(*self as u8 as Idx + start as Idx)
    } else {
      CharIdx::Char(*self)
    }
  }
}
impl<C> HasChar<C> for char {
  fn get_char(self) -> Option<char> {
    Some(self)
  }
  fn from_char(c: char) -> Option<Self> {
    Some(c)
  }
}
impl CharToIdx<CharIdx> for u8 {
  fn char_to_idx(&self, start: u64) -> CharIdx {
    CharIdx::Idx((*self as u64 + start) as Idx)
  }
}
impl<C> HasChar<C> for CharIdx {
  fn get_char(self) -> Option<char> {
    match self {
      CharIdx::Char(c) => Some(c),
      CharIdx::Idx(_) => None,
    }
  }
  fn from_char(c: char) -> Option<Self> {
    Some(CharIdx::Char(c))
  }
}
impl CharToIdx<CharIdx> for Character {
  fn char_to_idx(&self, start: u64) -> CharIdx {
    match self {
      Character::Unicode(c) => c.char_to_idx(start),
      Character::Byte(b) => b.char_to_idx(start),
    }
  }
}
