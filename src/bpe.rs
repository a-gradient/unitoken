use std::{collections::{BTreeMap, BTreeSet}, sync::Arc};

pub type Idx = u32;
pub type Word<C> = Arc<[C]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Character {
  Unicode(char),
  Byte(u8),
}

#[derive(Debug)]
pub struct PreToken<C, I> {
  pub src: Word<C>,
  pub idxs: Vec<I>,
  pub freq: u64,
}

#[derive(Debug)]
pub struct Merge<C, I> {
  pub tp: (I, I),
  pub content: (Word<C>, Word<C>),
  pub target: Option<I>,
  pub occurs_in: BTreeSet<u64>,
  pub freq: u64,
}

impl<C, I> Merge<C, I> {
  pub fn new(tp: (I, I), content: (Word<C>, Word<C>)) -> Self {
    Self {
      tp,
      content,
      target: None,
      occurs_in: BTreeSet::new(),
      freq: 0,
    }
  }

  pub fn add(&mut self, doc_id: u64, freq: u64) {
    self.occurs_in.insert(doc_id);
    self.freq += freq;
  }

  pub fn remove(&mut self, doc_id: &u64, freq: u64) {
    self.freq -= freq;
    self.occurs_in.remove(doc_id);
  }
}

#[derive(Debug, Default)]
pub struct BPE<C = Character> {
  pub vocab: BTreeMap<Idx, Word<C>>,
  pub merges: Vec<Merge<C, Idx>>,
  pub pre_merges: BTreeMap<(Idx, Idx), Merge<C, Idx>>,
  pub words: Vec<PreToken<C, Idx>>,
}

impl<C> BPE<C> {
  pub fn new(words: Vec<PreToken<C, Idx>>) -> Self {
    Self {
      vocab: BTreeMap::new(),
      merges: Vec::new(),
      pre_merges: BTreeMap::new(),
      words,
    }
  }

  fn init_training(&mut self) {
    self.pre_merges.clear();
    for (i, word) in self.words.iter().enumerate() {
      for (j1, j2) in word.idxs.iter().copied().zip(word.idxs.iter().skip(1).copied()) {
        let tp = (j1, j2);
        let merge = self.pre_merges.entry(tp).or_insert_with(|| {
          let content = (
            self.vocab.get(&j1).unwrap().clone(),
            self.vocab.get(&j2).unwrap().clone(),
          );
          Merge::new(tp, content)
        });
        merge.add(i as u64, word.freq);
      }
    }
  }

  fn step(&mut self) {
  }
}

#[cfg(test)]
mod tests {
}
