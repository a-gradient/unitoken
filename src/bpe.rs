use std::{collections::{BTreeMap, BTreeSet}, sync::Arc};

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

#[derive(Debug, Clone)]
pub struct Merge<C, I> {
  pub tp: (I, I),
  pub content: (Word<C>, Word<C>),
  pub target: Option<I>,
  pub occurs_in: BTreeSet<u64>,
  pub freq: Freq,
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

  pub fn add(&mut self, doc_id: u64, freq: Freq) {
    self.occurs_in.insert(doc_id);
    self.freq += freq;
  }

  pub fn remove(&mut self, doc_id: &u64, freq: Freq) {
    self.freq -= freq;
    self.occurs_in.remove(doc_id);
  }
}

#[derive(Debug, Default)]
pub struct BPE<C = u8> {
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

  fn merge(&mut self, merge: &Merge<C, Idx>) -> BTreeMap<(Idx, Idx), Freq> {
    let mut changes = BTreeMap::<(Idx, Idx), Freq>::new();
    for w in self.words.iter_mut() {
      let mut new_idxs = Vec::with_capacity(w.idxs.len());
      let mut i = 0;
      let mut last_tp = None;
      while i < w.idxs.len() - 1 {
        let tp = (w.idxs[i], w.idxs[i + 1]);
        if tp == merge.tp {
          if let Some(target) = merge.target {
            new_idxs.push(target);
          }
          i += 2;
          *changes.entry(tp).or_default() -= w.freq;
          if let Some(tp) = last_tp {
            *changes.entry(tp).or_default() -= w.freq;
            *changes.entry((tp.0, Idx::MAX)).or_default() += w.freq;
          }
          if i < w.idxs.len() {
            let tp = (w.idxs[i - 1], w.idxs[i]);
            *changes.entry(tp).or_default() -= w.freq;
            *changes.entry((Idx::MAX, tp.1)).or_default() += w.freq;
            last_tp = Some(tp);
          }
        } else {
          new_idxs.push(w.idxs[i]);
          last_tp = Some(tp);
          i += 1;
        }
      }
      w.idxs = new_idxs;
    }
    changes
  }

  fn step(&mut self) {

  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_bpe_merge() {
    let mut bpe = BPE::default();
    bpe.vocab.insert(0, Arc::from(&b"a"[..]));
    bpe.vocab.insert(1, Arc::from(&b"b"[..]));
    bpe.vocab.insert(2, Arc::from(&b"c"[..]));
    bpe.vocab.insert(3, Arc::from(&b"d"[..]));
    bpe.words.push(PreToken {
      src: Arc::from(&b"abcd"[..]),
      idxs: vec![0, 1, 2, 3],
      freq: 4,
    });
    bpe.words.push(PreToken {
      src: Arc::from(&b"abcdbcd"[..]),
      idxs: vec![0, 1, 2, 3, 1, 2, 3],
      freq: 30,
    });
    bpe.words.push(PreToken {
      src: Arc::from(&b"abcbcd"[..]),
      idxs: vec![0, 1, 2, 1, 2, 3],
      freq: 200,
    });
    bpe.init_training();
    let merge = bpe.pre_merges.get(&(1, 2)).unwrap().clone();
    let changes = bpe.merge(&merge);
    println!("{:?}", changes);
    let answer = vec![
      ((0, 1), -234),
      ((0, Idx::MAX), 234),
      ((1, 2), -464),
      ((2, 1), -400),
      ((2, 3), -264),
      ((2, Idx::MAX), 200),
      ((3, 1), -30),
      ((3, Idx::MAX), 30),
      ((Idx::MAX, 1), 200),
      ((Idx::MAX, 3), 264),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    assert_eq!(changes, answer);
  }
}
