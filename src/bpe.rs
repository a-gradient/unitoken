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
  pub data: MergeData,
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
  pub fn occurs_in<I: IntoIterator<Item = u64>>(self, iter: I) -> Self {
    Self {
      occurs_in: iter.into_iter().collect(),
      freq: self.freq,
    }
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

#[derive(Debug, Default)]
pub struct BPE<C = u8> {
  pub vocab: BTreeMap<Idx, Word<C>>,
  pub merges: Vec<Merge<C, Idx>>,
  pub pre_merges: BTreeMap<(Idx, Idx), Merge<C, Idx>>,
  pub words: Vec<PreToken<C, Idx>>,
}

impl<C: Clone> BPE<C> {
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

  fn merge(&mut self, merge: &Merge<C, Idx>, target_idx: Idx) -> BTreeMap<(Idx, Idx), MergeData> {
    // all tp with target_idx MUST be positive, so that occurs_in should be added.
    // while tp without target_idx MUST be negative, and occurs_in should be removed.
    let mut changes = BTreeMap::<(Idx, Idx), MergeData>::new();
    for (k, w) in self.words.iter_mut().enumerate() {
      // local freq tracks the frequency changes within this word.
      let mut local_freq = BTreeMap::<(Idx, Idx), Freq>::new();
      let mut new_idxs = Vec::with_capacity(w.idxs.len());
      let mut i = 0;
      let mut last_tp: Option<(Idx, Idx)> = None;
      while i < w.idxs.len() - 1 {
        let tp = (w.idxs[i], w.idxs[i + 1]);
        *local_freq.entry(tp).or_default() += 1;
        if tp == merge.tp {
          new_idxs.push(target_idx);
          i += 2;
          changes.entry(tp).or_default().freq -= w.freq;
          *local_freq.entry(tp).or_default() -= 1;
          if let Some(tp) = last_tp {
            let new_tp = (tp.0, target_idx);
            changes.entry(tp).or_default().freq -= w.freq;
            changes.entry(new_tp).or_default().freq += w.freq;
            *local_freq.entry(tp).or_default() -= 1;
            *local_freq.entry(new_tp).or_default() -= 1;
          }
          if i < w.idxs.len() {
            let tp = (w.idxs[i - 1], w.idxs[i]);
            let new_tp = (target_idx, tp.1);
            changes.entry(tp).or_default().freq -= w.freq;
            changes.entry(new_tp).or_default().freq += w.freq;
            *local_freq.entry(tp).or_default() -= 1;
            *local_freq.entry(new_tp).or_default() -= 1;
            last_tp = Some(tp);
          }
        } else {
          new_idxs.push(w.idxs[i]);
          last_tp = Some(tp);
          i += 1;
        }
      }

      local_freq.iter().filter(|(_, i)| **i <= 0).for_each(|(tp, _)| {
        changes.entry(*tp).and_modify(|d| { d.occurs_in.insert(k as _); });
      });

      w.idxs = new_idxs;
    }
    changes
  }

  fn step(&mut self) {
    let Some(merge) = self
      .pre_merges
      .values()
      .max_by_key(|m| m.data.freq)
      .cloned() else {
      return
    };
    let target_idx = self.vocab.len() as Idx;
    let changes = self.merge(&merge, target_idx);
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
      freq: 5,
    });
    bpe.words.push(PreToken {
      src: Arc::from(&b"abcdbcd"[..]),
      idxs: vec![0, 1, 2, 3, 1, 2, 3],
      freq: 30,
    });
    bpe.words.push(PreToken {
      src: Arc::from(&b"abcbcdab"[..]),
      idxs: vec![0, 1, 2, 1, 2, 3, 0, 1],
      freq: 200,
    });
    bpe.init_training();
    let tp = (1, 2);
    let target = Idx::MAX;
    let merge = bpe.pre_merges.get(&tp).unwrap().clone();
    let changes = bpe.merge(&merge, target);
    println!("{:?}", changes);
    let answer = vec![
      ((0, 1), MergeData::new(-235).occurs_in([0, 1])),
      ((0, target), MergeData::new(235).occurs_in([0, 1, 2])),
      ((1, 2), MergeData::new(-465).occurs_in([0, 1, 2])),
      ((2, 1), MergeData::new(-400).occurs_in([2])),
      ((2, 3), MergeData::new(-265).occurs_in([0, 1, 2])),
      ((2, target), MergeData::new(200).occurs_in([2])),
      ((3, 1), MergeData::new(-30).occurs_in([1])),
      ((3, target), MergeData::new(30).occurs_in([1])),
      ((target, 1), MergeData::new(200).occurs_in([2])),
      ((target, 3), MergeData::new(265).occurs_in([0, 1, 2])),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    assert_eq!(changes, answer);
    assert_eq!(merge.data.freq, -answer.get(&tp).cloned().unwrap().freq);
  }
}
