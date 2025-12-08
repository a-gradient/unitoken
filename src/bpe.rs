use std::{collections::{BTreeMap, BTreeSet}, fmt::Debug, sync::{Arc, atomic::AtomicU64}};

use lazy_static::lazy_static;
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
  pub fn occurs_in<I: IntoIterator<Item = u64>>(self, iter: I) -> Self {
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

#[derive(Debug, Default)]
pub struct BPE<C = u8> {
  pub start_vocab_idx: AtomicU64,
  pub vocab: BTreeMap<Idx, Word<C>>,
  pub merges: Vec<Merge<C, Idx>>,
  pub pre_merges: BTreeMap<(Idx, Idx), Merge<C, Idx>>,
  pub words: Vec<PreToken<C, Idx>>,
}

impl BPE<u8> {
  pub fn from_words<I: IntoIterator<Item = (String, Freq)>>(words: I) -> Self {
    let vocab_start_idx = 0;
    let mut tokens = Vec::new();
    for (w, freq) in words {
      let idxs = w.bytes().map(|b| b as Idx + vocab_start_idx).collect::<Vec<_>>();
      let pre_token = PreToken {
        src: w.to_word(),
        idxs,
        freq,
      };
      tokens.push(pre_token);
    }
    let mut bpe = BPE::new(tokens);
    bpe._set_vocab_idx(vocab_start_idx);
    bpe._vocab_insert_all_single_byte();
    bpe
  }

  pub fn _vocab_insert_all_single_byte(&mut self) -> Idx {
    let start_idx = self.start_vocab_idx.fetch_add(256, std::sync::atomic::Ordering::AcqRel) as Idx;
    let vocab = &mut self.vocab;
    for i in 0u8..=255 {
      vocab.insert(i as Idx + start_idx, [i].to_word());
    }
    start_idx + 256
  }

  pub fn save_vocab_json<W: std::io::Write>(&self, mut w: W) -> Result<(), std::io::Error> {
    let mut map = OrderMap::new();
    for (idx, word) in self.vocab.iter() {
      let s = _printable(word);
      map.insert(s, idx);
    }
    let json = serde_json::to_string_pretty(&map).unwrap();
    write!(w, "{}", json)
  }

  pub fn save_merges_txt<W: std::io::Write>(&self, mut w: W) -> Result<(), std::io::Error> {
    for merge in self.merges.iter() {
      let left = _printable(&merge.content.0);
      let right = _printable(&merge.content.1);
      writeln!(w, "{} {} => {}", left, right, merge.data.freq)?;
    }
    Ok(())
  }
}

impl<C: Clone> BPE<C>
where
  Word<C>: WordExt
{
  pub fn new(words: Vec<PreToken<C, Idx>>) -> Self {
    Self {
      start_vocab_idx: AtomicU64::new(0),
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

  fn _set_vocab_idx(&mut self, start_idx: Idx) {
    self.start_vocab_idx.store(start_idx as u64, std::sync::atomic::Ordering::Release);
  }

  fn _add_vocab_idx(&self) -> Idx {
    self.start_vocab_idx.fetch_add(1, std::sync::atomic::Ordering::AcqRel) as Idx
  }

  fn merge(&mut self, merge: &Merge<C, Idx>, target_idx: Idx) -> BTreeMap<(Idx, Idx), MergeData> {
    // all tp with target_idx MUST be positive, so that occurs_in should be added.
    // while tp without target_idx MUST be negative, and occurs_in should be removed.
    let mut changes = BTreeMap::<(Idx, Idx), MergeData>::new();
    for k in merge.data.occurs_in.iter().copied() {
      let w = &mut self.words[k as usize];
      // local freq tracks the frequency changes within this word.
      let mut local_freq = BTreeMap::<(Idx, Idx), Freq>::new();
      let w_idx = &w.idxs;
      let w_freq = w.freq;
      let mut new_idxs = Vec::with_capacity(w_idx.len());
      let mut i = 0;
      let mut last_tp: Option<(Idx, Idx)> = None;
      while i + 1 < w_idx.len() {
        let tp = (w_idx[i], w_idx[i + 1]);
        *local_freq.entry(tp).or_default() += 1;
        if tp == merge.tp {
          new_idxs.push(target_idx);
          i += 2;
          changes.entry(tp).or_default().freq -= w_freq;
          *local_freq.entry(tp).or_default() -= 1;
          // deal with left neighbor,
          // e.g. in "abcd", when merging "b" and "c",
          // old_tp = ("a", "b"), new_tp = ("a", "bc")
          if let Some(old_tp) = last_tp {
            let new_tp = (old_tp.0, target_idx);
            changes.entry(old_tp).or_default().freq -= w_freq;
            changes.entry(new_tp).or_default().freq += w_freq;
            *local_freq.entry(old_tp).or_default() -= 1;
            *local_freq.entry(new_tp).or_default() -= 1;
            last_tp = Some(new_tp);
          }
          // deal with right neighbor,
          // e.g. in "abcd", when merging "b" and "c",
          // old_tp = ("c", "d"), new_tp = ("bc", "d")
          if i < w_idx.len() {
            let old_tp = (tp.1, w_idx[i]);
            let new_tp = (target_idx, old_tp.1);
            changes.entry(old_tp).or_default().freq -= w_freq;
            changes.entry(new_tp).or_default().freq += w_freq;
            // old_tp is not increased, so that it should not be decreased
            *local_freq.entry(old_tp).or_default() -= 0;
            *local_freq.entry(new_tp).or_default() -= 1;
            last_tp = Some(new_tp);
          }
        } else {
          new_idxs.push(w_idx[i]);
          last_tp = Some(tp);
          i += 1;
        }
      }
      if i < w_idx.len() {
        new_idxs.push(w_idx[i]);
      }

      local_freq.iter().filter(|(_, i)| **i <= 0).for_each(|(tp, _)| {
        changes.entry(*tp).and_modify(|d| { d.occurs_in.insert(k as _); });
      });

      w.idxs = new_idxs;
    }
    changes
  }

  fn step(&mut self) -> Option<Idx> where C: Ord {
    // find the most frequent merge,
    // if the frequency is the same, choose the lexicographically largest one.
    let Some(merge) = self
      .pre_merges
      .values()
      .max_by_key(|m| (m.data.freq, m.content.clone()))
      .cloned() else {
      return None;
    };
    let target_idx = self._add_vocab_idx();
    let changes = self.merge(&merge, target_idx);
    // println!("Merge {:?} (freq={}) into idx {}", merge.tp, merge.data.freq, target_idx);
    let merge = merge.with_target(target_idx);
    let merged_word = merge.merged_content();
    self.vocab.insert(target_idx, merged_word);
    for (tp, data) in changes {
      if tp == merge.tp {
        assert_eq!(-data.freq, merge.data.freq);
        continue;
      }
      if data.freq == 0 {
        continue;
      }
      let entry = self.pre_merges.entry(tp).or_insert_with(|| {
        let content = (
          self.vocab.get(&tp.0).unwrap().clone(),
          self.vocab.get(&tp.1).unwrap().clone(),
        );
        Merge::new(tp, content)
      });
      // println!("  Change {:?} {:?} {:?}: freq {} -> {}", tp, entry.content.0.display(), entry.content.1.display(), entry.data.freq, entry.data.freq + data.freq);
      entry.data.freq += data.freq;
      if data.freq > 0 {
        entry.data.occurs_in.extend(data.occurs_in);
      } else {
        data.occurs_in.iter().for_each(|doc_id| {
          entry.data.occurs_in.remove(doc_id);
        });
      }
    }
    self.pre_merges.remove(&merge.tp);
    self.merges.push(merge);
    Some(target_idx)
  }
}

trait ToWord<C> {
  fn to_word(self) -> Word<C>;
}

impl<C> ToWord<C> for Vec<C> {
  fn to_word(self) -> Word<C> {
    Arc::from(self.into_boxed_slice())
  }
}

impl ToWord<u8> for &[u8] {
  fn to_word(self) -> Word<u8> {
    Arc::from(self.to_owned().into_boxed_slice())
  }
}

impl ToWord<u8> for &str {
  fn to_word(self) -> Word<u8> {
    Arc::from(self.as_bytes().to_owned().into_boxed_slice())
  }
}

pub trait WordExt {
  fn display(&self) -> String;
}

impl WordExt for Word<u8> {
  fn display(&self) -> String {
    _printable(self)
  }
}

impl WordExt for Word<Character> {
  fn display(&self) -> String {
    self
      .iter()
      .map(|c| match c {
        Character::Unicode(ch) => *ch,
        Character::Byte(b) => PRINTABLE.get(b).copied().unwrap_or('.'),
      })
      .collect()
  }
}

lazy_static! {
  static ref PRINTABLE: BTreeMap<u8, char> = {
    let mut map = BTreeMap::new();
    for range in [33u8..=126, 161..=172, 174..=255].iter() {
      for b in range.clone() {
        map.insert(b as u8, b as char);
      }
    }
    for b in 0u32..=255 {
      map.entry(b as u8).or_insert(char::from_u32(b + 256).unwrap());
    }
    map
  };
}

fn _printable(w: &Word<u8>) -> String {
  w.iter()
    .map(|b| PRINTABLE.get(b).copied().unwrap_or('.'))
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  fn _test_bpe_merge(pretokens: &[(&str, Freq)], merges: &[((&str, &str), Vec<(&str, &str, MergeData)>)]) {
    fn pretoken(s: &str, freq: Freq) -> PreToken<u8, Idx> {
      let idxs = s.bytes().map(|b| b as Idx - 'a' as Idx).collect::<Vec<_>>();
      PreToken {
        src: s.to_word(),
        idxs,
        freq,
      }
    }
    fn lookup(bpe: &BPE, s: &str) -> Option<Idx> {
      bpe.vocab.iter().find_map(|(i, w)| {
        if w.as_ref() == s.as_bytes() {
          Some(*i)
        } else {
          None
        }
      })
    }
    fn display(bpe: &BPE, changes: &BTreeMap<(u32, u32), MergeData>) -> String {
      let mut parts = Vec::new();
      let target = ("__target__").to_word();
      for (tp, data) in changes.iter() {
        let left = bpe.vocab.get(&tp.0).unwrap_or(&target).display();
        let right = bpe.vocab.get(&tp.1).unwrap_or(&target).display();
        parts.push(format!("({:?}, {:?}, MergeData::new({}).occurs_in({:?}))", left, right, data.freq, data.occurs_in_vec()));
      }
      format!("{{\n  {}\n}}", parts.join(",\n  "))
    }

    let mut bpe = BPE::default();
    bpe.vocab.extend(
      ('a' ..= 'z').enumerate().map(|(i, c)| (i as Idx, c.to_string().to_word()))
    );
    bpe._set_vocab_idx(100);
    bpe.words.extend(
      pretokens.iter().map(|(s, f)| pretoken(s, *f))
    );
    bpe.init_training();
    for (m, expected) in merges {
      let merge_tp = (
        lookup(&bpe, m.0).unwrap(), lookup(&bpe, m.1).unwrap()
      );
      let merge = bpe.pre_merges.get(&merge_tp).unwrap().clone();
      let target = bpe._add_vocab_idx();
      let changes = bpe.merge(&merge, target);
      assert_eq!(merge.data.freq, -changes.get(&merge_tp).cloned().unwrap().freq);
      if expected.is_empty() {
        continue;
      }
      let expected = expected.into_iter().map(|(a, b, data)| {
        let tp_idx = (lookup(&bpe, a).unwrap_or(target), lookup(&bpe, b).unwrap_or(target));
        (tp_idx, data.clone())
      }).collect::<BTreeMap<_, _>>();
      assert_eq!(changes, expected, "\nExpected changes:\n{}\nActual changes:\n{}", display(&bpe, &expected), display(&bpe, &changes));
    }
  }

  #[test]
  fn test_bpe_merge() {
    _test_bpe_merge(&[("abcd", 5), ("abcdbcd", 30), ("abcbcdab", 200)], &[(("b", "c"), vec![
      ("a", "b", MergeData::new(-235).occurs_in([0, 1])),
      ("a", "bc", MergeData::new(235).occurs_in([0, 1, 2])),
      ("b", "c", MergeData::new(-465).occurs_in([0, 1, 2])),
      ("c", "b", MergeData::new(-200).occurs_in([2])),
      ("c", "d", MergeData::new(-265).occurs_in([0, 1, 2])),
      ("d", "b", MergeData::new(-30).occurs_in([1])),
      ("d", "bc", MergeData::new(30).occurs_in([1])),
      ("bc", "b", MergeData::new(0).occurs_in([2])),
      ("bc", "d", MergeData::new(265).occurs_in([0, 1, 2])),
      ("bc", "bc", MergeData::new(200).occurs_in([2])),
    ])]);

    _test_bpe_merge(&[("wherever", 10)],
    &[(("h", "e"), vec![
      ("e", "r", MergeData::new(-10).occurs_in([])),
      ("h", "e", MergeData::new(-10).occurs_in([0])),
      ("w", "h", MergeData::new(-10).occurs_in([0])),
      ("w", "he", MergeData::new(10).occurs_in([0])),
      ("he", "r", MergeData::new(10).occurs_in([0])),
    ])]);
  }

  #[test]
  fn test_bpe_step() {
    let mut bpe = BPE::from_words(vec![
      ("ababc".to_string(), 5),
      ("ababcbabc".to_string(), 30),
      ("abcbabcab".to_string(), 200),
    ]);
    bpe.init_training();
    for _ in 0..3 {
      bpe.step();
    }
    let result_vocab = bpe.vocab.into_iter().map(|(i, w)| (i, _printable(&w))).skip(256).collect::<Vec<_>>();
    assert_eq!(
      result_vocab,
      vec![
        (256, "ab".to_string()),
        (257, "abc".to_string()),
        (258, "babc".to_string()),
      ]
    );
    let result_merges = bpe.merges.into_iter().map(|m| {
      let left = _printable(&m.content.0);
      let right = _printable(&m.content.1);
      (left, right, m.data.freq)
    }).collect::<Vec<_>>();
    assert_eq!(
      result_merges,
      vec![
        ("a".to_string(), "b".to_string(), 700),
        ("ab".to_string(), "c".to_string(), 465),
        ("b".to_string(), "abc".to_string(), 230),
      ]
    );
  }

  #[test]
  fn test_bpe_from_words() {
    const NAME: &str = "tinystories_sample_5M";
    let input = std::fs::read_to_string(format!("fixtures/{NAME}_words.json")).unwrap();
    let words: BTreeMap<String, Freq> = serde_json::from_str(&input).unwrap();
    let mut bpe = BPE::from_words(words);
    bpe.init_training();
    while bpe.vocab.len() < 1000 {
      bpe.step().unwrap();
      // let m = &bpe.merges.last().unwrap();
      // println!("{} {} => {}", _printable(&m.content.0), _printable(&m.content.1), m.data.freq);
    }
    std::fs::create_dir_all("out").ok();
    bpe.save_vocab_json(std::fs::File::create(format!("out/vocab.{NAME}.json")).unwrap()).unwrap();
    bpe.save_merges_txt(std::fs::File::create(format!("out/merges.{NAME}.txt")).unwrap()).unwrap();
  }
}
