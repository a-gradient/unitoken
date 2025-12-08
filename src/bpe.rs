use std::{collections::{BTreeMap, BTreeSet, btree_map}, fmt::Debug, sync::{Arc, atomic::AtomicU64}};

use lazy_static::lazy_static;
use ordermap::OrderMap;

use crate::{MyError, MyResult};

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
pub struct BpeTrainer<C = u8> {
  pub start_vocab_idx: AtomicU64,
  pub vocab: BTreeMap<Idx, Word<C>>,
  pub merges: Vec<Merge<C, Idx>>,
  pub pre_merges: BTreeMap<(Idx, Idx), Merge<C, Idx>>,
  pub words: Vec<PreToken<C, Idx>>,
}

impl BpeTrainer<u8> {
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
    let mut bpe = BpeTrainer::new(tokens);
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

impl<C: Clone> BpeTrainer<C>
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

  pub fn init_training(&mut self) {
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

  fn update_pre_merges(&mut self, merge: &Merge<C, Idx>, changes: BTreeMap<(Idx, Idx), MergeData>) {
    _update_merge_map(&mut self.pre_merges, merge, changes, Some(&self.vocab));
  }

  fn merge(&mut self, merge: &Merge<C, Idx>, target_idx: Idx) -> BTreeMap<(Idx, Idx), MergeData> {
    _merge(&mut self.words, merge, target_idx)
  }

  fn _get_largest_merge(&self) -> Option<Merge<C, Idx>> where C: Ord {
    self
      .pre_merges
      .values()
      .max_by_key(|m| (m.data.freq, m.content.clone()))
      .cloned()
  }

  pub fn step(&mut self) -> Option<Idx> where C: Ord {
    // find the most frequent merge,
    // if the frequency is the same, choose the lexicographically largest one.
    let merge = self._get_largest_merge()?;
    let target_idx = self._add_vocab_idx();
    let changes = self.merge(&merge, target_idx);
    // println!("Merge {:?} (freq={}) into idx {}", merge.tp, merge.data.freq, target_idx);
    let merge = merge.with_target(target_idx);
    let merged_word = merge.merged_content();
    self.vocab.insert(target_idx, merged_word);
    assert_eq!(-changes.get(&merge.tp).map(|i| i.freq).unwrap_or(0), merge.data.freq);
    self.update_pre_merges(&merge, changes);
    self.pre_merges.remove(&merge.tp);
    self.merges.push(merge);
    Some(target_idx)
  }

  pub fn finish(self) -> BpeEncoder<C> where C: Ord {
    let merges = self.merges
      .into_iter()
      .map(|m| (m.tp, m.target.unwrap()))
      .collect();
    BpeEncoder::new(self.vocab, merges)
  }
}

pub struct BpeEncoder<C = u8> {
  pub vocab_bytes: BTreeMap<C, Idx>,
  pub vocab_rev: BTreeMap<Word<C>, Idx>,
  pub vocab: BTreeMap<Idx, Word<C>>,
  pub merges: Vec<((Idx, Idx), Idx)>,
  pub pre_merge_map: BTreeMap<(Idx, Idx), Merge<C, Idx>>,
}

impl BpeEncoder<u8> {
  fn _load_vocab<R: std::io::Read>(reader: R) -> MyResult<BTreeMap<Word<u8>, Idx>> {
    let input: BTreeMap<String, u64> = serde_json::from_reader(reader)?;
    input.into_iter().map(|(s, i)| {
      let w = _from_printable(&s).map_err(|e| MyError::InvalidPrintableChar(e))?;
      Ok((w, i as Idx))
    }).collect()
  }

  fn _load_merges<R: std::io::Read>(mut reader: R, vocab: &BTreeMap<Word<u8>, Idx>) -> MyResult<Vec<Merge<u8, Idx>>> {
    let mut result = Vec::new();
    let mut input = String::new();
    reader.read_to_string(&mut input)?;
    fn get_kv(vocab: &BTreeMap<Word<u8>, Idx>, s: &str) -> MyResult<(Idx, Word<u8>)> {
      let w = _from_printable(s).map_err(|e| MyError::InvalidPrintableChar(e))?;
      Ok((*vocab.get(&w).ok_or_else(|| MyError::Oov(w.display()))?, w))
    }
    for (i, line) in input.lines().enumerate() {
      if line.trim().is_empty() {
        continue;
      }
      let mut main = line;
      let mut freq = 0;
      if line.contains(" => ") {
        let split = line.rsplitn(2, " => ").collect::<Vec<_>>();
        main = split.last().unwrap();
        if split.len() > 1 {
          freq = split[0].trim().parse().unwrap_or_default();
        }
      }
      let parts = main.trim().split_whitespace().collect::<Vec<_>>();
      if parts.len() != 2 {
        return Err(MyError::MergeTxt("main parts is not 2", i))
      }
      let (a_idx, a) = get_kv(vocab, parts[0])?;
      let (b_idx, b) = get_kv(vocab, parts[1])?;
      let merged = format!("{}{}", parts[0], parts[1]);
      let (m_idx, _) = get_kv(vocab, &merged)?;
      let mut merge = Merge::new((a_idx, b_idx), (a, b)).with_target(m_idx);
      merge.data.freq = freq;
      result.push(merge);
    }
    Ok(result)
  }
}

impl<C: Ord + Clone> BpeEncoder<C>
where
  Word<C>: WordExt
{
  pub fn new(vocab: BTreeMap<Idx, Word<C>>, merges: Vec<((Idx, Idx), Idx)>) -> Self {
    let vocab_rev = vocab
      .iter()
      .map(|(k, v)| (v.clone(), *k))
      .collect();
    let vocab_bytes = vocab
      .iter()
      .filter_map(|(k, v)| {
        if v.len() == 1 {
          Some((v[0].clone(), *k))
        } else {
          None
        }
      })
      .collect();
    let pre_merge_map = merges.iter().copied().map(|(tp, target)| {
      (tp, Merge::new(tp, (
        vocab.get(&tp.0).unwrap().clone(),
        vocab.get(&tp.1).unwrap().clone(),
      )).with_target(target))
    }).collect::<BTreeMap<_, _>>();
    Self {
      vocab_bytes,
      vocab_rev,
      vocab,
      merges,
      pre_merge_map,
    }
  }

  pub fn _pretoken(&self, word: Word<C>, freq: Freq) -> MyResult<PreToken<C, Idx>> {
    let idxs = word.iter()
      .map(|c| self.vocab_bytes.get(c).copied().ok_or_else(|| crate::MyError::OovBytes(vec![c.clone()].to_word().display())))
      .collect::<Result<_, _>>()?;
    Ok(PreToken { src: word, idxs, freq })
  }

  pub fn encode_words(&self, input: &[Word<C>]) -> MyResult<Vec<Word<Idx>>> {
    let mut  words = input
      .iter()
      .map(|w| self._pretoken(w.clone(), 1))
      .collect::<Result<Vec<_>, _>>()?;
    let mut pre_merges = self.pre_merge_map.clone();

    // init
    for (i, word) in words.iter().enumerate() {
      for (j1, j2) in word.idxs.iter().copied().zip(word.idxs.iter().skip(1).copied()) {
        let tp = (j1, j2);
        if let Some(merge) = pre_merges.get_mut(&tp) {
          merge.add(i as u64, 1);
        }
      }
    }

    for (tp, target) in &self.merges {
      let Some(merge) = pre_merges.remove(&tp) else {
        continue;
      };
      let changes = _merge(&mut words, &merge, *target);
      _update_merge_map(&mut pre_merges, &merge, changes, None);
    }

    Ok(words.into_iter().map(|i| i.idxs.to_word()).collect())
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
    let mut i = 0;
    let mut next_char = || {
      i += 1;
      char::from_u32(i + 255).unwrap()
    };
    for b in 0u32..=255 {
      map.entry(b as u8).or_insert_with(&mut next_char);
    }
    map
  };
  static ref PRINTABLE_REV: BTreeMap<char, u8> = {
    let mut map = BTreeMap::new();
    for (b, ch) in PRINTABLE.iter() {
      map.insert(*ch, *b);
    }
    map
  };
}

fn _printable(w: &Word<u8>) -> String {
  w.iter()
    .map(|b| PRINTABLE.get(b).copied().unwrap_or('.'))
    .collect()
}

fn _from_printable(s: &str) -> Result<Word<u8>, char> {
  let bytes = s
    .chars()
    .map(|ch| PRINTABLE_REV.get(&ch).copied().ok_or(ch))
    .collect::<Result<Vec<_>, _>>()?;
  Ok(Arc::from(bytes.into_boxed_slice()))
}

fn _merge<C, I>(words: &mut Vec<PreToken<C, I>>, merge: &Merge<C, I>, target_idx: I) -> BTreeMap<(I, I), MergeData>
where
  I: Ord + Copy,
  C: Clone,
{
  // all tp with target_idx MUST be positive, so that occurs_in should be added.
  // while tp without target_idx MUST be negative, and occurs_in should be removed.
  let mut changes = BTreeMap::<(I, I), MergeData>::new();
  for k in merge.data.occurs_in.iter().copied() {
    let w = &mut words[k as usize];
    // local freq tracks the frequency changes within this word.
    let mut local_freq = BTreeMap::<(I, I), Freq>::new();
    let w_idx = &w.idxs;
    let w_freq = w.freq;
    let mut new_idxs = Vec::with_capacity(w_idx.len());
    let mut i = 0;
    let mut last_tp: Option<(I, I)> = None;
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
          // if i >= w_idx.len(), loop is end, and last_tp never reads
          // last_tp = Some(new_tp);
        }
        // deal with right neighbor, notice i+=2 above
        // e.g. in "abcd", when merging "b" and "c",
        // old_tp = ("c", "d"), new_tp = ("bc", "d")
        if i < w_idx.len() {
          let old_tp = (tp.1, w_idx[i]);
          let new_tp = (target_idx, old_tp.1);
          changes.entry(old_tp).or_default().freq -= w_freq;
          changes.entry(new_tp).or_default().freq += w_freq;
          // old_tp is not increased, so that it should not be decreased
          *local_freq.entry(old_tp).or_default() -= 0;
          // when combining "b" and "c" in "bcbc",
          // new_tp=("bc", "b") would be false positive occurs_in
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

fn _update_merge_map<C, I>(merge_map: &mut BTreeMap<(I, I), Merge<C, I>>, merge: &Merge<C, I>, changes: BTreeMap<(I, I), MergeData>, vocab: Option<&BTreeMap<I, Word<C>>>)
where
  I: Ord + Copy,
  C: Clone,
  Word<C>: WordExt,
{
  for (tp, data) in changes {
    if tp == merge.tp {
      continue;
    }
    if data.freq == 0 {
      continue;
    }
    let entry = merge_map.entry(tp);
    let entry = match entry {
      btree_map::Entry::Occupied(e) => e.into_mut(),
      btree_map::Entry::Vacant(e) => {
        if let Some(vocab) = vocab {
          let content = (
            vocab.get(&tp.0).unwrap().clone(),
            vocab.get(&tp.1).unwrap().clone(),
          );
          e.insert(Merge::new(tp, content))
        } else {
          continue;
        }
      }
    };
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
    fn lookup(bpe: &BpeTrainer, s: &str) -> Option<Idx> {
      bpe.vocab.iter().find_map(|(i, w)| {
        if w.as_ref() == s.as_bytes() {
          Some(*i)
        } else {
          None
        }
      })
    }
    fn display(bpe: &BpeTrainer, changes: &BTreeMap<(u32, u32), MergeData>) -> String {
      let mut parts = Vec::new();
      let target = ("__target__").to_word();
      for (tp, data) in changes.iter() {
        let left = bpe.vocab.get(&tp.0).unwrap_or(&target).display();
        let right = bpe.vocab.get(&tp.1).unwrap_or(&target).display();
        parts.push(format!("({:?}, {:?}, MergeData::new({}).occurs_in({:?}))", left, right, data.freq, data.occurs_in_vec()));
      }
      format!("{{\n  {}\n}}", parts.join(",\n  "))
    }

    let mut bpe = BpeTrainer::default();
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

    _test_bpe_merge(&[("aaa", 10), ("aaaa", 1)],
    &[(("a", "a"), vec![
      ("a", "a", MergeData::new(-23).occurs_in([0, 1])),
      ("aa", "a", MergeData::new(10).occurs_in([0, 1])),
      ("aa", "aa", MergeData::new(1).occurs_in([1])),
    ])]);
  }

  #[test]
  fn test_bpe_step() {
    let mut bpe = BpeTrainer::from_words(vec![
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
    let mut bpe = BpeTrainer::from_words(words);
    bpe.init_training();
    while bpe.vocab.len() < 1999 {
      bpe.step().unwrap();
      // let m = &bpe.merges.last().unwrap();
      // println!("{} {} => {}", _printable(&m.content.0), _printable(&m.content.1), m.data.freq);
    }
    std::fs::create_dir_all("out").ok();
    bpe.save_vocab_json(std::fs::File::create(format!("out/vocab.{NAME}.json")).unwrap()).unwrap();
    bpe.save_merges_txt(std::fs::File::create(format!("out/merges.{NAME}.txt")).unwrap()).unwrap();
  }

  #[test]
  fn test_bpe_encode_words() {
    const NAME: &str = "tinystories_sample_5M";
    let input: BTreeMap<String, Freq> = serde_json::from_str(&std::fs::read_to_string(format!("fixtures/{NAME}_words.json")).unwrap()).unwrap();
    let input = input.into_iter().map(|(k, _)| k.to_word()).collect::<Vec<_>>();
    let vocab = BpeEncoder::_load_vocab(std::fs::File::open(format!("fixtures/vocab.{NAME}.json")).unwrap()).unwrap();
    let merges = BpeEncoder::_load_merges(std::fs::File::open(format!("fixtures/merges.{NAME}.txt")).unwrap(), &vocab).unwrap();
    let vocab = vocab.into_iter().map(|(k, v)| (v, k)).collect();
    let merges = merges.into_iter().map(|m| (m.tp, m.target.unwrap())).collect();
    let bpe = BpeEncoder::new(vocab, merges);
    let result = bpe.encode_words(&input).unwrap();
    assert_eq!(result.len(), input.len());
  }
}
