use std::{collections::{BTreeMap, BTreeSet, btree_map}, sync::Arc};

pub mod trainer;
pub mod encoder;

pub use trainer::BpeTrainer;
pub use encoder::BpeEncoder;


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
