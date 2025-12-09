use std::{collections::{BTreeMap, HashMap}, sync::atomic::AtomicU64};

use super::*;

#[derive(Debug, Default)]
pub struct BpeTrainer<C = u8> {
  pub start_vocab_idx: AtomicU64,
  pub vocab: BTreeMap<Idx, Word<C>>,
  pub merges: Vec<Merge<C, Idx>>,
  pub pre_merges: HashMap<(Idx, Idx), Merge<C, Idx>>,
  pub words: Vec<PreToken<C, Idx>>,
}

impl BpeTrainer<u8> {
  pub fn from_words<I: IntoIterator<Item = (String, Freq)>>(words: I, special_tokens: &[String]) -> Self {
    let vocab_start_idx = special_tokens.len() as Idx;
    let mut tokens = Vec::new();
    for (w, freq) in words {
      if special_tokens.contains(&w) {
        continue;
      }
      let idxs = w.bytes().map(|b| b as Idx + vocab_start_idx).collect::<Vec<_>>();
      let pre_token = PreToken {
        src: w.to_word(),
        idxs,
        freq,
      };
      tokens.push(pre_token);
    }
    let mut bpe = BpeTrainer::new(tokens);
    bpe._set_vocab_idx(0);
    bpe._vocab_insert_special_tokens(special_tokens);
    bpe._vocab_insert_all_single_byte();
    bpe
  }

  pub fn _vocab_insert_special_tokens(&mut self, special_tokens: &[String]) -> Idx {
    let length = special_tokens.len();
    let start_idx = self.start_vocab_idx.fetch_add(length as u64, std::sync::atomic::Ordering::AcqRel) as Idx;
    let vocab = &mut self.vocab;
    for (i, token) in special_tokens.into_iter().enumerate() {
      vocab.insert(i as Idx + start_idx, token.as_str().to_word());
    }
    start_idx + length as Idx
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
      pre_merges: HashMap::new(),
      words,
    }
  }

  pub fn init_training(&mut self) {
    debug!("Initializing BPE training with {} words", self.words.len());
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
    self._metrics();
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
      .max_by_key(|m| (m.data.freq, &m.content))
      .cloned()
  }

  fn _get_largest_merge2(&self) -> Option<Merge<C, Idx>> where C: Ord + Send + Sync + 'static {
    use rayon::prelude::*;
    self
      .pre_merges
      .par_iter()
      .map(|(_, m)| m)
      .max_by_key(|m| (m.data.freq, &m.content))
      .cloned()
  }

  pub fn step(&mut self) -> Option<Idx> where C: Ord + Send + Sync + 'static {
    // find the most frequent merge,
    // if the frequency is the same, choose the lexicographically largest one.
    let merge = if self.pre_merges.len() < 100_000 {
      self._get_largest_merge()?
    } else {
      self._get_largest_merge2()?
    };
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
    if (target_idx + 1) % 100 == 0 {
      self._metrics();
    }
    Some(target_idx)
  }

  pub fn finish(self) -> BpeEncoder<C> where C: Ord + Cachable {
    let merges = self.merges
      .into_iter()
      .map(|m| (m.tp, m.target.unwrap()))
      .collect();
    BpeEncoder::new(self.vocab, merges)
  }

  pub fn _metrics(&self) {
    metrics::counter!("bpe_trainer.vocab_size").absolute(self.vocab.len() as u64);
    metrics::gauge!("bpe_trainer.pre_merges_count").set(self.pre_merges.len() as f64);
    metrics::gauge!("bpe_trainer.words_count").set(self.words.len() as f64);
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
      ("a", "b", MergeData::new(-235).add_occurs_in([0, 1])),
      ("a", "bc", MergeData::new(235).add_occurs_in([0, 1, 2])),
      ("b", "c", MergeData::new(-465).add_occurs_in([0, 1, 2])),
      ("c", "b", MergeData::new(-200).add_occurs_in([2])),
      ("c", "d", MergeData::new(-265).add_occurs_in([0, 1, 2])),
      ("d", "b", MergeData::new(-30).add_occurs_in([1])),
      ("d", "bc", MergeData::new(30).add_occurs_in([1])),
      ("bc", "b", MergeData::new(0).add_occurs_in([2])),
      ("bc", "d", MergeData::new(265).add_occurs_in([0, 1, 2])),
      ("bc", "bc", MergeData::new(200).add_occurs_in([2])),
    ])]);

    _test_bpe_merge(&[("wherever", 10)],
    &[(("h", "e"), vec![
      ("e", "r", MergeData::new(-10).add_occurs_in([])),
      ("h", "e", MergeData::new(-10).add_occurs_in([0])),
      ("w", "h", MergeData::new(-10).add_occurs_in([0])),
      ("w", "he", MergeData::new(10).add_occurs_in([0])),
      ("he", "r", MergeData::new(10).add_occurs_in([0])),
    ])]);

    _test_bpe_merge(&[("aaa", 10), ("aaaa", 1)],
    &[(("a", "a"), vec![
      ("a", "a", MergeData::new(-23).add_occurs_in([0, 1])),
      ("aa", "a", MergeData::new(10).add_occurs_in([0, 1])),
      ("aa", "aa", MergeData::new(1).add_occurs_in([1])),
    ])]);
  }

  #[test]
  fn test_bpe_step() {
    let mut bpe = BpeTrainer::from_words(vec![
      ("ababc".to_string(), 5),
      ("ababcbabc".to_string(), 30),
      ("abcbabcab".to_string(), 200),
    ], &vec![]);
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
    // const NAME: &str = "TinyStoriesV2-GPT4-train";
    let input = std::fs::read_to_string(format!("fixtures/_words.{NAME}.json")).unwrap();
    let words: BTreeMap<String, Freq> = serde_json::from_str(&input).unwrap();
    let mut bpe = BpeTrainer::from_words(words, &vec!["<|endoftext|>".to_string()]);
    bpe.init_training();
    let vocab_size = match NAME {
      "tinystories_sample_5M" => 2000,
      _ => 10000,
    };
    while bpe.vocab.len() < vocab_size {
      bpe.step().unwrap();
      // let m = &bpe.merges.last().unwrap();
      // println!("{} {} => {}", _printable(&m.content.0), _printable(&m.content.1), m.data.freq);
    }
    std::fs::create_dir_all("out").ok();
    bpe.save_vocab_json(std::fs::File::create(format!("out/vocab.{NAME}.json")).unwrap()).unwrap();
    bpe.save_merges_txt(std::fs::File::create(format!("out/merges.{NAME}.txt")).unwrap()).unwrap();

    let merges_txt = std::fs::read_to_string(format!("out/merges.{NAME}.txt")).unwrap();
    let merges_expect_txt = std::fs::read_to_string(format!("fixtures/merges.{NAME}.txt")).unwrap();
    assert_eq!(merges_txt, merges_expect_txt);
  }
}
