use std::collections::BTreeMap;

use crate::{MyError, MyResult};

use super::*;

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

#[cfg(test)]
mod tests {
  use super::*;

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
