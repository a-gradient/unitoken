use std::{
  collections::{BTreeMap, HashMap},
  fs::File,
  io::{Read as _, Seek as _},
  path::Path,
};

use moka::sync::Cache;

use crate::{
  MyError, MyResult,
  pretokenizer::{RE, find_chunk_boundaries, get_words_from_file, pretokenizer_tokens, split_special_tokens},
};

use super::*;

pub struct BpeEncoder<C = u8> {
  pub vocab_bytes: BTreeMap<C, Idx>,
  pub vocab_rev: BTreeMap<Word<C>, Idx>,
  pub vocab: BTreeMap<Idx, Word<C>>,
  pub merges: Vec<((Idx, Idx), Idx)>,
  /// with freq represents rank, or `merge.data.freq=-i` for i-th merge.
  /// with [`occurs_in={0}`](MergeData::occurs_in), in order to handle first word in [`Self::_encode_word`].
  pub pre_merge_map: HashMap<(Idx, Idx), Merge<C, Idx>>,
  pub cache: Cache<Word<C>, Word<Idx>>,
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

impl<C: Ord + Clone + Cachable> BpeEncoder<C>
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
    let pre_merge_map = merges.iter().copied().enumerate().map(|(i, (tp, target))| {
      let mut merge = Merge::new(tp, (
        vocab.get(&tp.0).unwrap().clone(),
        vocab.get(&tp.1).unwrap().clone(),
      )).with_target(target);
      merge.add(0, -(i as Freq));
      (tp, merge)
    }).collect::<HashMap<_, _>>();
    let max_cap = vocab.len() as u64 * 3 / 2;
    Self {
      vocab_bytes,
      vocab_rev,
      vocab,
      merges,
      pre_merge_map,
      cache: Cache::new(max_cap),
    }
  }

  pub fn _pretoken(&self, word: Word<C>, freq: Freq) -> MyResult<PreToken<C, Idx>> {
    let idxs = word.iter()
      .map(|c| self.vocab_bytes.get(c).copied().ok_or_else(|| crate::MyError::OovBytes(vec![c.clone()].to_word().display())))
      .collect::<Result<_, _>>()?;
    Ok(PreToken { src: word, idxs, freq })
  }

  fn _new_pre_merge_map(&self) -> HashMap<(Idx, Idx), Merge<C, Idx>> {
    let mut pre_merges = self.pre_merge_map.clone();
    pre_merges.iter_mut().for_each(|i| {
      i.1.data.freq = 0;
      i.1.data.occurs_in.clear();
    });
    pre_merges
  }

  /// this would merge pairs in all [`Word`] in `input`,
  /// in the same order and similar precedure of trainer.
  ///
  /// this method would be useful if you would like to build cache,
  /// or have large number of words to be encode at one time.
  ///
  /// See [`Self::encode_words`] for cached version
  pub fn _encode_words(&self, input: &[Word<C>]) -> MyResult<Vec<Word<Idx>>> {
    if input.len() == 0 {
      return Ok(Vec::new());
    }
    let mut words = input
      .iter()
      .map(|w| self._pretoken(w.clone(), 1))
      .collect::<Result<Vec<_>, _>>()?;
    let mut pre_merges = self._new_pre_merge_map();

    // init
    for (i, word) in words.iter().enumerate() {
      for (j1, j2) in word.idxs.iter().copied().zip(word.idxs.iter().skip(1).copied()) {
        let tp = (j1, j2);
        if let Some(merge) = pre_merges.get_mut(&tp) {
          merge.add(i as u64, 1);
        }
      }
    }

    // merge
    for (tp, target) in &self.merges {
      let Some(merge) = pre_merges.remove(&tp) else {
        continue;
      };
      let changes = _merge(&mut words, &merge, *target);
      _update_merge_map(&mut pre_merges, &merge, changes, None);
    }

    Ok(words.into_iter().map(|i| i.idxs.to_word()).collect())
  }

  pub fn encode_words(&self, input: &[Word<C>]) -> MyResult<Vec<Word<Idx>>> {
    let mut results = BTreeMap::new();
    let mut to_encode = Vec::new();
    let mut query = Vec::new();
    input.iter().enumerate().for_each(|(i, w)| {
      if let Some(cached) = self.cache.get(w) {
        results.insert(i, cached);
      } else {
        to_encode.push(w.clone());
        query.push(i);
      }
    });
    let encoded = self._encode_words(&to_encode)?;
    for (i, (w, e)) in query.into_iter().zip(to_encode.into_iter().zip(encoded.into_iter())) {
      self.cache.insert(w.clone(), e.clone());
      results.insert(i, e);
    }
    let final_results = results.values().cloned().collect::<Vec<_>>();
    assert_eq!(final_results.len(), input.len());
    Ok(final_results)
  }

  /// encode a single word without cache.
  /// see [`Self::encode_word`] for cached version.
  pub fn _encode_word(&self, input: &Word<C>) -> MyResult<Word<Idx>> {
    let mut queue = BTreeMap::new();
    let mut words = vec![self._pretoken(input.clone(), 1)?];
    for (i1, i2) in words[0].idxs.iter().copied().zip(words[0].idxs.iter().skip(1).copied()) {
      let tp = (i1, i2);
      if let Some(merge) = self.pre_merge_map.get(&tp) {
        queue.insert((merge.data.freq, tp), merge);
      }
    }
    while let Some((_, merge)) = queue.pop_last() {
      let changes = _merge(&mut words, merge, merge.target.unwrap());
      for (tp, data) in changes {
        if data.occurs_in.is_empty() {
          continue;
        }
        let Some(merge) = self.pre_merge_map.get(&tp) else {
          continue;
        };
        if data.freq < 0 {
          queue.remove(&(merge.data.freq, tp));
        } else {
          queue.insert((merge.data.freq, tp), merge);
        }
      }
    }
    Ok(words.into_iter().next().unwrap().idxs.to_word())
  }

  pub fn encode_word(&self, input: &Word<C>) -> MyResult<Word<Idx>> {
    if let Some(result) = self.cache.get(input) {
      return Ok(result);
    }
    let result = self._encode_word(input)?;
    self.cache.insert(input.clone(), result.clone());
    Ok(result)
  }

  fn _encode_segment<P: AsRef<Path>>(
    path: P, special_tokens: &Vec<String>, offset: u64, len: usize, cache: &HashMap<Arc<[C]>, Arc<[Idx]>>,
  ) -> MyResult<Vec<Arc<[Idx]>>>
  where
    for<'a> &'a str: ToWord<C>,
  {
    let mut file = File::open(&path)?;
    file.seek(std::io::SeekFrom::Start(offset))?;
    let mut buffer = vec![0; len];
    file.read_exact(&mut buffer)?;

    let content = String::from_utf8_lossy(&buffer);
    let parts = split_special_tokens(&content, &special_tokens)?;
    let mut res = Vec::new();
    for (part, is_special) in parts.iter() {
      if *is_special {
        let w = part.to_word();
        let idxs = cache.get(&w).ok_or_else(|| MyError::Oov(w.display()))?;
        res.push(idxs.clone());
      } else {
        pretokenizer_tokens(part, &RE)?
          .iter()
          .try_for_each(|token| -> MyResult<()> {
            let w = token.to_word();
            let idxs = cache.get(&w).ok_or(MyError::Oov(w.display()))?;
            res.push(idxs.clone());
            Ok(())
          })?;
      }
    }
    return Ok(res);
  }

  fn _create_cache_from_words(
    &self, input: Vec<Word<C>>, special_tokens: &Vec<String>
  ) -> MyResult<HashMap<Arc<[C]>, Arc<[Idx]>>>
  where
    for<'a> &'a str: ToWord<C>,
  {
    let encoded = self._encode_words(&input)?;
    let mut cache = HashMap::from_iter(input.into_iter().zip(encoded.into_iter()).map(|(k, v)| (k, v)));
    special_tokens.iter().try_for_each(|token| -> MyResult<()> {
      let w = token.to_word();
      let idx = *self.vocab_rev.get(&w).ok_or(MyError::Oov(w.display()))?;
      let encoded_special = vec![idx].to_word();
      cache.insert(w, encoded_special);
      Ok(())
    })?;
    Ok(cache)
  }

  pub fn encode_file<P: AsRef<std::path::Path>>(
    &self, path: P, num_chunks: u32, special_tokens: Vec<String>, split_special_token: Option<&str>,
  ) -> MyResult<Vec<Word<Idx>>>
  where
    for<'a> &'a str: ToWord<C>,
  {
    let words = get_words_from_file(&path, num_chunks, special_tokens.clone(), split_special_token)?;
    let input = words.into_iter().map(|(k, _)| k.to_word()).collect::<Vec<Arc<[C]>>>();
    let cache = self._create_cache_from_words(input, &special_tokens)?;

    let split_special_token = split_special_token.unwrap_or("<|endoftext|>");
    let boundaries = find_chunk_boundaries(&path, num_chunks, split_special_token)?;
    let path = path.as_ref().to_path_buf();
    let params = boundaries
      .iter()
      .zip(boundaries.iter().skip(1))
      .enumerate()
      .map(|(index, (start, end))| (index, *start, (*end - *start) as usize))
      .collect::<Vec<_>>();
    let mut segment_results = params
      .into_iter()
      .map(|(index, offset, len)| -> MyResult<(usize, Vec<Word<Idx>>)> {
        let segment_result =  Self::_encode_segment(&path, &special_tokens, offset, len, &cache)?;
        Ok((index, segment_result))
      })
      .collect::<MyResult<Vec<_>>>()?;
    segment_results.sort_by_key(|(index, _)| *index);
    let result = segment_results.into_iter().map(|(_, res)| res).flatten().collect::<Vec<_>>();
    Ok(result)
  }
}


#[cfg(test)]
mod tests {
  use super::*;

  fn _setup_bpe(name: &str) -> BpeEncoder<u8> {
    let vocab = BpeEncoder::_load_vocab(std::fs::File::open(format!("fixtures/vocab.{name}.json")).unwrap()).unwrap();
    let merges = BpeEncoder::_load_merges(std::fs::File::open(format!("fixtures/merges.{name}.txt")).unwrap(), &vocab).unwrap();
    let vocab = vocab.into_iter().map(|(k, v)| (v, k)).collect();
    let merges = merges.into_iter().map(|m| (m.tp, m.target.unwrap())).collect();
    BpeEncoder::new(vocab, merges)
  }

  #[test]
  fn test_bpe_encode_words() {
    const NAME: &str = "tinystories_sample_5M";
    // const NAME: &str = "TinyStoriesV2-GPT4-train";
    let input: BTreeMap<String, Freq> = serde_json::from_str(&std::fs::read_to_string(format!("fixtures/_words.{NAME}.json")).unwrap()).unwrap();
    let input = input.into_iter().map(|(k, _)| k.to_word()).collect::<Vec<_>>();
    let bpe = _setup_bpe(NAME);
    let result = bpe._encode_words(&input).unwrap();
    assert_eq!(result.len(), input.len());

    let result2 = input.iter().map(|w| bpe._encode_word(w).unwrap()).collect::<Vec<_>>();
    assert_eq!(result, result2);
    // for ((i, src), (r1, r2)) in input.iter().enumerate().zip(result.iter().zip(result2.iter())) {
    //   assert_eq!(r1, r2, "[{i}] src={}", src.display());
    // }
  }

  #[test]
  fn test_cache() {
    const NAME: &str = "tinystories_sample_5M";
    let input: BTreeMap<String, Freq> = serde_json::from_str(&std::fs::read_to_string(format!("fixtures/{NAME}_words.json")).unwrap()).unwrap();
    let input = input.into_iter().map(|(k, _)| k.to_word()).collect::<Vec<_>>();
    let mut bpe = _setup_bpe(NAME);
    bpe.cache = Cache::new(input.len() as u64 * 6 / 5);
    let result1 = bpe.encode_words(&input).unwrap();
    let result2 = bpe.encode_words(&input).unwrap();
    assert_eq!(result1, result2);
    println!("input size: {}, cache size: {}", input.len(), bpe.cache.weighted_size())
  }

  #[test]
  fn test_bpe_encode_file() {
    const NAME: &str = "tinystories_sample_5M";
    let bpe = _setup_bpe(NAME);
    let result = bpe.encode_file(
      format!("fixtures/{NAME}.txt"),
      1,
      vec!["<|endoftext|>".to_string()],
      None,
    ).unwrap();
    assert!(result.len() == 1269588);
    let total_index: usize = result.iter().map(|idxs| idxs.len()).sum();
    assert!(total_index == 1424324);
  }
}
