use std::{collections::{BTreeMap, HashMap}, io::BufReader, path::Path};

use fancy_regex::Regex;
use moka::sync::Cache;

use crate::{
  MyError, MyResult,
  pretokenizer::{RE, create_special_token_regex, find_chunk_boundaries, get_words_from_file, pretokenizer_tokens, read_file_to_buffer, split_special_tokens},
};

use super::*;

#[derive(Clone)]
pub struct BpeEncoder<C = u8> {
  pub vocab_bytes: BTreeMap<C, Idx>,
  pub vocab_rev: BTreeMap<Word<C>, Idx>,
  pub vocab: BTreeMap<Idx, Word<C>>,
  pub special_tokens: BTreeMap<String, Idx>,
  pub re_special_tokens: Regex,
  pub merges: Vec<((Idx, Idx), Idx)>,
  /// with freq represents rank, or `merge.data.freq=-i` for i-th merge.
  /// with [`occurs_in={0}`](MergeData::occurs_in), in order to handle first word in [`Self::_encode_word`].
  pub pre_merge_map: HashMap<(Idx, Idx), Merge<C, Idx>>,
  pub cache: Cache<String, Word<Idx>>,
}

impl BpeEncoder<u8> {
  fn _load_vocab<R: std::io::Read>(reader: R) -> MyResult<BTreeMap<Word<u8>, Idx>> {
    let input: BTreeMap<String, u64> = serde_json::from_reader(BufReader::new(reader))?;
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

  pub fn new_from_file<P: AsRef<std::path::Path>>(
    vocab_path: P, merges_path: P, special_tokens: Vec<String>,
  ) -> MyResult<Self> {
    let vocab = Self::_load_vocab(std::fs::File::open(vocab_path)?)?;
    let merges = Self::_load_merges(std::fs::File::open(merges_path)?, &vocab)?;
    let vocab = vocab.into_iter().map(|(k, v)| (v, k)).collect();
    let merges = merges.into_iter().map(|m| (m.tp, m.target.unwrap())).collect();
    Self::new(vocab, merges, special_tokens)
  }

  pub fn get_special_tokens_from_vocab<P: AsRef<Path>>(vocab_path: P) -> MyResult<Vec<String>> {
    let vocab = BpeEncoder::_load_vocab(std::fs::File::open(vocab_path)?)?;
    let vocab = vocab.into_iter().map(|(k, v)| (v, k)).collect::<BTreeMap<_, _>>();
    let mut special_tokens = Vec::new();
    for index in 0..vocab.len() {
      if let Some(token) = vocab.get(&(index as u32)) {
        if token.len() > 1 {
          special_tokens.push(token.display());
        } else {
          break;
        }
      }
    }
    Ok(special_tokens)
  }
}

impl<C: Ord + Clone + Cachable> BpeEncoder<C>
where
  Word<C>: WordExt,
  for<'a> &'a str: ToWord<C>,
{
  pub fn new(vocab: BTreeMap<Idx, Word<C>>, merges: Vec<((Idx, Idx), Idx)>, special_tokens: Vec<String>) -> MyResult<Self> {
    let vocab_rev = vocab
      .iter()
      .map(|(k, v)| (v.clone(), *k))
      .collect::<BTreeMap<_, _>>();
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
        vocab.get(&tp.0).ok_or_else(|| MyError::OovIdx(tp.0)).cloned()?,
        vocab.get(&tp.1).ok_or_else(|| MyError::OovIdx(tp.1)).cloned()?,
      )).with_target(target);
      merge.add(0, -(i as Freq));
      Ok((tp, merge))
    }).collect::<MyResult<_>>()?;
    let re_special_tokens = create_special_token_regex(&special_tokens);
    let special_tokens = special_tokens.into_iter().map(|s| {
      let w = s.to_word();
      let idx = *vocab_rev.get(&w).ok_or_else(|| MyError::Oov(w.display()))?;
      Ok((s, idx))
    }).collect::<MyResult<_>>()?;
    let max_cap = vocab.len() as u64 * 3 / 2;
    Ok(Self {
      vocab_bytes,
      vocab_rev,
      vocab,
      merges,
      pre_merge_map,
      special_tokens,
      re_special_tokens,
      cache: Cache::new(max_cap),
    })
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

  pub fn encode_words<S: AsRef<str>, I: IntoIterator<Item = S>>(&self, input: I) -> MyResult<Vec<Word<Idx>>>
  where
    for<'a> &'a str: ToWord<C>,
  {
    let mut results = BTreeMap::new();
    let mut to_encode = Vec::new();
    let mut query = Vec::new();
    let input_len = input.into_iter().enumerate().map(|(i, w)| {
      let w = w.as_ref();
      if let Some(cached) = self.cache.get(w) {
        results.insert(i, cached);
      } else {
        to_encode.push(w.to_word());
        query.push((i, w.to_string()));
      }
    }).count();
    let encoded = self._encode_words(&to_encode)?;
    for ((i, w), (_, e)) in query.into_iter().zip(to_encode.into_iter().zip(encoded.into_iter())) {
      self.cache.insert(w, e.clone());
      results.insert(i, e);
    }
    let final_results = results.values().cloned().collect::<Vec<_>>();
    assert_eq!(final_results.len(), input_len);
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

  pub fn encode_word(&self, input: &str) -> MyResult<Word<Idx>>
  where
    for<'a> &'a str: ToWord<C>,
  {
    if let Some(result) = self.cache.get(input) {
      return Ok(result);
    }
    let result = self._encode_word(&input.to_word())?;
    self.cache.insert(input.to_string(), result.clone());
    Ok(result)
  }

  fn encode_string(&self, input: &str) -> MyResult<Vec<Idx>>
  where
    for<'a> &'a str: ToWord<C>,
  {
    let parts = split_special_tokens(&input, &self.re_special_tokens)?;
    let mut res = Vec::new();
    for part in parts.iter() {
      if part.is_special() {
        let idx = *self.special_tokens.get(part.as_str()).ok_or_else(|| MyError::Oov(part.as_str().to_string()))?;
        res.push(idx);
      } else {
        pretokenizer_tokens(part.as_str(), &RE)?
          .iter()
          .try_for_each(|token| -> MyResult<()> {
            let idxs = self.encode_word(&token)?;
            res.extend_from_slice(&idxs);
            Ok(())
          })?;
      }
    }
    return Ok(res);
  }

  fn _create_cache_from_words(
    &self, input: Vec<String>
  ) -> MyResult<OrderMap<String, Arc<[Idx]>>>
  where
    for<'a> &'a str: ToWord<C>,
  {
    let words = input.iter().map(|s| s.to_word()).collect::<Vec<_>>();
    let encoded = self._encode_words(&words)?;
    let mut cache = OrderMap::from_iter(input.into_iter().zip(encoded.into_iter()).rev().map(|(k, v)| (k, v)));
    self.special_tokens.keys().try_for_each(|token| -> MyResult<()> {
      let w = token.to_word();
      let idx = *self.vocab_rev.get(&w).ok_or(MyError::Oov(w.display()))?;
      let encoded_special = vec![idx].to_word();
      cache.insert(token.to_string(), encoded_special);
      Ok(())
    })?;
    Ok(cache)
  }

  pub fn _split_special_token(&self) -> Option<&str> {
    self.special_tokens.iter().min_by_key(|(_, v)| *v).map(|(k, _)| k.as_str())
  }

  pub fn with_cache(mut self, cache: OrderMap<String, Arc<[Idx]>>) -> Self
  where
    for<'a> &'a str: ToWord<C>,
  {
    let max_cap = cache.len() as u64 * 3 / 2;
    self.cache = Cache::new(max_cap);
    for (k, v) in cache {
      self.cache.insert(k, v);
    }
    self
  }

  pub fn encode_file_with_cache<P: AsRef<std::path::Path>>(
    &self, path: P, num_chunks: u32,
  ) -> MyResult<Vec<Idx>>
  where
    for<'a> &'a str: ToWord<C>,
  {
    let split_special_token = self._split_special_token();
    let words = get_words_from_file(&path, num_chunks, self.re_special_tokens.clone(), split_special_token)?;
    let input = words.into_iter().map(|(k, _)| k).collect::<Vec<_>>();
    let cache = self._create_cache_from_words(input)?;
    let bpe_with_cache = self.clone().with_cache(cache);
    bpe_with_cache._encode_file(path, num_chunks)
  }

  pub fn _encode_file<P: AsRef<std::path::Path>>(
    &self, path: P, num_chunks: u32
  ) -> MyResult<Vec<Idx>>
  where
    for<'a> &'a str: ToWord<C>,
  {
    // TODO: handle this
    let split_special_token = self._split_special_token().unwrap_or("<|endoftext|>");
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
      .map(|(index, offset, len)| {
        let buffer = read_file_to_buffer(&path, offset, len)?;
        let content = String::from_utf8_lossy(&buffer);
        let segment_result =  self.encode_string(&content)?;
        Ok((index, segment_result))
      })
      .collect::<MyResult<Vec<_>>>()?;
    segment_results.sort_by_key(|(index, _)| *index);
    let result = segment_results.into_iter().map(|(_, res)| res).flatten().collect::<Vec<_>>();
    Ok(result)
  }
}


pub fn save_idxs<P: AsRef<std::path::Path>>(file_path: P, idxs: Vec<Idx>) -> MyResult<()> {
  let field = arrow::datatypes::Field::new("idx", arrow::datatypes::DataType::UInt32, false);
  let schema = Arc::new(arrow::datatypes::Schema::new(vec![field]));
  let array = arrow::array::UInt32Array::from(idxs);
  let batch = arrow::record_batch::RecordBatch::try_new(
    schema.clone(),
    vec![Arc::new(array)]
  )?;
  let file = std::fs::File::create(file_path)?;

  let props = parquet::file::properties::WriterProperties::builder().build();
  let mut writer = parquet::arrow::arrow_writer::ArrowWriter::try_new(file, schema, Some(props))?;
  writer.write(&batch)?;
  writer.close()?;
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  fn _setup_bpe(name: &str) -> BpeEncoder<u8> {
    let vocab = BpeEncoder::_load_vocab(std::fs::File::open(format!("fixtures/vocab.{name}.json")).unwrap()).unwrap();
    let merges = BpeEncoder::_load_merges(std::fs::File::open(format!("fixtures/merges.{name}.txt")).unwrap(), &vocab).unwrap();
    let vocab = vocab.into_iter().map(|(k, v)| (v, k)).collect();
    let merges = merges.into_iter().map(|m| (m.tp, m.target.unwrap())).collect();
    BpeEncoder::new(vocab, merges, vec!["<|endoftext|>".to_string()]).unwrap()
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
    let input: BTreeMap<String, Freq> = serde_json::from_str(&std::fs::read_to_string(format!("fixtures/_words.{NAME}.json")).unwrap()).unwrap();
    let input = input.iter().map(|(k, _)| k).collect::<Vec<_>>();
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
    let result = bpe.encode_file_with_cache(
      format!("fixtures/{NAME}.txt"),
      1,
    ).unwrap();
    // assert!(result.len() == 1269588);
    // let total_index: usize = result.iter().map(|idxs| idxs.len()).sum();
    assert!(result.len() == 1424324);
  }
}
