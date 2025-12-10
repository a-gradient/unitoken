use fancy_regex::Regex;
use lazy_static::lazy_static;
use memchr::memmem;
use rayon::iter::{IntoParallelIterator, ParallelIterator as _};
use std::{
  collections::{BTreeMap, BTreeSet, HashMap},
  fs::{self, File},
  io::{Read as _, Seek},
  path::Path,
};

use crate::{MyError, MyResult, bpe::Freq};

lazy_static! {
  /// PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  pub static ref RE: Regex = Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap();
}

/// input a string and a pattern, return a map of tokens and their counts
pub fn pretokenizer_counter(s: &str, pat: &Regex) -> MyResult<BTreeMap<String, Freq>> {
  let mut result = BTreeMap::new();
  for i in pat.find_iter(s) {
    match i {
      Ok(m) => {
        let token = m.as_str().to_string();
        *result.entry(token).or_default() += 1;
      }
      Err(e) => {
        return Err(MyError::Regex(e));
      }
    }
  }
  Ok(result)
}

pub fn pretokenizer_tokens(s: &str, pat: &Regex) -> MyResult<Vec<String>> {
  let mut result = Vec::new();
  for i in pat.find_iter(s) {
    match i {
      Ok(m) => {
        let token = m.as_str().to_string();
        result.push(token);
      }
      Err(e) => {
        return Err(MyError::Regex(e));
      }
    }
  }
  Ok(result)
}

pub fn find_chunk_boundaries<P: AsRef<Path>>(
  path: P, desired_num_chunks: u32, split_special_token: &str,
) -> MyResult<Vec<u64>> {
  let file_size = fs::metadata(&path)?.len();
  let chunk_size = file_size / desired_num_chunks as u64;
  let mini_chunk_size = 4096;
  let finder = memmem::Finder::new(split_special_token);
  debug!(
    file_size = file_size,
    chunk_size = chunk_size,
    desired_num_chunks = desired_num_chunks,
    "find_chunk_boundaries"
  );

  let mut boundaries = Vec::new();
  for i in 0..(desired_num_chunks) {
    boundaries.push(chunk_size * i as u64);
  }
  boundaries.push(file_size);

  let mut file = File::open(&path)?;
  for bi in 1..boundaries.len() - 1 {
    let mut initial_position = boundaries[bi];
    let _ = file.seek(std::io::SeekFrom::Start(initial_position))?;
    loop {
      let mut buffer = vec![0; mini_chunk_size as usize];
      let bytes_read = file.read(&mut buffer)?;
      if bytes_read < mini_chunk_size as usize {
        boundaries[bi] = file_size;
        break;
      }
      if let Some(pos) = finder.find(buffer[..bytes_read].as_ref()) {
        let boundary = initial_position + pos as u64;
        boundaries[bi] = boundary;
        break;
      }
      initial_position += mini_chunk_size;
    }
  }

  let deduplicated_boundaries = boundaries.into_iter().collect::<BTreeSet<_>>();
  debug!(boundaries.len=?deduplicated_boundaries.len(), "find_chunk_boundaries");
  Ok(deduplicated_boundaries.into_iter().collect())
}

pub enum SplitChunk<'a> {
  Special(&'a str),
  Chunk(&'a str),
}

impl SplitChunk<'_> {
  pub fn as_str(&self) -> &str {
    match self {
      SplitChunk::Special(s) => s,
      SplitChunk::Chunk(s) => s,
    }
  }

  pub fn is_special(&self) -> bool {
    matches!(self, SplitChunk::Special(_))
  }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum SplitToken {
  Special(String),
  Token(String),
}

impl SplitToken {
  pub fn as_str(&self) -> &str {
    match self {
      SplitToken::Special(s) => s.as_str(),
      SplitToken::Token(s) => s.as_str(),
    }
  }

  pub fn is_special(&self) -> bool {
    matches!(self, SplitToken::Special(_))
  }
}

impl std::ops::Deref for SplitToken {
  type Target = str;

  fn deref(&self) -> &Self::Target {
    self.as_str()
  }
}

pub fn create_special_token_regex(special_tokens: &[String]) -> Regex {
  if special_tokens.is_empty() {
    return Regex::new("$^").unwrap(); // matches nothing
  }
  let pattern = special_tokens
    .iter()
    .map(|s| fancy_regex::escape(s).into_owned())
    .collect::<Vec<String>>()
    .join("|");
  Regex::new(&pattern).unwrap()
}

pub fn split_special_tokens<'a>(text: &'a str, special_tokens: &Regex) -> MyResult<Vec<SplitChunk<'a>>> {
  let mut parts = Vec::new();
  let mut last_pos = 0;
  for mat in special_tokens.find_iter(text) {
    match mat {
      Ok(m) => {
        if m.start() > last_pos {
          parts.push(SplitChunk::Chunk(&text[last_pos..m.start()]));
        }
        parts.push(SplitChunk::Special(&text[m.start()..m.end()]));
        last_pos = m.end();
      }
      Err(e) => return Err(MyError::Regex(e)),
    }
  }
  if last_pos < text.len() {
    parts.push(SplitChunk::Chunk(&text[last_pos..]));
  }
  Ok(parts)
}

pub fn read_file_to_buffer<P: AsRef<Path>>(path: P, offset: u64, len: usize) -> MyResult<Vec<u8>> {
  let mut file = File::open(&path)?;
  file.seek(std::io::SeekFrom::Start(offset))?;
  let mut buffer = vec![0; len];
  file.read_exact(&mut buffer)?;
  Ok(buffer)
}

pub fn get_words_from_segment<P: AsRef<Path>>(
  path: P, re_special_tokens: &Regex, offset: u64, len: usize,
) -> MyResult<BTreeMap<String, Freq>> {
  let _span = trace_span!("get_words_from_segment", offset = offset, len = len).entered();

  metrics::counter!("get_words_from_segment.calls").increment(1);
  let buffer = read_file_to_buffer(&path, offset, len)?;

  let content = String::from_utf8_lossy(&buffer);
  let parts = split_special_tokens(&content, &re_special_tokens)?;
  let mut words = BTreeMap::new();
  for part in parts.iter().filter(|i| !i.is_special()) {
    for (token, count) in pretokenizer_counter(part.as_str(), &RE)? {
      *words.entry(token).or_default() += count;
    }
  }
  metrics::histogram!("get_words_from_segment.words_count").record(words.len() as f64);
  metrics::counter!("get_words_from_segment.len").increment(len as _);

  trace!(words_len=?words.len(), "result");
  Ok(words)
}


pub fn get_tokens_index_from_segment<P: AsRef<Path>>(
  path: P, re_special_tokens: &Regex, offset: u64, len: usize,
) -> MyResult<HashMap<SplitToken, Vec<usize>>> {
  let _span = trace_span!("get_tokens_index_from_segment", offset = offset, len = len).entered();

  metrics::counter!("get_tokens_index_from_segment.calls").increment(1);
  let buffer = read_file_to_buffer(&path, offset, len)?;

  let content = String::from_utf8_lossy(&buffer);
  let parts = split_special_tokens(&content, &re_special_tokens)?;
  let mut tokens_index: HashMap<SplitToken, Vec<usize>> = HashMap::new();
  let mut doc_idx = 0;
  for part in parts.iter() {
    if part.is_special() {
      tokens_index.entry(SplitToken::Special(part.as_str().to_string())).or_default().push(doc_idx);
      doc_idx += 1;
    } else {
      for token in pretokenizer_tokens(part.as_str(), &RE)? {
        tokens_index.entry(SplitToken::Token(token)).or_default().push(doc_idx);
        doc_idx += 1;
      }
    }
  }
  Ok(tokens_index)
}

pub fn get_words_from_file<P: AsRef<Path>>(
  path: P, num_chunks: u32, re_special_tokens: Regex, split_special_token: Option<&str>,
) -> MyResult<BTreeMap<String, Freq>> {
  let split_special_token = split_special_token.unwrap_or("<|endoftext|>");
  let boundaries = find_chunk_boundaries(&path, num_chunks, split_special_token)?;
  let path = path.as_ref().to_path_buf();
  let params = boundaries
    .iter()
    .zip(boundaries.iter().skip(1))
    .map(|(start, end)| (*start, (*end - *start) as usize))
    .collect::<Vec<_>>();

  let words = params
    .into_par_iter()
    .map(|(offset, len)| get_words_from_segment(&path, &re_special_tokens.clone(), offset, len))
    .try_reduce(
      || BTreeMap::new(),
      |mut a, b| {
        for (k, v) in b.into_iter() {
          *a.entry(k).or_default() += v;
        }
        Ok(a)
      },
    )?;
  Ok(words)
}

pub fn get_tokens_index_from_file<P: AsRef<Path>>(
  path: P, num_chunks: u32, re_special_tokens: Regex, split_special_token: Option<&str>,
) -> MyResult<HashMap<SplitToken, Vec<usize>>> {
  let split_special_token = split_special_token.unwrap_or("<|endoftext|>");
  let boundaries = find_chunk_boundaries(&path, num_chunks, split_special_token)?;
  let path = path.as_ref().to_path_buf();
  let params = boundaries
    .iter()
    .zip(boundaries.iter().skip(1))
    .enumerate()
    .map(|(index , (start, end))| (index, *start, (*end - *start) as usize))
    .collect::<Vec<_>>();

  let mut segments_tokens_index = params
    .into_par_iter()
    .map(|(index, offset, len)| {
      get_tokens_index_from_segment(&path, &re_special_tokens.clone(), offset, len)
        .map(|segment_tokens_index| (index, segment_tokens_index))
    })
    .collect::<MyResult<Vec<_>>>()?;

  segments_tokens_index.sort_by(|(chunk_id_a, _), (chunk_id_b, _)| chunk_id_a.cmp(chunk_id_b));
  let mut segments_tokens_index = segments_tokens_index.into_iter().map(|(_, m)| m).collect::<Vec<_>>();
  let token_nums = segments_tokens_index.iter().map( | words_index| {
    words_index.values().map(|v| v.len()).sum::<usize>()
  }).collect::<Vec<_>>();
  let index_offset = prefix_sum(&token_nums);
  for (segment_words_index, &start_index) in segments_tokens_index.iter_mut().zip(&index_offset) {
    for idxs in segment_words_index.values_mut() {
      for idx in idxs {
        *idx += start_index;
      }
    }
  }

  let mut tokens_index: HashMap<SplitToken, Vec<usize>> = HashMap::new();

  segments_tokens_index
    .into_iter()
    .flatten()
    .for_each(|(token, doc_idxs)| {
      tokens_index.entry(token).or_default().extend(doc_idxs);
    });
  Ok(tokens_index)
}

fn prefix_sum(v: &[usize]) -> Vec<usize> {
  let mut result = Vec::with_capacity(v.len());
  let mut sum = 0;
  result.push(sum);
  for i in v[..v.len()-1].iter() {
    sum += i;
    result.push(sum);
  }
  result
}

pub fn sort_words(words: &BTreeMap<String, Freq>) -> ordermap::OrderMap<String, Freq> {
  let mut word_freq_vec: Vec<(String, Freq)> = words.iter().map(|(k,v)| (k.clone(), *v)).collect();
  word_freq_vec.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)).reverse());
  word_freq_vec.into_iter().collect()
}

pub fn save_words<W: std::io::Write>(w: W, words: &ordermap::OrderMap<String, Freq>) -> Result<(), std::io::Error> {
  serde_json::to_writer_pretty(w, &words)?;
  Ok(())
}

#[cfg(test)]
mod tests {
  use ordermap::OrderMap;
  use super::*;
  #[test]
  fn test_pretokenizer() {
    let s = "Hello, world! It's 2024.";
    let tokens = pretokenizer_counter(s, &RE).unwrap();
    let expected_tokens = vec![
      ("Hello".to_string(), 1),
      (",".to_string(), 1),
      (" world".to_string(), 1),
      ("!".to_string(), 1),
      (" It".to_string(), 1),
      ("'s".to_string(), 1),
      (" 2024".to_string(), 1),
      (".".to_string(), 1),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    assert_eq!(tokens, expected_tokens);

    let s = "你好，世界！Now是2024年。";
    let tokens = pretokenizer_counter(s, &RE).unwrap();
    let expected_tokens = vec![
      ("你好".to_string(), 1),
      ("，".to_string(), 1),
      ("世界".to_string(), 1),
      ("！".to_string(), 1),
      ("Now是".to_string(), 1),
      ("2024".to_string(), 1),
      ("年".to_string(), 1),
      ("。".to_string(), 1),
    ]
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    assert_eq!(tokens, expected_tokens);
  }

  #[test]
  fn test_sample() {
    let input = std::fs::read_to_string("fixtures/tinystories_sample_5M.txt").unwrap();
    let tokens = pretokenizer_counter(&input, &RE).unwrap();
    assert_eq!(tokens.get(" the").cloned().unwrap_or(0), 48886);
  }

  #[test]
  fn test_find_chunk_boundaries() {
    let path = std::path::Path::new("fixtures/tinystories_sample_5M.txt");

    let desired_num_chunks = 4;
    let boundaries = find_chunk_boundaries(path, desired_num_chunks, "<|endoftext|>").unwrap();
    let expect = vec![0, 1310951, 2621933, 3932548, 5242880];
    assert!(boundaries == expect, "{:?} != {:?}", boundaries, expect);

    let desired_num_chunks = 10;
    let boundaries = find_chunk_boundaries(path, desired_num_chunks, "<|endoftext|>").unwrap();
    let expect = vec![
      0, 525166, 1048920, 1573438, 2097691, 2621933, 3146237, 3670035, 4196392, 4718956, 5242880,
    ];
    assert!(boundaries == expect, "{:?} != {:?}", boundaries, expect);
  }

  #[test]
  fn test_get_words_from_file() {
    const NAME: &str = "tinystories_sample_5M";
    // const NAME: &str = "TinyStoriesV2-GPT4-train";
    let path = format!("fixtures/{NAME}.txt");
    let num_chunks = 16;
    let words = get_words_from_file(
      path,
      num_chunks,
      create_special_token_regex(&["<|endoftext|>".to_string()]),
      Some("<|endoftext|>"),
    )
    .unwrap();
    let words = sort_words(&words);
    if NAME == "tinystories_sample_5M" {
      assert_eq!(words.get(" the").cloned().unwrap_or(0), 48886);
    }
    std::fs::create_dir_all("out").ok();
    serde_json::to_writer_pretty(std::fs::File::create(format!("out/_words.{NAME}.json")).unwrap(), &words).unwrap();
    let answer = std::fs::read_to_string(format!("fixtures/_words.{NAME}.json")).unwrap();
    let expected: OrderMap<String, Freq> = serde_json::from_str(&answer).unwrap();
    assert_eq!(words, expected);
  }

  #[test]
  fn test_split_special_tokens() {
    const NAME: &str = "tinystories_sample_5M";
    let path = format!("fixtures/{NAME}.txt");
    let text = std::fs::read_to_string(&path).unwrap();
    let parts = split_special_tokens(
      &text,
      &create_special_token_regex(&["<|endoftext|>".to_string()]),
    ).unwrap();
    assert!(parts.len() == 12915);
  }

  #[test]
  fn test_get_tokens_index_from_segment() {
    const NAME: &str = "tinystories_sample_5M";
    let path = format!("fixtures/{NAME}.txt");
    let tokens_index = get_tokens_index_from_segment(
      &path,
      &create_special_token_regex(&["<|endoftext|>".to_string()]),
      0, 5242880,
    ).unwrap();
    let idxs = tokens_index.get(&SplitToken::Token(" the".to_string())).unwrap();
    println!("the idxs length: {:?}", idxs.len());
    assert_ne!( idxs.len(), 0);
  }

  #[test]
  fn test_get_tokens_index_from_file() {
    const NAME: &str = "tinystories_sample_5M";
    let path = format!("fixtures/{NAME}.txt");
    let num_chunks = 4;
    let tokens_index = get_tokens_index_from_file(
      path,
      num_chunks,
      create_special_token_regex(&["<|endoftext|>".to_string()]),
      Some("<|endoftext|>"),
    ).unwrap();
    assert_ne!(tokens_index.len(), 0);
    let tokens_index = tokens_index.into_iter().map(|(w, v)| (w.as_str().to_string(), v)).collect::<HashMap<_, _>>();
    serde_json::to_writer_pretty(std::fs::File::create(format!("out/_tokens_index.{NAME}.json")).unwrap(), &tokens_index).unwrap();
  }
}
