use fancy_regex::Regex;
use lazy_static::lazy_static;
use memchr::memmem;
use std::{
  collections::BTreeMap,
  fs::{self, File},
  io::{Read as _, Seek},
  path::Path,
};

use crate::{MyError, MyResult, bpe::Freq};

lazy_static! {
  /// PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  static ref RE: Regex = Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap();
}

/// input a string and a pattern, return a map of tokens and their counts
pub fn pretokenizer(s: &str, pat: &Regex) -> MyResult<BTreeMap<String, Freq>> {
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

pub fn find_chunk_boundaries<P: AsRef<Path>>(
  path: P, desired_num_chunks: u32, split_special_token: &str,
) -> MyResult<Vec<u64>> {
  let file_size = fs::metadata(&path)?.len();
  let chunk_size = file_size / desired_num_chunks as u64;
  let mini_chunk_size = 4096;
  let finder = memmem::Finder::new(split_special_token);

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

  let mut set = std::collections::BTreeSet::new();
  let mut deduplicated_boundaries = boundaries
    .iter()
    .filter(|&item| set.insert(*item))
    .cloned()
    .collect::<Vec<u64>>();
  deduplicated_boundaries.sort();
  Ok(deduplicated_boundaries)
}

pub fn get_words_from_file<P: AsRef<Path>>(
  path: P, split_special_token: Option<&str>,
) -> MyResult<BTreeMap<String, Freq>> {
  let split_special_token = split_special_token.unwrap_or("<|endoftext|>");
  let file_size = path.as_ref().metadata()?.len();
  let max_content_size = 256 * 1024 * 1024; // 256 MB
  let num_chunks = (file_size as f64 / max_content_size as f64).ceil();

  let boundaries = find_chunk_boundaries(&path, num_chunks as u32, split_special_token)?;
  let mut words = BTreeMap::new();
  for (start, end) in boundaries.iter().zip(boundaries.iter().skip(1)) {
    let mut file = File::open(&path)?;
    file.seek(std::io::SeekFrom::Start(*start))?;
    let mut buffer = vec![0; (end - start) as usize];
    file.read_exact(&mut buffer)?;

    let content = String::from_utf8_lossy(&buffer);
    let tokens = pretokenizer(&content, &RE)?;
    for (token, count) in tokens {
      *words.entry(token).or_default() += count;
    }
  }
  Ok(words)
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn test_pretokenizer() {
    let s = "Hello, world! It's 2024.";
    let tokens = pretokenizer(s, &RE).unwrap();
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
    let tokens = pretokenizer(s, &RE).unwrap();
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
    let tokens = pretokenizer(&input, &RE).unwrap();
    assert_eq!(tokens.get(" the").cloned().unwrap_or(0), 48886);
    // serde_json::to_writer_pretty(std::fs::File::create("fixtures/tinystories_sample_5M_words.json").unwrap(), &tokens).unwrap();
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
    let path = std::path::Path::new("fixtures/tinystories_sample_5M.txt");
    let words = get_words_from_file(path, Some("<|endoftext|>")).unwrap();
    assert_eq!(words.get(" the").cloned().unwrap_or(0), 48886);
  }
}
