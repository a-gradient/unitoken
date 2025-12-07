
use std::collections::BTreeMap;

use fancy_regex::Regex;
use lazy_static::lazy_static;

use crate::Error;

lazy_static! {
  /// PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  static ref RE: Regex = Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap();
}

/// input a string and a pattern, return a map of tokens and their counts
pub fn pretokenizer(s: &str, pat: &Regex) -> Result<BTreeMap<String, u64>, Error> {
  let mut result = BTreeMap::new();
  for i in pat.find_iter(s) {
    match i {
      Ok(m) => {
        let token = m.as_str().to_string();
        *result.entry(token).or_default() += 1;
      }
      Err(e) => {
        return Err(Error {
          msg: format!("Regex error: {}", e),
          loc: (file!(), line!()),
        });
      }
    }
  }
  Ok(result)
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
}
