use std::{collections::BTreeMap, io::BufReader, sync::Arc};

use fancy_regex::Regex;
use lazy_static::lazy_static;
use ordermap::OrderMap;

use crate::{MyError, MyResult, bpe::{Character, Idx, Merge, Word}, spec::Spec};

pub struct UniSpec;

impl Spec<Idx, Character> for UniSpec {
  fn encode_vocab<W: std::io::Write>(&self, mut w: W, vocab: &BTreeMap<Idx, Word<Character>>) -> MyResult<()> {
    let mut map = OrderMap::new();
    for (idx, word) in vocab.iter() {
      let s = _printable(word);
      map.insert(s, idx);
    }
    let json = serde_json::to_string_pretty(&map).unwrap();
    write!(w, "{}", json)?;
    Ok(())
  }

  fn decode_vocab<R: std::io::Read>(&self, r: R) -> MyResult<BTreeMap<Idx, Word<Character>>> {
    let map: OrderMap<String, Idx> = serde_json::from_reader(BufReader::new(r))?;
    map.into_iter().map(|(s, idx)| {
      let word = _from_printable(&s)?;
      Ok((idx, word))
    }).collect()
  }

  fn encode_merges<W: std::io::Write>(&self, mut w: W, merges: &Vec<Merge<Character, Idx>>) -> MyResult<()> {
    for merge in merges.iter() {
      let left = _printable(&merge.content.0);
      let right = _printable(&merge.content.1);
      writeln!(w, "{} {} => {}", left, right, merge.data.freq)?;
    }
    Ok(())
  }

  fn decode_merges<R: std::io::Read>(&self, mut reader: R, vocab: &BTreeMap<Idx, Word<Character>>) -> MyResult<Vec<Merge<Character, Idx>>> {
    let mut result = Vec::new();
    let mut input = String::new();
    reader.read_to_string(&mut input)?;
    let vocab = vocab.iter().map(|(k, v)| (v.clone(), *k)).collect::<BTreeMap<_, _>>();
    fn get_kv(vocab: &BTreeMap<Word<Character>, Idx>, s: &str) -> MyResult<(Idx, Word<Character>)> {
      let w = _from_printable(s)?;
      Ok((*vocab.get(&w).ok_or_else(|| MyError::Oov(_printable(&w)))?, w))
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
      let (a_idx, a) = get_kv(&vocab, parts[0])?;
      let (b_idx, b) = get_kv(&vocab, parts[1])?;
      let merged = format!("{}{}", parts[0], parts[1]);
      let (m_idx, _) = get_kv(&vocab, &merged)?;
      let mut merge = Merge::new((a_idx, b_idx), (a, b)).with_target(m_idx);
      merge.data.freq = freq;
      result.push(merge);
    }
    Ok(result)
  }
}

fn display_char(ch: &Character) -> String {
  match ch {
    Character::Unicode(' ') => '␣'.to_string(),
    Character::Unicode('␣') => format!("{{u{:04x}}}", '␣' as u32),
    Character::Unicode('{') => "{u007b}".to_string(),
    Character::Unicode('}') => "{u007d}".to_string(),
    Character::Unicode(c) => {
      c.to_string()
    }
    Character::Byte(b) => format!("{{x{:02x}}}", *b),
  }
}

fn _try_combine(word: &Word<u8>) -> Word<Character> {
  let mut chars = Vec::with_capacity(word.len());
  let mut c = vec![];
  fn convert_str(c: Vec<u8>) -> Vec<Character> {
    match String::from_utf8(c) {
      Ok(s) => s.chars().map(|ch| Character::Unicode(ch)).collect(),
      Err(e) => e.as_bytes().iter().map(|b| Character::Byte(*b)).collect(),
    }
  }
  for &b in word.iter() {
    if b.is_ascii() {
      if !c.is_empty() {
        chars.extend(convert_str(c));
        c = vec![];
      }
      chars.push(Character::Unicode(b as char));
    } else if b < 0b_1100_0000 {
      if !c.is_empty() {
        c.push(b);
      } else {
        chars.push(Character::Byte(b));
      }
      continue;
    } else {
      chars.extend(convert_str(c));
      c = vec![b];
    }
  }
  Arc::from(chars.into_boxed_slice())
}

fn _printable(word: &Word<Character>) -> String {
  word.iter().map(|c| display_char(c)).collect()
}

fn _printable_u8(word: &Word<u8>) -> String {
  _printable(&_try_combine(word))
}

lazy_static! {
  static ref PRINTABLE_REGEX: Regex = Regex::new(r"\{([ux][0-9a-fA-F]{2,})\}").unwrap();
}
fn parse_str(s: &str) -> MyResult<Vec<Character>> {
  let mut result = Vec::new();
  let mut last_i = 0;
  for m in PRINTABLE_REGEX.find_iter(s) {
    let m = m?;
    for c in s[last_i..m.start()].chars() {
      result.push(Character::Unicode(c));
    }
    last_i = m.end();
    let token = m.as_str();
    let token = &token[1..token.len() - 1]; // strip {}
    if token.starts_with('u') {
      let codepoint = u32::from_str_radix(&token[1..], 16).map_err(|_| MyError::InvalidPrintableChar('?'))?;
      if let Some(ch) = std::char::from_u32(codepoint) {
        result.push(Character::Unicode(ch));
      } else {
        return Err(MyError::InvalidPrintableEscape(token.to_string()));
      }
    } else if token.starts_with('x') {
      let byte = u8::from_str_radix(&token[1..], 16).map_err(|_| MyError::InvalidPrintableEscape(token.to_string()))?;
      result.push(Character::Byte(byte));
    } else {
      return Err(MyError::InvalidPrintableEscape(token.to_string()));
    }
  }
  Ok(result)
}

fn _from_printable(s: &str) -> MyResult<Word<Character>> {
  let chars = parse_str(s)?;
  Ok(Arc::from(chars.into_boxed_slice()))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_display_char() {
    assert_eq!(display_char(&Character::Unicode('a')), "a".to_string());
    assert_eq!(display_char(&Character::Unicode(' ')), "␣".to_string());
    assert_eq!(display_char(&Character::Unicode('␣')), "{00a0}".to_string());
    assert_eq!(display_char(&Character::Unicode('{')), "{007b}".to_string());
    assert_eq!(display_char(&Character::Unicode('}')), "{007d}".to_string());
    assert_eq!(display_char(&Character::Byte(0x41)), "{41}".to_string());
  }

  #[test]
  fn test_parse_str() {
    let s = "a{u0041} {x42}{x43}{u0044}";
    let chars = parse_str(s).unwrap();
    let expected = vec![
      Character::Unicode('a'),
      Character::Unicode('A'),
      Character::Unicode(' '),
      Character::Byte(0x42),
      Character::Byte(0x43),
      Character::Unicode('D'),
    ];
    assert_eq!(chars, expected);
  }
}
