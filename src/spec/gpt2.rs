
use std::{collections::BTreeMap, io::BufReader, sync::Arc};

use lazy_static::lazy_static;
use ordermap::OrderMap;

use crate::{MyError, MyResult, bpe::{Idx, Merge, Word, utils::WordExt}, spec::Spec};

pub struct Gpt2Spec;

impl Spec<Idx, u8> for Gpt2Spec {
  fn encode_vocab<W: std::io::Write>(&self, mut w: W, vocab: &BTreeMap<Idx, Word<u8>>) -> MyResult<()> {
    let mut map = OrderMap::new();
    for (idx, word) in vocab.iter() {
      let s = _printable(word);
      map.insert(s, idx);
    }
    let json = serde_json::to_string_pretty(&map).unwrap();
    write!(w, "{}", json)?;
    Ok(())
  }

  fn decode_vocab<R: std::io::Read>(&self, r: R) -> MyResult<BTreeMap<Idx, Word<u8>>> {
    let map: OrderMap<String, Idx> = serde_json::from_reader(BufReader::new(r))?;
    map.into_iter().map(|(s, idx)| {
      let word = _from_printable(&s).map_err(|ch| crate::MyError::InvalidPrintableChar(ch))?;
      Ok((idx, word))
    }).collect()
  }

  fn encode_merges<W: std::io::Write>(&self, mut w: W, merges: &Vec<Merge<u8, Idx>>) -> MyResult<()> {
    for merge in merges.iter() {
      let left = _printable(&merge.content.0);
      let right = _printable(&merge.content.1);
      writeln!(w, "{} {} => {}", left, right, merge.data.freq)?;
    }
    Ok(())
  }

  fn decode_merges<R: std::io::Read>(&self, mut reader: R, vocab: &BTreeMap<Idx, Word<u8>>) -> MyResult<Vec<Merge<u8, Idx>>> {
    let mut result = Vec::new();
    let mut input = String::new();
    reader.read_to_string(&mut input)?;
    let vocab = vocab.iter().map(|(k, v)| (v.clone(), *k)).collect::<BTreeMap<Word<u8>, Idx>>();
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

pub(crate) fn _printable(w: &Word<u8>) -> String {
  w.iter()
    .map(|b| PRINTABLE.get(b).copied().unwrap_or('.'))
    .collect()
}

pub(crate) fn _from_printable(s: &str) -> Result<Word<u8>, char> {
  let bytes = s
    .chars()
    .map(|ch| PRINTABLE_REV.get(&ch).copied().ok_or(ch))
    .collect::<Result<Vec<_>, _>>()?;
  Ok(Arc::from(bytes.into_boxed_slice()))
}
