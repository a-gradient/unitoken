
use std::{collections::BTreeMap, io::BufReader, sync::Arc};

use lazy_static::lazy_static;
use ordermap::OrderMap;

use crate::{MyError, MyResult, bpe::{Idx, Merge, Word}, spec::{Spec, WordDisplay}};

pub struct Gpt2Spec;

impl<C: Ord> Spec<C, Idx> for Gpt2Spec
where
  Self: WordDisplay<C>,
{
  fn encode_vocab(&self, w: &mut dyn std::io::Write, vocab: &BTreeMap<Idx, Word<C>>) -> MyResult<()> {
    let mut map = OrderMap::new();
    for (idx, word) in vocab.iter() {
      let s = self.word_display(word);
      map.insert(s, idx);
    }
    let json = serde_json::to_string_pretty(&map).unwrap();
    write!(w, "{}", json)?;
    Ok(())
  }

  fn decode_vocab(&self, r: &mut dyn std::io::Read) -> MyResult<BTreeMap<Idx, Word<C>>> {
    let map: OrderMap<String, Idx> = serde_json::from_reader(BufReader::new(r))?;
    map.into_iter().map(|(s, idx)| {
      let word = self.word_parse(&s)?;
      Ok((idx, word))
    }).collect()
  }

  fn encode_merges(&self, w: &mut dyn std::io::Write, merges: &Vec<Merge<C, Idx>>) -> MyResult<()> {
    for merge in merges.iter() {
      let left = self.word_display(&merge.content.0);
      let right = self.word_display(&merge.content.1);
      writeln!(w, "{} {} => {}", left, right, merge.data.freq)?;
    }
    Ok(())
  }

  fn decode_merges_raw(&self, reader: &mut dyn std::io::Read) -> MyResult<Vec<Merge<C, Word<C>>>> {
    let mut result = Vec::new();
    let mut input = String::new();
    reader.read_to_string(&mut input)?;
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
      let a = self.word_parse(parts[0])?;
      let b = self.word_parse(parts[1])?;
      let mut merge = Merge::new((a.clone(), b.clone()), (a, b));
      merge.data.freq = freq;
      result.push(merge);
    }
    Ok(result)
  }

  fn decode_merges(&self, r: &mut dyn std::io::Read, vocab: &BTreeMap<Idx, Word<C>>) -> MyResult<Vec<Merge<C, Idx>>> {
    let merges_raw = self.decode_merges_raw(r)?;
    let vocab = vocab.iter().map(|(k, v)| (v.clone(), *k)).collect::<BTreeMap<_, _>>();
    let get_kv = |vocab: &BTreeMap<Word<C>, Idx>, w: &Word<C>| -> MyResult<Idx> {
      Ok(*vocab.get(w).ok_or_else(|| MyError::Oov(self.word_display(w)))?)
    };
    let result = merges_raw.into_iter().map(|merge| {
      let (a, b) = &merge.content;
      let a_idx = get_kv(&vocab, &a)?;
      let b_idx = get_kv(&vocab, &b)?;
      let merged = format!("{}{}", self.word_display(&merge.content.0), self.word_display(&merge.content.1));
      let m_idx = get_kv(&vocab, &self.word_parse(&merged)?)?;
      let mut merge_new = Merge::new((a_idx, b_idx), merge.content).with_target(m_idx);
      merge_new.data.freq = merge.data.freq;
      Ok(merge_new)
    }).collect::<MyResult<_>>()?;
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

impl WordDisplay<u8> for Gpt2Spec {
  fn word_display(&self, word: &Word<u8>) -> String {
    _printable(word)
  }

  fn word_parse(&self, s: &str) -> MyResult<Word<u8>> {
    let w = _from_printable(s).map_err(|e| MyError::InvalidPrintableChar(e))?;
    Ok(w)
  }
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
