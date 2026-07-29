use lazy_static::lazy_static;
use regex_syntax::{
  hir::{Class, ClassUnicode, HirKind},
  Parser,
};

use crate::MyResult;

pub(super) const PATTERN: &str =
  r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

lazy_static! {
  static ref LETTER_CLASS: ClassUnicode = unicode_class(r"\p{L}");
  static ref NUMBER_CLASS: ClassUnicode = unicode_class(r"\p{N}");
  static ref WHITESPACE_CLASS: ClassUnicode = unicode_class(r"\s");
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CharClass {
  Letter,
  Number,
  Whitespace,
  Other,
}

pub(super) fn for_each<'a>(
  text: &'a str,
  mut emit: impl FnMut(&'a str) -> MyResult<()>,
) -> MyResult<()> {
  let mut start = 0;
  while start < text.len() {
    let end = pretoken_end(text, start);
    debug_assert!(end > start);
    debug_assert!(text.is_char_boundary(end));
    emit(&text[start..end])?;
    start = end;
  }
  Ok(())
}

fn pretoken_end(text: &str, start: usize) -> usize {
  let bytes = text.as_bytes();
  let contraction = if bytes[start] == b'\'' {
    contraction_len(&bytes[start..])
  } else {
    None
  };
  if let Some(len) = contraction {
    return start + len;
  }

  if bytes[start] == b' ' && start + 1 < bytes.len() {
    let next_start = start + 1;
    let next = text[next_start..]
      .chars()
      .next()
      .expect("input continues after ASCII space");
    let class = char_class(next);
    if class != CharClass::Whitespace {
      return scan_class(text, next_start, class);
    }
  }

  let first = text[start..]
    .chars()
    .next()
    .expect("start is before end of text");
  let class = char_class(first);
  if class == CharClass::Whitespace {
    return scan_whitespace(text, start);
  }
  scan_class(text, start, class)
}

fn contraction_len(bytes: &[u8]) -> Option<usize> {
  for suffix in [b"ll".as_slice(), b"ve", b"re", b"s", b"d", b"m", b"t"] {
    if bytes.get(1..1 + suffix.len()) == Some(suffix) {
      return Some(1 + suffix.len());
    }
  }
  None
}

fn scan_class(text: &str, start: usize, class: CharClass) -> usize {
  debug_assert_ne!(class, CharClass::Whitespace);
  let mut end = start;
  for (offset, ch) in text[start..].char_indices() {
    if char_class(ch) != class {
      break;
    }
    end = start + offset + ch.len_utf8();
  }
  end
}

fn scan_whitespace(text: &str, start: usize) -> usize {
  let mut end = start;
  let mut last_start = start;
  let mut count = 0;
  for (offset, ch) in text[start..].char_indices() {
    if char_class(ch) != CharClass::Whitespace {
      break;
    }
    last_start = start + offset;
    end = last_start + ch.len_utf8();
    count += 1;
  }

  if end < text.len() && count > 1 {
    last_start
  } else {
    end
  }
}

#[inline]
fn char_class(ch: char) -> CharClass {
  if ch.is_ascii() {
    return match ch {
      'A'..='Z' | 'a'..='z' => CharClass::Letter,
      '0'..='9' => CharClass::Number,
      '\t' | '\n' | '\x0B' | '\x0C' | '\r' | ' ' => CharClass::Whitespace,
      _ => CharClass::Other,
    };
  }
  if class_contains(&LETTER_CLASS, ch) {
    CharClass::Letter
  } else if class_contains(&NUMBER_CLASS, ch) {
    CharClass::Number
  } else if class_contains(&WHITESPACE_CLASS, ch) {
    CharClass::Whitespace
  } else {
    CharClass::Other
  }
}

#[inline]
fn class_contains(class: &ClassUnicode, ch: char) -> bool {
  class
    .ranges()
    .binary_search_by(|range| {
      if ch < range.start() {
        std::cmp::Ordering::Greater
      } else if ch > range.end() {
        std::cmp::Ordering::Less
      } else {
        std::cmp::Ordering::Equal
      }
    })
    .is_ok()
}

fn unicode_class(pattern: &str) -> ClassUnicode {
  let hir = Parser::new()
    .parse(pattern)
    .expect("default Unicode class must parse");
  match hir.kind() {
    HirKind::Class(Class::Unicode(class)) => class.clone(),
    _ => panic!("default Unicode class must compile to one Unicode class"),
  }
}
