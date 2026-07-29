use lazy_static::lazy_static;
use regex_syntax::{
  hir::{Class, ClassUnicode, HirKind},
  Parser,
};

use super::backend::{AsciiPredicate, Backend};

lazy_static! {
  static ref CHAR_CLASSES: Box<[u8]> = unicode_class_table();
  static ref CASE_D: ClassUnicode = unicode_class(r"(?i:d)");
  static ref CASE_E: ClassUnicode = unicode_class(r"(?i:e)");
  static ref CASE_L: ClassUnicode = unicode_class(r"(?i:l)");
  static ref CASE_M: ClassUnicode = unicode_class(r"(?i:m)");
  static ref CASE_R: ClassUnicode = unicode_class(r"(?i:r)");
  static ref CASE_S: ClassUnicode = unicode_class(r"(?i:s)");
  static ref CASE_T: ClassUnicode = unicode_class(r"(?i:t)");
  static ref CASE_V: ClassUnicode = unicode_class(r"(?i:v)");
}

const LETTER: u8 = 1 << 0;
const NUMBER: u8 = 1 << 1;
const WHITESPACE: u8 = 1 << 2;
const O200K_UPPER_OR_SHARED: u8 = 1 << 3;
const O200K_LOWER_OR_SHARED: u8 = 1 << 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CharClass {
  Letter,
  Number,
  Whitespace,
  Other,
}

#[inline]
pub(super) fn char_class(ch: char) -> CharClass {
  if ch.is_ascii() {
    return match ch {
      'A'..='Z' | 'a'..='z' => CharClass::Letter,
      '0'..='9' => CharClass::Number,
      '\t' | '\n' | '\x0B' | '\x0C' | '\r' | ' ' => CharClass::Whitespace,
      _ => CharClass::Other,
    };
  }
  let classes = CHAR_CLASSES[ch as usize];
  if classes & LETTER != 0 {
    CharClass::Letter
  } else if classes & NUMBER != 0 {
    CharClass::Number
  } else if classes & WHITESPACE != 0 {
    CharClass::Whitespace
  } else {
    CharClass::Other
  }
}

#[inline]
pub(super) fn is_letter(ch: char) -> bool {
  char_class(ch) == CharClass::Letter
}

#[inline]
pub(super) fn is_number(ch: char) -> bool {
  char_class(ch) == CharClass::Number
}

#[inline]
pub(super) fn is_whitespace(ch: char) -> bool {
  char_class(ch) == CharClass::Whitespace
}

#[inline]
pub(super) fn is_other(ch: char) -> bool {
  char_class(ch) == CharClass::Other
}

#[inline]
pub(super) fn is_o200k_upper_or_shared(ch: char) -> bool {
  if ch.is_ascii() {
    ch.is_ascii_uppercase()
  } else {
    CHAR_CLASSES[ch as usize] & O200K_UPPER_OR_SHARED != 0
  }
}

#[inline]
pub(super) fn is_o200k_lower_or_shared(ch: char) -> bool {
  if ch.is_ascii() {
    ch.is_ascii_lowercase()
  } else {
    CHAR_CLASSES[ch as usize] & O200K_LOWER_OR_SHARED != 0
  }
}

pub(super) fn char_at(text: &str, start: usize) -> char {
  text[start..]
    .chars()
    .next()
    .expect("start is before end of text")
}

pub(super) fn next_boundary(text: &str, start: usize) -> usize {
  start + char_at(text, start).len_utf8()
}

pub(super) fn scan_while(
  text: &str,
  start: usize,
  mut predicate: impl FnMut(char) -> bool,
) -> usize {
  let mut end = start;
  for (offset, ch) in text[start..].char_indices() {
    if !predicate(ch) {
      break;
    }
    end = start + offset + ch.len_utf8();
  }
  end
}

pub(super) fn scan_same_class(
  text: &str,
  start: usize,
  class: CharClass,
) -> usize {
  debug_assert_ne!(class, CharClass::Whitespace);
  scan_while(text, start, |ch| char_class(ch) == class)
}

#[inline]
pub(super) fn scan_same_class_with<B: Backend>(
  text: &str,
  start: usize,
  class: CharClass,
  backend: &mut B,
) -> usize {
  debug_assert_ne!(class, CharClass::Whitespace);
  if !text.as_bytes()[start].is_ascii() {
    return scan_while(text, start, |ch| char_class(ch) == class);
  }
  scan_predicate(text, start, predicate_for_class(class), backend)
}

#[inline]
pub(super) fn scan_predicate<B: Backend>(
  text: &str,
  start: usize,
  predicate: AsciiPredicate,
  backend: &mut B,
) -> usize {
  if start == text.len() {
    return start;
  }
  if !text.as_bytes()[start].is_ascii() {
    return scan_unicode_predicate(text, start, predicate);
  }
  let ascii_end = backend.scan_ascii(text.as_bytes(), start, predicate);
  if ascii_end == text.len() || text.as_bytes()[ascii_end].is_ascii() {
    return ascii_end;
  }
  scan_unicode_predicate(text, ascii_end, predicate)
}

pub(super) fn scan_whitespace(text: &str, start: usize) -> usize {
  let mut end = start;
  let mut last_start = start;
  let mut count = 0;
  for (offset, ch) in text[start..].char_indices() {
    if !is_whitespace(ch) {
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

pub(super) fn scan_through_last_newline(text: &str, start: usize) -> Option<usize> {
  let mut last_newline_end = None;
  for (offset, ch) in text[start..].char_indices() {
    if !is_whitespace(ch) {
      break;
    }
    if matches!(ch, '\r' | '\n') {
      last_newline_end = Some(start + offset + ch.len_utf8());
    }
  }
  last_newline_end
}

pub(super) fn case_insensitive_contraction_end(text: &str, start: usize) -> Option<usize> {
  if char_at(text, start) != '\'' {
    return None;
  }
  for suffix in ["ll", "ve", "re", "s", "d", "m", "t"] {
    let mut end = start + 1;
    let mut matched = true;
    for expected in suffix.chars() {
      if end >= text.len() {
        matched = false;
        break;
      }
      let actual = char_at(text, end);
      if !case_char_matches(actual, expected) {
        matched = false;
        break;
      }
      end += actual.len_utf8();
    }
    if matched {
      return Some(end);
    }
  }
  None
}

#[inline]
fn case_char_matches(actual: char, expected: char) -> bool {
  let class = match expected {
    'd' => &*CASE_D,
    'e' => &*CASE_E,
    'l' => &*CASE_L,
    'm' => &*CASE_M,
    'r' => &*CASE_R,
    's' => &*CASE_S,
    't' => &*CASE_T,
    'v' => &*CASE_V,
    _ => return actual == expected,
  };
  class_contains(class, actual)
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

#[inline]
fn predicate_for_class(class: CharClass) -> AsciiPredicate {
  match class {
    CharClass::Letter => AsciiPredicate::Letter,
    CharClass::Number => AsciiPredicate::Number,
    CharClass::Whitespace => AsciiPredicate::Whitespace,
    CharClass::Other => AsciiPredicate::Other,
  }
}

#[inline]
fn scan_unicode_predicate(
  text: &str,
  start: usize,
  predicate: AsciiPredicate,
) -> usize {
  match predicate {
    AsciiPredicate::Letter => scan_while(text, start, is_letter),
    AsciiPredicate::Number => scan_while(text, start, is_number),
    AsciiPredicate::Whitespace => scan_while(text, start, is_whitespace),
    AsciiPredicate::Other => scan_while(text, start, is_other),
    AsciiPredicate::Uppercase => {
      scan_while(text, start, is_o200k_upper_or_shared)
    }
    AsciiPredicate::Lowercase => {
      scan_while(text, start, is_o200k_lower_or_shared)
    }
    AsciiPredicate::CrLf => {
      scan_while(text, start, |ch| matches!(ch, '\r' | '\n'))
    }
    AsciiPredicate::CrLfOrSlash => scan_while(text, start, |ch| {
      matches!(ch, '\r' | '\n' | '/')
    }),
  }
}

fn unicode_class(pattern: &str) -> ClassUnicode {
  let hir = Parser::new()
    .parse(pattern)
    .expect("known Unicode class must parse");
  match hir.kind() {
    HirKind::Class(Class::Unicode(class)) => class.clone(),
    _ => panic!("known Unicode class must compile to one Unicode class"),
  }
}

fn unicode_class_table() -> Box<[u8]> {
  // One byte per scalar keeps the regex engine's exact Unicode semantics while
  // avoiding repeated range searches in multilingual scanner hot loops.
  let mut table = vec![0_u8; char::MAX as usize + 1];
  for (pattern, flag) in [
    (r"\p{L}", LETTER),
    (r"\p{N}", NUMBER),
    (r"\s", WHITESPACE),
    (
      r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]",
      O200K_UPPER_OR_SHARED,
    ),
    (
      r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]",
      O200K_LOWER_OR_SHARED,
    ),
  ] {
    for range in unicode_class(pattern).ranges() {
      for classes in &mut table[range.start() as usize..=range.end() as usize] {
        *classes |= flag;
      }
    }
  }
  table.into_boxed_slice()
}
