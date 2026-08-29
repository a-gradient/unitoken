use std::sync::LazyLock;

use regex_syntax::{
  Parser,
  hir::{Class, ClassUnicode, HirKind},
};

use super::ascii;

static CHAR_CLASSES: LazyLock<Box<[u8]>> = LazyLock::new(unicode_class_table);
static CASE_D: LazyLock<ClassUnicode> = LazyLock::new(|| unicode_class(r"(?i:d)"));
static CASE_E: LazyLock<ClassUnicode> = LazyLock::new(|| unicode_class(r"(?i:e)"));
static CASE_L: LazyLock<ClassUnicode> = LazyLock::new(|| unicode_class(r"(?i:l)"));
static CASE_M: LazyLock<ClassUnicode> = LazyLock::new(|| unicode_class(r"(?i:m)"));
static CASE_R: LazyLock<ClassUnicode> = LazyLock::new(|| unicode_class(r"(?i:r)"));
static CASE_S: LazyLock<ClassUnicode> = LazyLock::new(|| unicode_class(r"(?i:s)"));
static CASE_T: LazyLock<ClassUnicode> = LazyLock::new(|| unicode_class(r"(?i:t)"));
static CASE_V: LazyLock<ClassUnicode> = LazyLock::new(|| unicode_class(r"(?i:v)"));

const LETTER: u8 = 1 << 0;
const NUMBER: u8 = 1 << 1;
const WHITESPACE: u8 = 1 << 2;
const O200K_UPPER_OR_SHARED: u8 = 1 << 3;
const O200K_LOWER_OR_SHARED: u8 = 1 << 4;

#[derive(Clone, Copy)]
pub(super) struct ClassTable(&'static [u8]);

impl ClassTable {
  #[inline]
  pub(super) fn get() -> Self {
    Self(&CHAR_CLASSES)
  }

  #[inline(always)]
  fn flags(self, codepoint: u32) -> u8 {
    debug_assert!(codepoint <= char::MAX as u32);
    // SAFETY: UTF-8 decoding only produces Unicode scalar values, and the
    // table has one entry for every value through `char::MAX`.
    unsafe { *self.0.get_unchecked(codepoint as usize) }
  }

  #[inline(always)]
  fn is_letter(self, codepoint: u32) -> bool {
    self.flags(codepoint) & LETTER != 0
  }

  #[inline(always)]
  fn case_class(self, codepoint: u32) -> CaseClass {
    case_class_from_flags(self.flags(codepoint))
  }

  #[inline(always)]
  pub(super) fn case_class_at(self, text: &str, start: usize) -> (CaseClass, usize) {
    let bytes = text.as_bytes();
    debug_assert!(start < bytes.len());
    debug_assert!(text.is_char_boundary(start));
    debug_assert!(!bytes[start].is_ascii());
    let (codepoint, width) = decode_non_ascii(bytes, start);
    (self.case_class(codepoint), width)
  }
}

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
pub(super) fn case_class(ch: char) -> CaseClass {
  if ch.is_ascii_uppercase() {
    return CaseClass::Upper;
  }
  if ch.is_ascii_lowercase() {
    return CaseClass::Lower;
  }
  if ch.is_ascii() {
    return CaseClass::Other;
  }
  ClassTable::get().case_class(ch as u32)
}

#[inline(always)]
fn case_class_from_flags(flags: u8) -> CaseClass {
  match flags & (O200K_UPPER_OR_SHARED | O200K_LOWER_OR_SHARED) {
    O200K_UPPER_OR_SHARED => CaseClass::Upper,
    O200K_LOWER_OR_SHARED => CaseClass::Lower,
    0 => CaseClass::Other,
    _ => CaseClass::Shared,
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CaseClass {
  Upper,
  Lower,
  Shared,
  Other,
}

#[inline]
pub(super) fn char_at(text: &str, start: usize) -> char {
  let byte = text.as_bytes()[start];
  if byte.is_ascii() {
    return char::from(byte);
  }
  text[start..]
    .chars()
    .next()
    .expect("start is before end of text")
}

#[inline]
pub(super) fn next_boundary(text: &str, start: usize) -> usize {
  start + char_at(text, start).len_utf8()
}

#[inline]
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

#[inline]
pub(super) fn scan_letters(text: &str, start: usize) -> usize {
  let bytes = text.as_bytes();
  let start = ascii::scan_letters(bytes, start);
  if start == bytes.len() || bytes[start].is_ascii() {
    return start;
  }
  scan_unicode_letters(text, start)
}

// Keep the Unicode state machine out of the frequent ASCII call sites.
#[inline(never)]
fn scan_unicode_letters(text: &str, mut start: usize) -> usize {
  const HIGH_BITS: u64 = 0x8080_8080_8080_8080;
  let bytes = text.as_bytes();
  let mut classes = None;
  loop {
    // Direct decoding wins on dense multilingual runs, but adds overhead to
    // short script transitions. Use it only when the next eight bytes are all
    // non-ASCII; otherwise Rust's UTF-8 iterator handles the short run.
    if bytes.len() - start >= 8 {
      let word = u64::from_ne_bytes(bytes[start..start + 8].try_into().unwrap());
      if word & HIGH_BITS == HIGH_BITS {
        let classes = *classes.get_or_insert_with(ClassTable::get);
        while start < bytes.len() && !bytes[start].is_ascii() {
          let (codepoint, width) = decode_non_ascii(bytes, start);
          if !classes.is_letter(codepoint) {
            return start;
          }
          start += width;
        }
        start = ascii::scan_letters(bytes, start);
        if start == bytes.len() || bytes[start].is_ascii() {
          return start;
        }
        continue;
      }
    }

    // Decode a contiguous non-ASCII run without restarting the iterator at
    // each scalar; return to SWAR when ASCII letters resume.
    let mut end = start;
    for (offset, ch) in text[start..].char_indices() {
      if ch.is_ascii() {
        if !ch.is_ascii_alphabetic() {
          return start + offset;
        }
        break;
      }
      if !is_letter(ch) {
        return start + offset;
      }
      end = start + offset + ch.len_utf8();
    }
    if end == text.len() {
      return end;
    }
    start = ascii::scan_letters(bytes, end);
    if start == bytes.len() || bytes[start].is_ascii() {
      return start;
    }
  }
}

/// Decode the non-ASCII scalar beginning at `start`.
///
/// Callers only pass byte slices borrowed from `str` and maintain `start` at
/// a character boundary, so the leading byte determines a complete 2-4 byte
/// sequence contained in `bytes`.
#[inline(always)]
fn decode_non_ascii(bytes: &[u8], start: usize) -> (u32, usize) {
  debug_assert!(start < bytes.len());
  debug_assert!(bytes[start] >= 0x80);
  // SAFETY: the function's invariant gives a complete, valid UTF-8 sequence
  // at `start`; every indexed continuation byte is therefore in bounds.
  unsafe {
    let first = u32::from(*bytes.get_unchecked(start));
    let second = u32::from(*bytes.get_unchecked(start + 1) & 0x3f);
    if first < 0xe0 {
      return (((first & 0x1f) << 6) | second, 2);
    }
    let third = u32::from(*bytes.get_unchecked(start + 2) & 0x3f);
    if first < 0xf0 {
      return (((first & 0x0f) << 12) | (second << 6) | third, 3);
    }
    let fourth = u32::from(*bytes.get_unchecked(start + 3) & 0x3f);
    (
      ((first & 0x07) << 18) | (second << 12) | (third << 6) | fourth,
      4,
    )
  }
}

#[inline]
pub(super) fn scan_same_class(text: &str, start: usize, class: CharClass) -> usize {
  debug_assert_ne!(class, CharClass::Whitespace);
  match class {
    CharClass::Letter => scan_letters(text, start),
    CharClass::Number => scan_while(text, start, is_number),
    CharClass::Other => scan_while(text, start, is_other),
    CharClass::Whitespace => unreachable!("whitespace has separate boundary rules"),
  }
}

pub(super) fn scan_whitespace(text: &str, start: usize) -> usize {
  WhitespaceRun::scan(text, start).without_newlines(text.len(), start)
}

pub(super) fn scan_whitespace_with_newlines(
  text: &str,
  start: usize,
  trailing_whitespace_first: bool,
) -> usize {
  let run = WhitespaceRun::scan(text, start);
  // cl100k's `\s++$` precedes the newline branch; o200k has no such branch.
  if trailing_whitespace_first && run.end == text.len() {
    return run.end;
  }
  run
    .last_newline_end
    .unwrap_or_else(|| run.without_newlines(text.len(), start))
}

struct WhitespaceRun {
  end: usize,
  last_start: usize,
  last_newline_end: Option<usize>,
}

impl WhitespaceRun {
  fn scan(text: &str, start: usize) -> Self {
    let mut run = Self {
      end: start,
      last_start: start,
      last_newline_end: None,
    };
    for (offset, ch) in text[start..].char_indices() {
      if !is_whitespace(ch) {
        break;
      }
      run.last_start = start + offset;
      run.end = run.last_start + ch.len_utf8();
      if matches!(ch, '\r' | '\n') {
        run.last_newline_end = Some(run.end);
      }
    }
    run
  }

  fn without_newlines(&self, text_len: usize, start: usize) -> usize {
    if self.end < text_len && self.last_start > start {
      self.last_start
    } else {
      self.end
    }
  }
}

#[inline]
pub(super) fn case_insensitive_contraction_end(text: &str, start: usize) -> Option<usize> {
  let bytes = text.as_bytes();
  if bytes.get(start) != Some(&b'\'') {
    return None;
  }
  let first = bytes.get(start + 1)?.to_ascii_lowercase();
  match first {
    b's' | b'd' | b'm' | b't' => Some(start + 2),
    b'l' | b'v' | b'r' => {
      let expected = if first == b'l' { 'l' } else { 'e' };
      let &second = bytes.get(start + 2)?;
      if second.is_ascii() {
        (char::from(second.to_ascii_lowercase()) == expected).then_some(start + 3)
      } else {
        let actual = char_at(text, start + 2);
        case_char_matches(actual, expected).then_some(start + 2 + actual.len_utf8())
      }
    }
    _ if !first.is_ascii() => unicode_contraction_end(text, start),
    _ => None,
  }
}

fn unicode_contraction_end(text: &str, start: usize) -> Option<usize> {
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
  if actual.is_ascii() {
    return actual.to_ascii_lowercase() == expected;
  }
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

pub(super) fn scan_limited_numbers(text: &str, start: usize) -> usize {
  let mut end = start;
  for (offset, ch) in text[start..].char_indices().take(3) {
    if !is_number(ch) {
      break;
    }
    end = start + offset + ch.len_utf8();
  }
  end
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
    (r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]", O200K_UPPER_OR_SHARED),
    (r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]", O200K_LOWER_OR_SHARED),
  ] {
    for range in unicode_class(pattern).ranges() {
      for classes in &mut table[range.start() as usize..=range.end() as usize] {
        *classes |= flag;
      }
    }
  }
  table.into_boxed_slice()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn non_ascii_decoder_matches_every_unicode_scalar() {
    let mut buffer = [0_u8; 4];
    for codepoint in 0x80..=char::MAX as u32 {
      let Some(ch) = char::from_u32(codepoint) else {
        continue;
      };
      let text = ch.encode_utf8(&mut buffer);
      assert_eq!(
        decode_non_ascii(text.as_bytes(), 0),
        (codepoint, text.len())
      );
    }
  }
}
