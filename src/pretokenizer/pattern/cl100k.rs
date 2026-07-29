use crate::MyResult;

use super::{
  backend::{AsciiPredicate, Backend},
  common::{
    case_insensitive_contraction_end, char_at, is_letter, is_number,
    is_other, is_whitespace, next_boundary, scan_predicate,
    scan_through_last_newline, scan_whitespace,
  },
  engine::Pattern,
};

pub(super) const PATTERN: &str =
  r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";

pub(super) struct Cl100k;

pub(super) fn for_each_scalar<'a>(
  text: &'a str,
  mut emit: impl FnMut(&'a str) -> MyResult<()>,
) -> MyResult<()> {
  let mut start = 0;
  while start < text.len() {
    let end = scalar_pretoken_end(text, start);
    debug_assert!(end > start);
    debug_assert!(text.is_char_boundary(end));
    emit(&text[start..end])?;
    start = end;
  }
  Ok(())
}

fn scalar_pretoken_end(text: &str, start: usize) -> usize {
  if let Some(end) = case_insensitive_contraction_end(text, start) {
    return end;
  }
  if let Some(end) = scalar_letter_end(text, start) {
    return end;
  }

  let first = char_at(text, start);
  if is_number(first) {
    return scan_limited_numbers(text, start);
  }
  if let Some(end) = scalar_punctuation_end(text, start) {
    return end;
  }
  if is_whitespace(first) {
    if let Some(end) = scan_through_last_newline(text, start) {
      return end;
    }
    return scan_whitespace(text, start);
  }
  next_boundary(text, start)
}

fn scalar_letter_end(text: &str, start: usize) -> Option<usize> {
  let first = char_at(text, start);
  let word_start = if !matches!(first, '\r' | '\n')
    && !is_letter(first)
    && !is_number(first)
  {
    next_boundary(text, start)
  } else {
    start
  };
  if word_start >= text.len() || !is_letter(char_at(text, word_start)) {
    return None;
  }
  Some(super::common::scan_while(
    text,
    word_start,
    is_letter,
  ))
}

fn scalar_punctuation_end(
  text: &str,
  start: usize,
) -> Option<usize> {
  let word_start = if text.as_bytes()[start] == b' ' {
    start + 1
  } else {
    start
  };
  if word_start >= text.len() || !is_other(char_at(text, word_start)) {
    return None;
  }
  let punctuation_end =
    super::common::scan_while(text, word_start, is_other);
  Some(super::common::scan_while(
    text,
    punctuation_end,
    |ch| matches!(ch, '\r' | '\n'),
  ))
}

impl Pattern for Cl100k {
  #[inline(always)]
  fn pretoken_end<B: Backend>(
    text: &str,
    start: usize,
    backend: &mut B,
  ) -> usize {
    if let Some(end) = case_insensitive_contraction_end(text, start) {
      return end;
    }
    if let Some(end) = letter_end(text, start, backend) {
      return end;
    }

    let first = char_at(text, start);
    if is_number(first) {
      return scan_limited_numbers(text, start);
    }
    if let Some(end) = punctuation_end(text, start, backend) {
      return end;
    }
    if is_whitespace(first) {
      if let Some(end) = scan_through_last_newline(text, start) {
        return end;
      }
      return scan_whitespace(text, start);
    }
    next_boundary(text, start)
  }
}

#[inline]
fn letter_end<B: Backend>(
  text: &str,
  start: usize,
  backend: &mut B,
) -> Option<usize> {
  let first = char_at(text, start);
  let word_start = if !matches!(first, '\r' | '\n') && !is_letter(first) && !is_number(first) {
    next_boundary(text, start)
  } else {
    start
  };
  if word_start >= text.len() || !is_letter(char_at(text, word_start)) {
    return None;
  }
  if !text.as_bytes()[word_start].is_ascii() {
    return Some(super::common::scan_while(
      text,
      word_start,
      is_letter,
    ));
  }
  Some(scan_predicate(
    text,
    word_start,
    AsciiPredicate::Letter,
    backend,
  ))
}

fn scan_limited_numbers(text: &str, start: usize) -> usize {
  let mut end = start;
  for (count, (offset, ch)) in text[start..].char_indices().enumerate() {
    if count == 3 || !is_number(ch) {
      break;
    }
    end = start + offset + ch.len_utf8();
  }
  end
}

#[inline]
fn punctuation_end<B: Backend>(
  text: &str,
  start: usize,
  backend: &mut B,
) -> Option<usize> {
  let word_start = if text.as_bytes()[start] == b' ' {
    start + 1
  } else {
    start
  };
  if word_start >= text.len() || !is_other(char_at(text, word_start)) {
    return None;
  }
  let punctuation_end = if text.as_bytes()[word_start].is_ascii() {
    scan_predicate(
      text,
      word_start,
      AsciiPredicate::Other,
      backend,
    )
  } else {
    super::common::scan_while(text, word_start, is_other)
  };
  Some(scan_predicate(
    text,
    punctuation_end,
    AsciiPredicate::CrLf,
    backend,
  ))
}
