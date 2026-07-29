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

impl Pattern for Cl100k {
  fn pretoken_end<B: Backend>(text: &str, start: usize) -> usize {
    if let Some(end) = case_insensitive_contraction_end(text, start) {
      return end;
    }
    if let Some(end) = letter_end::<B>(text, start) {
      return end;
    }

    let first = char_at(text, start);
    if is_number(first) {
      return scan_limited_numbers(text, start);
    }
    if let Some(end) = punctuation_end::<B>(text, start) {
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

fn letter_end<B: Backend>(text: &str, start: usize) -> Option<usize> {
  let first = char_at(text, start);
  let word_start = if !matches!(first, '\r' | '\n') && !is_letter(first) && !is_number(first) {
    next_boundary(text, start)
  } else {
    start
  };
  if word_start >= text.len() || !is_letter(char_at(text, word_start)) {
    return None;
  }
  Some(scan_predicate::<B>(
    text,
    word_start,
    AsciiPredicate::Letter,
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

fn punctuation_end<B: Backend>(
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
    scan_predicate::<B>(text, word_start, AsciiPredicate::Other);
  Some(scan_predicate::<B>(
    text,
    punctuation_end,
    AsciiPredicate::CrLf,
  ))
}
