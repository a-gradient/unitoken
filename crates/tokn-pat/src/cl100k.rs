use super::common::{
  case_insensitive_contraction_end, char_at, is_letter, is_number, is_other,
  is_whitespace, next_boundary, scan_through_last_newline, scan_whitespace,
  scan_while,
};

pub(super) const PATTERN: &str =
  r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";

pub(super) fn pretoken_end(text: &str, start: usize) -> usize {
  if let Some(end) = case_insensitive_contraction_end(text, start) {
    return end;
  }
  if let Some(end) = letter_end(text, start) {
    return end;
  }

  let first = char_at(text, start);
  if is_number(first) {
    return scan_limited_numbers(text, start);
  }
  if let Some(end) = punctuation_end(text, start) {
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

fn letter_end(text: &str, start: usize) -> Option<usize> {
  let first = char_at(text, start);
  let word_start = if !matches!(first, '\r' | '\n') && !is_letter(first) && !is_number(first) {
    next_boundary(text, start)
  } else {
    start
  };
  if word_start >= text.len() || !is_letter(char_at(text, word_start)) {
    return None;
  }
  Some(scan_while(text, word_start, is_letter))
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

fn punctuation_end(text: &str, start: usize) -> Option<usize> {
  let word_start = if text.as_bytes()[start] == b' ' {
    start + 1
  } else {
    start
  };
  if word_start >= text.len() || !is_other(char_at(text, word_start)) {
    return None;
  }
  let punctuation_end = scan_while(text, word_start, is_other);
  Some(scan_while(text, punctuation_end, |ch| matches!(ch, '\r' | '\n')))
}
