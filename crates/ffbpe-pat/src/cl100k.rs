use super::common::{
  CharClass, case_insensitive_contraction_end, char_at, char_class, is_letter, is_other,
  scan_letters, scan_limited_numbers, scan_while, scan_whitespace_with_newlines,
};

pub(super) const PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";

pub(super) fn pretoken_end(text: &str, start: usize) -> usize {
  if text.as_bytes()[start].is_ascii_alphabetic() {
    return scan_letters(text, start + 1);
  }
  if let Some(end) = case_insensitive_contraction_end(text, start) {
    return end;
  }
  let first = char_at(text, start);
  let first_end = start + first.len_utf8();
  let class = char_class(first);
  match class {
    CharClass::Letter => return scan_letters(text, first_end),
    CharClass::Number => return scan_limited_numbers(text, start),
    _ => {}
  }
  // The optional prefix is one non-letter/number scalar, except CR/LF.
  if !matches!(first, '\r' | '\n') && first_end < text.len() {
    let next = char_at(text, first_end);
    if is_letter(next) {
      return scan_letters(text, first_end + next.len_utf8());
    }
    if first == ' ' && is_other(next) {
      return punctuation_end(text, first_end + next.len_utf8());
    }
  }
  if class == CharClass::Other {
    return punctuation_end(text, first_end);
  }
  scan_whitespace_with_newlines(text, start, true)
}

fn punctuation_end(text: &str, start: usize) -> usize {
  let end = scan_while(text, start, is_other);
  scan_while(text, end, |ch| matches!(ch, '\r' | '\n'))
}
