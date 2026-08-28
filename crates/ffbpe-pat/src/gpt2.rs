use super::common::{
  CharClass, char_at, char_class, scan_letters, scan_same_class, scan_whitespace,
};

pub(super) const PATTERN: &str =
  r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub(super) const LEGACY_PATTERN: &str =
  r"'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub(super) const R50K_PATTERN: &str =
  r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

pub(super) fn pretoken_end(text: &str, start: usize) -> usize {
  let bytes = text.as_bytes();
  if bytes[start].is_ascii_alphabetic() {
    return scan_letters(text, start + 1);
  }
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
    if bytes[next_start].is_ascii_alphabetic() {
      return scan_letters(text, next_start + 1);
    }
    let next = char_at(text, next_start);
    let class = char_class(next);
    if class != CharClass::Whitespace {
      return scan_same_class(text, next_start + next.len_utf8(), class);
    }
  }

  let first = char_at(text, start);
  let class = char_class(first);
  if class == CharClass::Whitespace {
    return scan_whitespace(text, start);
  }
  scan_same_class(text, start + first.len_utf8(), class)
}

fn contraction_len(bytes: &[u8]) -> Option<usize> {
  match bytes.get(1)? {
    b's' | b'd' | b'm' | b't' => Some(2),
    b'l' if bytes.get(2) == Some(&b'l') => Some(3),
    b'v' | b'r' if bytes.get(2) == Some(&b'e') => Some(3),
    _ => None,
  }
}
