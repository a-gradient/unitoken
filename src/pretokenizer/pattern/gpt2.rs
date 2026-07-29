use super::{
  backend::Backend,
  common::{
    char_at, char_class, scan_same_class_with, scan_whitespace, CharClass,
  },
  engine::Pattern,
};

pub(super) const PATTERN: &str =
  r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub(super) const LEGACY_PATTERN: &str =
  r"'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub(super) const R50K_PATTERN: &str =
  r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

pub(super) fn recognizes(pattern: &str) -> bool {
  matches!(pattern, PATTERN | LEGACY_PATTERN | R50K_PATTERN)
}

pub(super) struct Gpt2;

impl Pattern for Gpt2 {
  fn pretoken_end<B: Backend>(text: &str, start: usize) -> usize {
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
      let class = char_class(char_at(text, next_start));
      if class != CharClass::Whitespace {
        return scan_same_class_with::<B>(text, next_start, class);
      }
    }

    let class = char_class(char_at(text, start));
    if class == CharClass::Whitespace {
      return scan_whitespace(text, start);
    }
    scan_same_class_with::<B>(text, start, class)
  }
}

fn contraction_len(bytes: &[u8]) -> Option<usize> {
  for suffix in [b"ll".as_slice(), b"ve", b"re", b"s", b"d", b"m", b"t"] {
    if bytes.get(1..1 + suffix.len()) == Some(suffix) {
      return Some(1 + suffix.len());
    }
  }
  None
}
