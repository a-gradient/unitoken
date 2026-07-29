use super::{AsciiPredicate, Backend, is_pattern_whitespace};

pub(in crate::pretokenizer::pattern) struct Scalar;

impl Backend for Scalar {
  #[inline]
  fn scan_ascii(
    bytes: &[u8],
    start: usize,
    predicate: AsciiPredicate,
  ) -> usize {
    scan_ascii_to(bytes, start, bytes.len(), predicate)
  }
}

#[inline]
pub(in crate::pretokenizer::pattern) fn scan_ascii_up_to(
  bytes: &[u8],
  start: usize,
  predicate: AsciiPredicate,
  max_bytes: usize,
) -> usize {
  let limit = bytes.len().min(start.saturating_add(max_bytes));
  scan_ascii_to(bytes, start, limit, predicate)
}

#[inline]
fn scan_ascii_to(
  bytes: &[u8],
  start: usize,
  limit: usize,
  predicate: AsciiPredicate,
) -> usize {
  match predicate {
    AsciiPredicate::Letter => scan_with(
      bytes,
      start,
      limit,
      |byte| byte.is_ascii_alphabetic(),
    ),
    AsciiPredicate::Number => {
      scan_with(bytes, start, limit, |byte| byte.is_ascii_digit())
    }
    AsciiPredicate::Whitespace => scan_with(
      bytes,
      start,
      limit,
      is_pattern_whitespace,
    ),
    AsciiPredicate::Other => scan_with(bytes, start, limit, |byte| {
      byte.is_ascii()
        && !byte.is_ascii_alphabetic()
        && !byte.is_ascii_digit()
        && !is_pattern_whitespace(byte)
    }),
    AsciiPredicate::Uppercase => scan_with(
      bytes,
      start,
      limit,
      |byte| byte.is_ascii_uppercase(),
    ),
    AsciiPredicate::Lowercase => scan_with(
      bytes,
      start,
      limit,
      |byte| byte.is_ascii_lowercase(),
    ),
    AsciiPredicate::CrLf => scan_with(bytes, start, limit, |byte| {
      matches!(byte, b'\r' | b'\n')
    }),
    AsciiPredicate::CrLfOrSlash => {
      scan_with(bytes, start, limit, |byte| {
        matches!(byte, b'\r' | b'\n' | b'/')
      })
    }
  }
}

#[inline]
fn scan_with(
  bytes: &[u8],
  start: usize,
  limit: usize,
  predicate: impl Fn(u8) -> bool,
) -> usize {
  let mut end = start;
  while end < limit && predicate(bytes[end]) {
    end += 1;
  }
  end
}
