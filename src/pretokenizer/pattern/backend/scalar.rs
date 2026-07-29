use super::{AsciiPredicate, Backend, is_pattern_whitespace};

#[derive(Default)]
pub(in crate::pretokenizer::pattern) struct Scalar;

impl Backend for Scalar {
  #[inline]
  fn scan_ascii(
    &mut self,
    bytes: &[u8],
    start: usize,
    predicate: AsciiPredicate,
  ) -> usize {
    let mut end = start;
    match predicate {
      AsciiPredicate::Letter => {
        while end < bytes.len() && bytes[end].is_ascii_alphabetic() {
          end += 1;
        }
      }
      AsciiPredicate::Number => {
        while end < bytes.len() && bytes[end].is_ascii_digit() {
          end += 1;
        }
      }
      AsciiPredicate::Whitespace => {
        while end < bytes.len() && is_pattern_whitespace(bytes[end]) {
          end += 1;
        }
      }
      AsciiPredicate::Other => {
        while end < bytes.len()
          && bytes[end].is_ascii()
          && !bytes[end].is_ascii_alphabetic()
          && !bytes[end].is_ascii_digit()
          && !is_pattern_whitespace(bytes[end])
        {
          end += 1;
        }
      }
      AsciiPredicate::Uppercase => {
        while end < bytes.len() && bytes[end].is_ascii_uppercase() {
          end += 1;
        }
      }
      AsciiPredicate::Lowercase => {
        while end < bytes.len() && bytes[end].is_ascii_lowercase() {
          end += 1;
        }
      }
      AsciiPredicate::CrLfOrSlash => {
        while end < bytes.len()
          && matches!(bytes[end], b'\r' | b'\n' | b'/')
        {
          end += 1;
        }
      }
    }
    end
  }
}
