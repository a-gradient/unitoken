//! ASCII run scanners selected once per PAT input segment.
//!
//! Every backend stops before non-ASCII input. Unicode classification therefore
//! remains in the shared scalar path, and SIMD code never has to interpret
//! UTF-8 continuation bytes.

mod scalar;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

pub(super) use scalar::Scalar;

#[cfg(target_arch = "aarch64")]
pub(super) use aarch64::Neon;
#[cfg(target_arch = "x86_64")]
pub(super) use x86_64::{Avx2, Sse2};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AsciiPredicate {
  Letter,
  Number,
  Whitespace,
  Other,
  Uppercase,
  Lowercase,
  CrLf,
  CrLfOrSlash,
}

pub(super) trait Backend {
  fn scan_ascii(
    bytes: &[u8],
    start: usize,
    predicate: AsciiPredicate,
  ) -> usize;
}

#[inline]
pub(super) fn matches_ascii(
  byte: u8,
  predicate: AsciiPredicate,
) -> bool {
  if !byte.is_ascii() {
    return false;
  }
  match predicate {
    AsciiPredicate::Letter => byte.is_ascii_alphabetic(),
    AsciiPredicate::Number => byte.is_ascii_digit(),
    AsciiPredicate::Whitespace => is_pattern_whitespace(byte),
    AsciiPredicate::Other => {
      !byte.is_ascii_alphabetic()
        && !byte.is_ascii_digit()
        && !is_pattern_whitespace(byte)
    }
    AsciiPredicate::Uppercase => byte.is_ascii_uppercase(),
    AsciiPredicate::Lowercase => byte.is_ascii_lowercase(),
    AsciiPredicate::CrLf => matches!(byte, b'\r' | b'\n'),
    AsciiPredicate::CrLfOrSlash => {
      matches!(byte, b'\r' | b'\n' | b'/')
    }
  }
}

#[inline]
pub(super) fn is_pattern_whitespace(byte: u8) -> bool {
  matches!(
    byte,
    b'\t' | b'\n' | b'\x0B' | b'\x0C' | b'\r' | b' '
  )
}

#[cfg(test)]
mod tests {
  use super::*;

  const PREDICATES: [AsciiPredicate; 8] = [
    AsciiPredicate::Letter,
    AsciiPredicate::Number,
    AsciiPredicate::Whitespace,
    AsciiPredicate::Other,
    AsciiPredicate::Uppercase,
    AsciiPredicate::Lowercase,
    AsciiPredicate::CrLf,
    AsciiPredicate::CrLfOrSlash,
  ];
  const RUN_LENGTHS: [usize; 11] =
    [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65];
  const START_OFFSETS: [usize; 6] = [0, 1, 7, 15, 16, 31];

  fn accepted_byte(predicate: AsciiPredicate) -> u8 {
    match predicate {
      AsciiPredicate::Letter => b'a',
      AsciiPredicate::Number => b'7',
      AsciiPredicate::Whitespace => b' ',
      AsciiPredicate::Other => b'!',
      AsciiPredicate::Uppercase => b'Z',
      AsciiPredicate::Lowercase => b'z',
      AsciiPredicate::CrLf => b'\n',
      AsciiPredicate::CrLfOrSlash => b'/',
    }
  }

  fn assert_matches_scalar<B: Backend>() {
    for predicate in PREDICATES {
      for run_length in RUN_LENGTHS {
        for start in START_OFFSETS {
          for candidate in 0..=u8::MAX {
            let mut bytes = vec![0_u8; start];
            bytes.extend(
              std::iter::repeat_n(
                accepted_byte(predicate),
                run_length,
              ),
            );
            bytes.push(candidate);
            bytes.extend(
              std::iter::repeat_n(accepted_byte(predicate), 65),
            );
            assert_eq!(
              B::scan_ascii(&bytes, start, predicate),
              Scalar::scan_ascii(&bytes, start, predicate),
              "predicate={predicate:?} run={run_length} start={start} candidate={candidate:#04x}",
            );
          }
        }
      }
    }
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn neon_backend_matches_scalar() {
    if std::arch::is_aarch64_feature_detected!("neon") {
      assert_matches_scalar::<Neon>();
    }
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn sse2_backend_matches_scalar() {
    assert_matches_scalar::<Sse2>();
  }

  #[cfg(target_arch = "x86_64")]
  #[test]
  fn avx2_backend_matches_scalar() {
    if std::arch::is_x86_feature_detected!("avx2") {
      assert_matches_scalar::<Avx2>();
    }
  }
}
