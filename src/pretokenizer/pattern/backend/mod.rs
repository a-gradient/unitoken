//! Byte-indexed character-class masks shared across pretoken boundaries.
//!
//! SIMD backends classify a requested property for an entire block. Pattern
//! scanners can then consume several short runs from the cached mask without
//! restarting at each pretoken boundary. Non-ASCII bytes reject every ASCII
//! predicate and remain on the scalar Unicode path.

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
#[repr(u8)]
pub(super) enum AsciiPredicate {
  Letter,
  Number,
  Whitespace,
  Other,
  Uppercase,
  Lowercase,
  CrLfOrSlash,
}

pub(super) trait Backend: Default {
  fn scan_ascii(
    &mut self,
    bytes: &[u8],
    start: usize,
    predicate: AsciiPredicate,
  ) -> usize;
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BlockCache {
  start: usize,
  valid: u32,
  masks: [u32; 7],
  classified: u8,
  initialized: bool,
}

impl BlockCache {
  #[inline]
  pub(super) fn scan_block<const WIDTH: usize>(
    &mut self,
    bytes: &[u8],
    start: usize,
    predicate: AsciiPredicate,
    mut classify: impl FnMut(
      &[u8],
      usize,
      AsciiPredicate,
    ) -> u32,
  ) -> usize {
    debug_assert!(WIDTH <= u32::BITS as usize);
    if start == bytes.len() {
      return start;
    }
    let block_start = start / WIDTH * WIDTH;
    if !self.initialized || self.start != block_start {
      self.start = block_start;
      let block_len = (bytes.len() - block_start).min(WIDTH);
      self.valid = if block_len == u32::BITS as usize {
        u32::MAX
      } else {
        (1_u32 << block_len) - 1
      };
      self.classified = 0;
      self.initialized = true;
    }

    let predicate_index = predicate as usize;
    let predicate_bit = 1_u8 << predicate_index;
    if self.classified & predicate_bit == 0 {
      self.masks[predicate_index] =
        classify(bytes, block_start, predicate) & self.valid;
      self.classified |= predicate_bit;
    }
    let offset = start - block_start;
    let valid = self.valid >> offset;
    let accepted = self.masks[predicate_index] >> offset;
    let rejected = valid & !accepted;
    if rejected != 0 {
      start + rejected.trailing_zeros() as usize
    } else {
      (block_start + WIDTH).min(bytes.len())
    }
  }
}

#[inline]
pub(super) fn classify_scalar_predicate(
  bytes: &[u8],
  start: usize,
  width: usize,
  predicate: AsciiPredicate,
) -> u32 {
  let end = bytes.len().min(start.saturating_add(width));
  let mut mask = 0_u32;
  for (lane, byte) in bytes[start..end].iter().copied().enumerate() {
    if matches_ascii(byte, predicate) {
      mask |= 1_u32 << lane;
    }
  }
  mask
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

  const PREDICATES: [AsciiPredicate; 7] = [
    AsciiPredicate::Letter,
    AsciiPredicate::Number,
    AsciiPredicate::Whitespace,
    AsciiPredicate::Other,
    AsciiPredicate::Uppercase,
    AsciiPredicate::Lowercase,
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
      AsciiPredicate::CrLfOrSlash => b'/',
    }
  }

  fn assert_matches_scalar<B: Backend>() {
    for predicate in PREDICATES {
      for run_length in RUN_LENGTHS {
        for start in START_OFFSETS {
          for candidate in 0..=u8::MAX {
            let mut bytes = vec![0_u8; start];
            bytes.extend(std::iter::repeat_n(
              accepted_byte(predicate),
              run_length,
            ));
            bytes.push(candidate);
            bytes.extend(std::iter::repeat_n(
              accepted_byte(predicate),
              65,
            ));
            let mut backend = B::default();
            let mut scalar = Scalar;
            assert_eq!(
              backend.scan_ascii(&bytes, start, predicate),
              scalar.scan_ascii(&bytes, start, predicate),
              "predicate={predicate:?} run={run_length} start={start} candidate={candidate:#04x}",
            );
          }
        }
      }
    }
  }

  #[test]
  fn block_cache_reuses_property_masks_across_pretokens() {
    let bytes = b"aaaa bbbb!";
    let mut cache = BlockCache::default();
    let mut classifications = 0;
    let mut classify = |
      bytes: &[u8],
      start: usize,
      predicate: AsciiPredicate,
    | {
      classifications += 1;
      classify_scalar_predicate(bytes, start, 16, predicate)
    };

    assert_eq!(
      cache.scan_block::<16>(
        bytes,
        0,
        AsciiPredicate::Letter,
        &mut classify,
      ),
      4,
    );
    assert_eq!(
      cache.scan_block::<16>(
        bytes,
        4,
        AsciiPredicate::Whitespace,
        &mut classify,
      ),
      5,
    );
    assert_eq!(
      cache.scan_block::<16>(
        bytes,
        5,
        AsciiPredicate::Letter,
        &mut classify,
      ),
      9,
    );
    assert_eq!(classifications, 2);
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
