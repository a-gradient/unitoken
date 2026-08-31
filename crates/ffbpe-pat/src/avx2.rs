//! x86_64 AVX2 classification for 64-byte ASCII windows.

use std::arch::x86_64::*;

use crate::simd::AsciiMasks;

pub(super) fn is_available() -> bool {
  std::is_x86_feature_detected!("avx2")
}

#[target_feature(enable = "avx2")]
#[inline(never)]
pub(super) unsafe fn ascii_masks(pointer: *const u8) -> Option<AsciiMasks> {
  // SAFETY: the shared boundary scanner supplies 64 readable bytes and calls
  // this function only after detecting AVX2 at iterator construction.
  unsafe {
    let chunks = [
      _mm256_loadu_si256(pointer.cast()),
      _mm256_loadu_si256(pointer.add(32).cast()),
    ];
    if _mm256_movemask_epi8(chunks[0]) | _mm256_movemask_epi8(chunks[1]) != 0 {
      return None;
    }

    let case_bits = _mm256_set1_epi8(0x20);
    let before_a = _mm256_set1_epi8((b'a' - 1) as i8);
    let after_z = _mm256_set1_epi8((b'z' + 1) as i8);
    let before_zero = _mm256_set1_epi8((b'0' - 1) as i8);
    let after_nine = _mm256_set1_epi8((b'9' + 1) as i8);
    let space = _mm256_set1_epi8(b' ' as i8);
    let before_whitespace = _mm256_set1_epi8(8);
    let after_whitespace = _mm256_set1_epi8(14);
    let apostrophe = _mm256_set1_epi8(b'\'' as i8);

    let letters = chunks.map(|chunk| {
      let lower = _mm256_or_si256(chunk, case_bits);
      _mm256_and_si256(
        _mm256_cmpgt_epi8(lower, before_a),
        _mm256_cmpgt_epi8(after_z, lower),
      )
    });
    let digits = chunks.map(|chunk| {
      _mm256_and_si256(
        _mm256_cmpgt_epi8(chunk, before_zero),
        _mm256_cmpgt_epi8(after_nine, chunk),
      )
    });
    let spaces = chunks.map(|chunk| _mm256_cmpeq_epi8(chunk, space));
    let whitespace = chunks.map(|chunk| {
      let control = _mm256_and_si256(
        _mm256_cmpgt_epi8(chunk, before_whitespace),
        _mm256_cmpgt_epi8(after_whitespace, chunk),
      );
      _mm256_or_si256(_mm256_cmpeq_epi8(chunk, space), control)
    });
    let apostrophes = chunks.map(|chunk| _mm256_cmpeq_epi8(chunk, apostrophe));
    Some(AsciiMasks {
      letters: movemask64(letters),
      digits: movemask64(digits),
      spaces: movemask64(spaces),
      whitespace: movemask64(whitespace),
      apostrophes: movemask64(apostrophes),
    })
  }
}

#[inline(always)]
unsafe fn movemask64(masks: [__m256i; 2]) -> u64 {
  // SAFETY: callers must already be executing with AVX2 enabled.
  unsafe {
    let [low, high] = masks;
    (_mm256_movemask_epi8(low) as u32 as u64) | ((_mm256_movemask_epi8(high) as u32 as u64) << 32)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn availability_matches_runtime_detection() {
    assert_eq!(is_available(), std::is_x86_feature_detected!("avx2"));
  }
}
