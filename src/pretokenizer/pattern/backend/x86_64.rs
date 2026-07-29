use std::arch::x86_64::{
  __m128i, __m256i, _mm_and_si128, _mm_cmpeq_epi8, _mm_loadu_si128,
  _mm_max_epu8, _mm_min_epu8, _mm_movemask_epi8, _mm_or_si128,
  _mm_set1_epi8, _mm_setzero_si128, _mm256_and_si256,
  _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_max_epu8,
  _mm256_min_epu8, _mm256_movemask_epi8, _mm256_or_si256,
  _mm256_set1_epi8, _mm256_setzero_si256,
};

use super::{
  AsciiPredicate, Backend, BlockCache, classify_scalar_predicate,
};

#[derive(Default)]
pub(in crate::pretokenizer::pattern) struct Sse2 {
  cache: BlockCache,
}

#[derive(Default)]
pub(in crate::pretokenizer::pattern) struct Avx2 {
  cache: BlockCache,
}

impl Backend for Sse2 {
  #[inline]
  fn scan_ascii(
    &mut self,
    bytes: &[u8],
    start: usize,
    predicate: AsciiPredicate,
  ) -> usize {
    let end = self.cache.scan_block::<16>(
      bytes,
      start,
      predicate,
      |bytes, start, predicate| {
        if bytes.len() - start < 16 {
          classify_scalar_predicate(bytes, start, 16, predicate)
        } else {
          // SAFETY: SSE2 is part of the x86-64 baseline, and the length check
          // guarantees one complete unaligned load.
          unsafe { classify_sse2(bytes, start, predicate) }
        }
      },
    );
    if end == bytes.len() || end % 16 != 0 {
      return end;
    }
    // SAFETY: SSE2 is part of the x86-64 baseline. `end` is a complete block
    // boundary.
    unsafe { scan_sse2(bytes, end, predicate) }
  }
}

impl Backend for Avx2 {
  #[inline]
  fn scan_ascii(
    &mut self,
    bytes: &[u8],
    start: usize,
    predicate: AsciiPredicate,
  ) -> usize {
    let end = self.cache.scan_block::<32>(
      bytes,
      start,
      predicate,
      |bytes, start, predicate| {
        if bytes.len() - start < 32 {
          classify_scalar_predicate(bytes, start, 32, predicate)
        } else {
          // SAFETY: Native dispatch and tests select this backend only after
          // runtime AVX2 detection. The length check guarantees one complete
          // unaligned load.
          unsafe { classify_avx2(bytes, start, predicate) }
        }
      },
    );
    if end == bytes.len() || end % 32 != 0 {
      return end;
    }
    // SAFETY: Native dispatch and tests select this backend only after runtime
    // AVX2 detection. `end` is a complete block boundary.
    unsafe { scan_avx2(bytes, end, predicate) }
  }
}

#[target_feature(enable = "sse2")]
unsafe fn scan_sse2(
  bytes: &[u8],
  start: usize,
  predicate: AsciiPredicate,
) -> usize {
  let mut end = start;
  while bytes.len() - end >= 16 {
    let mask = unsafe { classify_sse2(bytes, end, predicate) };
    let rejected = !mask & 0xffff;
    if rejected != 0 {
      return end + rejected.trailing_zeros() as usize;
    }
    end += 16;
  }
  scan_scalar_tail(bytes, end, predicate)
}

#[target_feature(enable = "avx2")]
unsafe fn scan_avx2(
  bytes: &[u8],
  start: usize,
  predicate: AsciiPredicate,
) -> usize {
  let mut end = start;
  while bytes.len() - end >= 32 {
    let mask = unsafe { classify_avx2(bytes, end, predicate) };
    if mask != u32::MAX {
      return end + (!mask).trailing_zeros() as usize;
    }
    end += 32;
  }
  scan_scalar_tail(bytes, end, predicate)
}

#[inline]
fn scan_scalar_tail(
  bytes: &[u8],
  start: usize,
  predicate: AsciiPredicate,
) -> usize {
  let mask = classify_scalar_predicate(bytes, start, 32, predicate);
  let valid_len = bytes.len() - start;
  let valid = if valid_len == 32 {
    u32::MAX
  } else {
    (1_u32 << valid_len) - 1
  };
  let rejected = valid & !mask;
  if rejected == 0 {
    bytes.len()
  } else {
    start + rejected.trailing_zeros() as usize
  }
}

#[target_feature(enable = "sse2")]
unsafe fn classify_sse2(
  bytes: &[u8],
  start: usize,
  predicate: AsciiPredicate,
) -> u32 {
  // SAFETY: The caller guarantees 16 readable bytes.
  let value =
    unsafe { _mm_loadu_si128(bytes.as_ptr().add(start).cast::<__m128i>()) };
  let ascii = _mm_cmpeq_epi8(
    _mm_and_si128(value, _mm_set1_epi8(0x80_u8 as i8)),
    _mm_setzero_si128(),
  );
  let accepted = match predicate {
    AsciiPredicate::Letter => _mm_or_si128(
      range_sse2(value, b'A', b'Z'),
      range_sse2(value, b'a', b'z'),
    ),
    AsciiPredicate::Number => range_sse2(value, b'0', b'9'),
    AsciiPredicate::Whitespace => _mm_or_si128(
      range_sse2(value, b'\t', b'\r'),
      _mm_cmpeq_epi8(value, _mm_set1_epi8(b' ' as i8)),
    ),
    AsciiPredicate::Other => {
      let letters = _mm_or_si128(
        range_sse2(value, b'A', b'Z'),
        range_sse2(value, b'a', b'z'),
      );
      let whitespace = _mm_or_si128(
        range_sse2(value, b'\t', b'\r'),
        _mm_cmpeq_epi8(value, _mm_set1_epi8(b' ' as i8)),
      );
      let classified = _mm_or_si128(
        _mm_or_si128(letters, range_sse2(value, b'0', b'9')),
        whitespace,
      );
      _mm_cmpeq_epi8(classified, _mm_setzero_si128())
    }
    AsciiPredicate::Uppercase => range_sse2(value, b'A', b'Z'),
    AsciiPredicate::Lowercase => range_sse2(value, b'a', b'z'),
    AsciiPredicate::CrLf => _mm_or_si128(
      _mm_cmpeq_epi8(value, _mm_set1_epi8(b'\r' as i8)),
      _mm_cmpeq_epi8(value, _mm_set1_epi8(b'\n' as i8)),
    ),
    AsciiPredicate::CrLfOrSlash => _mm_or_si128(
      _mm_or_si128(
        _mm_cmpeq_epi8(value, _mm_set1_epi8(b'\r' as i8)),
        _mm_cmpeq_epi8(value, _mm_set1_epi8(b'\n' as i8)),
      ),
      _mm_cmpeq_epi8(value, _mm_set1_epi8(b'/' as i8)),
    ),
  };
  _mm_movemask_epi8(_mm_and_si128(ascii, accepted)) as u32
    & 0xffff
}

#[target_feature(enable = "avx2")]
unsafe fn classify_avx2(
  bytes: &[u8],
  start: usize,
  predicate: AsciiPredicate,
) -> u32 {
  // SAFETY: The caller guarantees 32 readable bytes.
  let value =
    unsafe { _mm256_loadu_si256(bytes.as_ptr().add(start).cast::<__m256i>()) };
  let ascii = _mm256_cmpeq_epi8(
    _mm256_and_si256(value, _mm256_set1_epi8(0x80_u8 as i8)),
    _mm256_setzero_si256(),
  );
  let accepted = match predicate {
    AsciiPredicate::Letter => _mm256_or_si256(
      range_avx2(value, b'A', b'Z'),
      range_avx2(value, b'a', b'z'),
    ),
    AsciiPredicate::Number => range_avx2(value, b'0', b'9'),
    AsciiPredicate::Whitespace => _mm256_or_si256(
      range_avx2(value, b'\t', b'\r'),
      _mm256_cmpeq_epi8(value, _mm256_set1_epi8(b' ' as i8)),
    ),
    AsciiPredicate::Other => {
      let letters = _mm256_or_si256(
        range_avx2(value, b'A', b'Z'),
        range_avx2(value, b'a', b'z'),
      );
      let whitespace = _mm256_or_si256(
        range_avx2(value, b'\t', b'\r'),
        _mm256_cmpeq_epi8(value, _mm256_set1_epi8(b' ' as i8)),
      );
      let classified = _mm256_or_si256(
        _mm256_or_si256(
          letters,
          range_avx2(value, b'0', b'9'),
        ),
        whitespace,
      );
      _mm256_cmpeq_epi8(classified, _mm256_setzero_si256())
    }
    AsciiPredicate::Uppercase => range_avx2(value, b'A', b'Z'),
    AsciiPredicate::Lowercase => range_avx2(value, b'a', b'z'),
    AsciiPredicate::CrLf => _mm256_or_si256(
      _mm256_cmpeq_epi8(value, _mm256_set1_epi8(b'\r' as i8)),
      _mm256_cmpeq_epi8(value, _mm256_set1_epi8(b'\n' as i8)),
    ),
    AsciiPredicate::CrLfOrSlash => _mm256_or_si256(
      _mm256_or_si256(
        _mm256_cmpeq_epi8(value, _mm256_set1_epi8(b'\r' as i8)),
        _mm256_cmpeq_epi8(value, _mm256_set1_epi8(b'\n' as i8)),
      ),
      _mm256_cmpeq_epi8(value, _mm256_set1_epi8(b'/' as i8)),
    ),
  };
  _mm256_movemask_epi8(_mm256_and_si256(ascii, accepted)) as u32
}

#[target_feature(enable = "sse2")]
fn range_sse2(value: __m128i, lower: u8, upper: u8) -> __m128i {
  let lower = _mm_set1_epi8(lower as i8);
  let upper = _mm_set1_epi8(upper as i8);
  _mm_and_si128(
    _mm_cmpeq_epi8(_mm_max_epu8(value, lower), value),
    _mm_cmpeq_epi8(_mm_min_epu8(value, upper), value),
  )
}

#[target_feature(enable = "avx2")]
fn range_avx2(value: __m256i, lower: u8, upper: u8) -> __m256i {
  let lower = _mm256_set1_epi8(lower as i8);
  let upper = _mm256_set1_epi8(upper as i8);
  _mm256_and_si256(
    _mm256_cmpeq_epi8(_mm256_max_epu8(value, lower), value),
    _mm256_cmpeq_epi8(_mm256_min_epu8(value, upper), value),
  )
}
