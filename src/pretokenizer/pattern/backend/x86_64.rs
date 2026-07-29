use std::arch::x86_64::{
  __m128i, __m256i, _mm_and_si128, _mm_cmpeq_epi8, _mm_loadu_si128,
  _mm_max_epu8, _mm_min_epu8, _mm_movemask_epi8, _mm_or_si128,
  _mm_set1_epi8, _mm_setzero_si128, _mm256_and_si256,
  _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_max_epu8,
  _mm256_min_epu8, _mm256_movemask_epi8, _mm256_or_si256,
  _mm256_set1_epi8, _mm256_setzero_si256,
};

use super::{
  AsciiPredicate, Backend, Scalar, matches_ascii,
  scalar::scan_ascii_up_to,
};

pub(in crate::pretokenizer::pattern) struct Sse2;
pub(in crate::pretokenizer::pattern) struct Avx2;

impl Backend for Sse2 {
  #[inline]
  fn scan_ascii(
    bytes: &[u8],
    start: usize,
    predicate: AsciiPredicate,
  ) -> usize {
    let vector_start = scan_ascii_up_to(bytes, start, predicate, 16);
    if vector_start == bytes.len()
      || vector_start - start < 16
      || !matches_ascii(bytes[vector_start], predicate)
    {
      return vector_start;
    }
    // SAFETY: SSE2 is part of the x86-64 baseline. The implementation performs
    // only unaligned loads that stay within `bytes`.
    unsafe { scan_sse2(bytes, vector_start, predicate) }
  }
}

impl Backend for Avx2 {
  #[inline]
  fn scan_ascii(
    bytes: &[u8],
    start: usize,
    predicate: AsciiPredicate,
  ) -> usize {
    let vector_start = scan_ascii_up_to(bytes, start, predicate, 32);
    if vector_start == bytes.len()
      || vector_start - start < 32
      || !matches_ascii(bytes[vector_start], predicate)
    {
      return vector_start;
    }
    // SAFETY: Native dispatch calls this backend only after runtime AVX2
    // detection. Tests apply the same guard.
    unsafe { scan_avx2(bytes, vector_start, predicate) }
  }
}

#[target_feature(enable = "sse2")]
unsafe fn scan_sse2(
  bytes: &[u8],
  start: usize,
  predicate: AsciiPredicate,
) -> usize {
  let mut end = start;
  while bytes.len().saturating_sub(end) >= 16 {
    // SAFETY: The loop condition guarantees 16 readable bytes.
    let value = unsafe {
      _mm_loadu_si128(bytes.as_ptr().add(end).cast::<__m128i>())
    };
    let accepted = accepted_sse2(value, predicate);
    let mask = _mm_movemask_epi8(accepted) as u32 & 0xffff;
    let rejected = !mask & 0xffff;
    if rejected != 0 {
      return end + rejected.trailing_zeros() as usize;
    }
    end += 16;
  }
  Scalar::scan_ascii(bytes, end, predicate)
}

#[target_feature(enable = "avx2")]
unsafe fn scan_avx2(
  bytes: &[u8],
  start: usize,
  predicate: AsciiPredicate,
) -> usize {
  let mut end = start;
  while bytes.len().saturating_sub(end) >= 32 {
    // SAFETY: The loop condition guarantees 32 readable bytes.
    let value = unsafe {
      _mm256_loadu_si256(bytes.as_ptr().add(end).cast::<__m256i>())
    };
    let accepted = accepted_avx2(value, predicate);
    let mask = _mm256_movemask_epi8(accepted) as u32;
    let rejected = !mask;
    if rejected != 0 {
      return end + rejected.trailing_zeros() as usize;
    }
    end += 32;
  }
  Sse2::scan_ascii(bytes, end, predicate)
}

#[target_feature(enable = "sse2")]
fn accepted_sse2(
  value: __m128i,
  predicate: AsciiPredicate,
) -> __m128i {
  let ascii = _mm_cmpeq_epi8(
    _mm_and_si128(value, _mm_set1_epi8(0x80_u8 as i8)),
    _mm_setzero_si128(),
  );
  let letters = _mm_or_si128(
    range_sse2(value, b'A', b'Z'),
    range_sse2(value, b'a', b'z'),
  );
  let numbers = range_sse2(value, b'0', b'9');
  let whitespace = _mm_or_si128(
    range_sse2(value, b'\t', b'\r'),
    _mm_cmpeq_epi8(value, _mm_set1_epi8(b' ' as i8)),
  );
  let accepted = match predicate {
    AsciiPredicate::Letter => letters,
    AsciiPredicate::Number => numbers,
    AsciiPredicate::Whitespace => whitespace,
    AsciiPredicate::Other => {
      let classified = _mm_or_si128(
        _mm_or_si128(letters, numbers),
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
  _mm_and_si128(ascii, accepted)
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
fn accepted_avx2(
  value: __m256i,
  predicate: AsciiPredicate,
) -> __m256i {
  let ascii = _mm256_cmpeq_epi8(
    _mm256_and_si256(value, _mm256_set1_epi8(0x80_u8 as i8)),
    _mm256_setzero_si256(),
  );
  let letters = _mm256_or_si256(
    range_avx2(value, b'A', b'Z'),
    range_avx2(value, b'a', b'z'),
  );
  let numbers = range_avx2(value, b'0', b'9');
  let whitespace = _mm256_or_si256(
    range_avx2(value, b'\t', b'\r'),
    _mm256_cmpeq_epi8(value, _mm256_set1_epi8(b' ' as i8)),
  );
  let accepted = match predicate {
    AsciiPredicate::Letter => letters,
    AsciiPredicate::Number => numbers,
    AsciiPredicate::Whitespace => whitespace,
    AsciiPredicate::Other => {
      let classified = _mm256_or_si256(
        _mm256_or_si256(letters, numbers),
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
  _mm256_and_si256(ascii, accepted)
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
