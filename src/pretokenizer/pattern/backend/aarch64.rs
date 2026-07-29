use std::arch::aarch64::{
  uint8x16_t, vandq_u8, vceqq_u8, vcgeq_u8, vcleq_u8,
  vdupq_n_u8, vget_high_u8, vget_low_u8, vld1q_u8, vmul_u8,
  vmvnq_u8, vorrq_u8, vshrq_n_u8, vaddv_u8, vcreate_u8,
};

use super::{
  AsciiPredicate, Backend, BlockCache, classify_scalar_predicate,
};

#[derive(Default)]
pub(in crate::pretokenizer::pattern) struct Neon {
  cache: BlockCache,
}

impl Backend for Neon {
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
          // SAFETY: Native dispatch and tests select this backend only after
          // runtime NEON detection. The length check guarantees one complete
          // load; `vld1q_u8` accepts unaligned addresses.
          unsafe { classify_neon(bytes, start, predicate) }
        }
      },
    );
    if end == bytes.len() || end % 16 != 0 {
      return end;
    }
    // SAFETY: Native dispatch and tests select this backend only after runtime
    // NEON detection. `end` is a complete block boundary.
    unsafe { scan_neon(bytes, end, predicate) }
  }
}

#[target_feature(enable = "neon")]
unsafe fn scan_neon(
  bytes: &[u8],
  start: usize,
  predicate: AsciiPredicate,
) -> usize {
  let mut end = start;
  while bytes.len() - end >= 16 {
    // SAFETY: The loop condition guarantees one complete load.
    let value = unsafe { vld1q_u8(bytes.as_ptr().add(end)) };
    let accepted = accepted_neon(value, predicate);
    if std::arch::aarch64::vminvq_u8(accepted) != u8::MAX {
      let mask = mask_neon(accepted);
      return end + (!mask & 0xffff).trailing_zeros() as usize;
    }
    end += 16;
  }
  let mask = classify_scalar_predicate(bytes, end, 16, predicate);
  let valid_len = bytes.len() - end;
  let valid = if valid_len == 16 {
    0xffff
  } else {
    (1_u32 << valid_len) - 1
  };
  let rejected = valid & !mask;
  if rejected == 0 {
    bytes.len()
  } else {
    end + rejected.trailing_zeros() as usize
  }
}

#[target_feature(enable = "neon")]
unsafe fn classify_neon(
  bytes: &[u8],
  start: usize,
  predicate: AsciiPredicate,
) -> u32 {
  // SAFETY: The caller guarantees 16 readable bytes.
  let value = unsafe { vld1q_u8(bytes.as_ptr().add(start)) };
  let accepted = accepted_neon(value, predicate);
  mask_neon(accepted)
}

#[target_feature(enable = "neon")]
fn accepted_neon(
  value: uint8x16_t,
  predicate: AsciiPredicate,
) -> uint8x16_t {
  let ascii = vceqq_u8(
    vandq_u8(value, vdupq_n_u8(0x80)),
    vdupq_n_u8(0),
  );
  let accepted = match predicate {
    AsciiPredicate::Letter => vorrq_u8(
      range_neon(value, b'A', b'Z'),
      range_neon(value, b'a', b'z'),
    ),
    AsciiPredicate::Number => range_neon(value, b'0', b'9'),
    AsciiPredicate::Whitespace => vorrq_u8(
      range_neon(value, b'\t', b'\r'),
      vceqq_u8(value, vdupq_n_u8(b' ')),
    ),
    AsciiPredicate::Other => {
      let letters = vorrq_u8(
        range_neon(value, b'A', b'Z'),
        range_neon(value, b'a', b'z'),
      );
      let whitespace = vorrq_u8(
        range_neon(value, b'\t', b'\r'),
        vceqq_u8(value, vdupq_n_u8(b' ')),
      );
      vmvnq_u8(vorrq_u8(
        vorrq_u8(letters, range_neon(value, b'0', b'9')),
        whitespace,
      ))
    }
    AsciiPredicate::Uppercase => range_neon(value, b'A', b'Z'),
    AsciiPredicate::Lowercase => range_neon(value, b'a', b'z'),
    AsciiPredicate::CrLf => vorrq_u8(
      vceqq_u8(value, vdupq_n_u8(b'\r')),
      vceqq_u8(value, vdupq_n_u8(b'\n')),
    ),
    AsciiPredicate::CrLfOrSlash => vorrq_u8(
      vorrq_u8(
        vceqq_u8(value, vdupq_n_u8(b'\r')),
        vceqq_u8(value, vdupq_n_u8(b'\n')),
      ),
      vceqq_u8(value, vdupq_n_u8(b'/')),
    ),
  };
  vandq_u8(ascii, accepted)
}

#[target_feature(enable = "neon")]
fn mask_neon(value: uint8x16_t) -> u32 {
  let bits = vshrq_n_u8::<7>(value);
  let weights = vcreate_u8(0x8040_2010_0804_0201);
  let low = vaddv_u8(vmul_u8(vget_low_u8(bits), weights)) as u32;
  let high = vaddv_u8(vmul_u8(vget_high_u8(bits), weights)) as u32;
  low | high << 8
}

#[target_feature(enable = "neon")]
fn range_neon(
  value: uint8x16_t,
  lower: u8,
  upper: u8,
) -> uint8x16_t {
  vandq_u8(
    vcgeq_u8(value, vdupq_n_u8(lower)),
    vcleq_u8(value, vdupq_n_u8(upper)),
  )
}
