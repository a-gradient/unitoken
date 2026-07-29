use std::arch::aarch64::{
  uint8x16_t, vandq_u8, vceqq_u8, vcgeq_u8, vcleq_u8, vdupq_n_u8,
  vld1q_u8, vminvq_u8, vmvnq_u8, vorrq_u8, vst1q_u8,
};

use super::{
  AsciiPredicate, Backend, matches_ascii, scalar::scan_ascii_up_to,
};

pub(in crate::pretokenizer::pattern) struct Neon;

impl Backend for Neon {
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
    // SAFETY: Native dispatch calls this backend only after runtime NEON
    // detection. The implementation performs only in-bounds vector loads.
    unsafe { scan_neon(bytes, vector_start, predicate) }
  }
}

#[target_feature(enable = "neon")]
unsafe fn scan_neon(
  bytes: &[u8],
  start: usize,
  predicate: AsciiPredicate,
) -> usize {
  let mut end = start;
  while bytes.len().saturating_sub(end) >= 16 {
    // SAFETY: The loop condition guarantees 16 readable bytes.
    let value = unsafe { vld1q_u8(bytes.as_ptr().add(end)) };
    let accepted = accepted_neon(value, predicate);
    if vminvq_u8(accepted) != u8::MAX {
      let mut lanes = [0_u8; 16];
      // SAFETY: `lanes` has space for the complete vector.
      unsafe { vst1q_u8(lanes.as_mut_ptr(), accepted) };
      return end
        + lanes
          .iter()
          .position(|lane| *lane == 0)
          .expect("reduction found a rejected lane");
    }
    end += 16;
  }
  super::Scalar::scan_ascii(bytes, end, predicate)
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
  let letters = vorrq_u8(
    range_neon(value, b'A', b'Z'),
    range_neon(value, b'a', b'z'),
  );
  let numbers = range_neon(value, b'0', b'9');
  let whitespace = vorrq_u8(
    range_neon(value, b'\t', b'\r'),
    vceqq_u8(value, vdupq_n_u8(b' ')),
  );
  let accepted = match predicate {
    AsciiPredicate::Letter => letters,
    AsciiPredicate::Number => numbers,
    AsciiPredicate::Whitespace => whitespace,
    AsciiPredicate::Other => {
      vmvnq_u8(vorrq_u8(vorrq_u8(letters, numbers), whitespace))
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
