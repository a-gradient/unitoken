//! AArch64 NEON classification for 64-byte ASCII windows.

use std::arch::aarch64::*;

use crate::simd::AsciiMasks;

#[inline(never)]
pub(super) unsafe fn ascii_masks(pointer: *const u8) -> Option<AsciiMasks> {
  // SAFETY: the shared boundary scanner supplies 64 readable bytes. NEON is a
  // baseline target feature for Rust's supported AArch64 targets.
  unsafe {
    let chunks = [
      vld1q_u8(pointer),
      vld1q_u8(pointer.add(16)),
      vld1q_u8(pointer.add(32)),
      vld1q_u8(pointer.add(48)),
    ];
    let high = chunks.map(|chunk| vcltzq_s8(vreinterpretq_s8_u8(chunk)));
    let any_high = vorrq_u8(vorrq_u8(high[0], high[1]), vorrq_u8(high[2], high[3]));
    if vmaxvq_u8(any_high) != 0 {
      return None;
    }

    let case_bits = vdupq_n_u8(0x20);
    let lower_a = vdupq_n_u8(b'a');
    let letter_width = vdupq_n_u8(25);
    let zero = vdupq_n_u8(b'0');
    let digit_width = vdupq_n_u8(9);
    let space = vdupq_n_u8(b' ');
    let whitespace_start = vdupq_n_u8(9);
    let whitespace_width = vdupq_n_u8(4);
    let carriage_return = vdupq_n_u8(b'\r');
    let line_feed = vdupq_n_u8(b'\n');
    let apostrophe = vdupq_n_u8(b'\'');

    let letters =
      chunks.map(|chunk| vcleq_u8(vsubq_u8(vorrq_u8(chunk, case_bits), lower_a), letter_width));
    let digits = chunks.map(|chunk| vcleq_u8(vsubq_u8(chunk, zero), digit_width));
    let spaces = chunks.map(|chunk| vceqq_u8(chunk, space));
    let whitespace = chunks.map(|chunk| {
      vorrq_u8(
        vceqq_u8(chunk, space),
        vcleq_u8(vsubq_u8(chunk, whitespace_start), whitespace_width),
      )
    });
    let newlines =
      chunks.map(|chunk| vorrq_u8(vceqq_u8(chunk, carriage_return), vceqq_u8(chunk, line_feed)));
    let apostrophes = chunks.map(|chunk| vceqq_u8(chunk, apostrophe));
    Some(AsciiMasks {
      letters: movemask64(letters),
      digits: movemask64(digits),
      spaces: movemask64(spaces),
      whitespace: movemask64(whitespace),
      newlines: movemask64(newlines),
      apostrophes: movemask64(apostrophes),
    })
  }
}

#[inline(always)]
unsafe fn movemask64(masks: [uint8x16_t; 4]) -> u64 {
  // The weighted pairwise-add tree is the simdjson-style movemask used by
  // GigaToken. Pinning `addp` avoids LLVM expanding each pairwise reduction
  // into a longer unzip/or sequence.
  unsafe {
    const WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let weights = vld1q_u8(WEIGHTS.as_ptr());
    let [mask0, mask1, mask2, mask3] = masks;
    let mut value0 = vandq_u8(mask0, weights);
    let value1 = vandq_u8(mask1, weights);
    let value2 = vandq_u8(mask2, weights);
    let value3 = vandq_u8(mask3, weights);
    core::arch::asm!(
      "addp {value0:v}.16b, {value0:v}.16b, {value1:v}.16b",
      "addp {value2:v}.16b, {value2:v}.16b, {value3:v}.16b",
      "addp {value0:v}.16b, {value0:v}.16b, {value2:v}.16b",
      "addp {value0:v}.16b, {value0:v}.16b, {value0:v}.16b",
      value0 = inout(vreg) value0,
      value1 = in(vreg) value1,
      value2 = inout(vreg) value2 => _,
      value3 = in(vreg) value3,
      options(pure, nomem, nostack, preserves_flags),
    );
    vgetq_lane_u64::<0>(vreinterpretq_u64_u8(value0))
  }
}
