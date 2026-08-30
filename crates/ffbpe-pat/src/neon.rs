//! Optional AArch64 NEON boundary masks for ASCII GPT-2/r50k windows.

use std::arch::aarch64::*;

const BATCH_BYTES: usize = 64;
// Boundaries through byte 60 are independent of contractions that begin too
// close to the right edge. Later tokens fall back to the scalar scanner.
const TRUSTED_BITS: u64 = (1_u64 << 61) - 1;

#[derive(Clone, Debug)]
pub(super) struct BoundaryState {
  enabled: bool,
  base: usize,
  cursor: Option<usize>,
  starts: u64,
  blocked_until: usize,
}

impl BoundaryState {
  pub(super) fn for_text(bytes: &[u8], supported_pattern: bool) -> Self {
    Self {
      enabled: supported_pattern && bytes.len() >= BATCH_BYTES && bytes[..BATCH_BYTES].is_ascii(),
      base: 0,
      cursor: None,
      starts: 0,
      blocked_until: 0,
    }
  }

  #[inline(always)]
  pub(super) fn is_enabled(&self) -> bool {
    self.enabled
  }

  /// Return a cached token end, or defer this token to the scalar scanner.
  #[inline(never)]
  pub(super) fn next_end(&mut self, bytes: &[u8], start: usize) -> Option<usize> {
    if self.cursor == Some(start) {
      if let Some(end) = self.pop_start() {
        return Some(end);
      }
      self.blocked_until = self.base.saturating_add(BATCH_BYTES);
    }

    self.cursor = None;
    self.starts = 0;
    if start < self.blocked_until || bytes.len() - start < BATCH_BYTES {
      return None;
    }
    if !bytes[start].is_ascii() {
      // Dense Unicode runs commonly advance by only a few bytes per token.
      // Do not pay for a four-vector ASCII eligibility probe on each block.
      self.blocked_until = start + BATCH_BYTES;
      return None;
    }

    let Some(starts) = token_starts(bytes, start) else {
      // Avoid reclassifying the same Unicode-containing window for every
      // scalar token within it.
      self.blocked_until = start + BATCH_BYTES;
      return None;
    };
    let starts = starts & TRUSTED_BITS & !1;
    if starts == 0 {
      self.blocked_until = start + BATCH_BYTES;
      return None;
    }

    self.base = start;
    self.cursor = Some(start);
    self.starts = starts;
    self.pop_start()
  }

  #[inline]
  fn pop_start(&mut self) -> Option<usize> {
    if self.starts == 0 {
      return None;
    }
    let offset = self.starts.trailing_zeros() as usize;
    self.starts &= self.starts - 1;
    let end = self.base + offset;
    self.cursor = Some(end);
    Some(end)
  }
}

#[derive(Clone, Copy)]
struct AsciiMasks {
  letters: u64,
  digits: u64,
  spaces: u64,
  whitespace: u64,
  apostrophes: u64,
}

#[inline(never)]
fn token_starts(bytes: &[u8], start: usize) -> Option<u64> {
  let masks = ascii_masks(bytes, start)?;
  let other = !(masks.letters | masks.digits | masks.whitespace);
  let continues = (masks.letters & (masks.letters << 1))
    | (masks.digits & (masks.digits << 1))
    | (other & (other << 1));
  let after_space = masks.spaces << 1;
  let non_whitespace = !masks.whitespace & !continues & !after_space;

  // A whitespace run starts a token at its first byte. When followed by
  // content, its last byte starts the optional-space-prefixed next token.
  let followed_by_content = masks.whitespace & (!masks.whitespace >> 1);
  let previous_whitespace = masks.whitespace << 1;
  let whitespace = masks.whitespace & (!previous_whitespace | followed_by_content);
  let mut starts = non_whitespace | whitespace | 1;

  // The contraction alternative precedes ordinary punctuation/letter runs.
  // Move the boundary after a token-start apostrophe over its suffix.
  let mut candidates = masks.apostrophes & starts & TRUSTED_BITS;
  while candidates != 0 {
    let offset = candidates.trailing_zeros() as usize;
    candidates &= candidates - 1;
    let suffix_len = match bytes[start + offset + 1] {
      b's' | b'd' | b'm' | b't' => 2,
      b'l' if bytes[start + offset + 2] == b'l' => 3,
      b'v' | b'r' if bytes[start + offset + 2] == b'e' => 3,
      _ => 0,
    };
    if suffix_len != 0 {
      starts &= !(1_u64 << (offset + 1));
      starts |= 1_u64 << (offset + suffix_len);
    }
  }
  Some(starts)
}

fn ascii_masks(bytes: &[u8], start: usize) -> Option<AsciiMasks> {
  debug_assert!(bytes.len() - start >= BATCH_BYTES);
  // SAFETY: the length check above makes each 16-byte load in bounds. NEON is
  // a baseline target feature for Rust's AArch64 targets.
  unsafe {
    let pointer = bytes.as_ptr().add(start);
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
    let apostrophes = chunks.map(|chunk| vceqq_u8(chunk, apostrophe));
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

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn activation_is_conservative() {
    let ascii = [b'a'; BATCH_BYTES];
    assert!(BoundaryState::for_text(&ascii, true).is_enabled());
    assert!(!BoundaryState::for_text(&ascii, false).is_enabled());
    assert!(!BoundaryState::for_text(b"short input", true).is_enabled());
    assert!(!BoundaryState::for_text("中文开头的输入不会启用 NEON".as_bytes(), true).is_enabled());
  }

  #[test]
  fn masks_match_scalar_token_starts() {
    let text = concat!(
      "The  quick brown fox can't jump 42 times!\n",
      "'s'd'm't'll've're punctuation... words and 12345 tabs\ttoo. ",
      "Another line keeps every test window longer than sixty-four bytes."
    );
    let bytes = text.as_bytes();
    let mut batch_start = 0;
    while bytes.len() - batch_start >= BATCH_BYTES {
      let actual = token_starts(bytes, batch_start).unwrap() & TRUSTED_BITS;
      let mut expected = 1_u64;
      let mut position = batch_start;
      loop {
        position = crate::gpt2::pretoken_end(text, position);
        let offset = position - batch_start;
        if offset >= 61 {
          break;
        }
        expected |= 1_u64 << offset;
      }
      assert_eq!(actual, expected, "batch_start={batch_start}");
      batch_start = crate::gpt2::pretoken_end(text, batch_start);
    }
  }

  #[test]
  fn boundary_state_matches_scalar_on_ascii_stream() {
    const ALPHABET: &[u8] = b"aaaZZ019 '!?_-/\t\n\r";
    let mut bytes = Vec::with_capacity(32_768);
    let mut random = 0x9e37_79b9_7f4a_7c15_u64;
    for _ in 0..bytes.capacity() {
      random ^= random >> 12;
      random ^= random << 25;
      random ^= random >> 27;
      bytes.push(ALPHABET[(random as usize) % ALPHABET.len()]);
    }
    let text = String::from_utf8(bytes).unwrap();
    let mut state = BoundaryState::for_text(text.as_bytes(), true);
    let mut position = 0;
    let mut cached = 0;
    while position < text.len() {
      let expected = crate::gpt2::pretoken_end(&text, position);
      let actual = state.next_end(text.as_bytes(), position);
      cached += usize::from(actual.is_some());
      assert_eq!(actual.unwrap_or(expected), expected, "position={position}");
      position = expected;
    }
    assert!(cached > 1_000, "boundary cache did not engage often enough");
  }
}
