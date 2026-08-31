//! Architecture-neutral state and boundary algebra for SIMD ASCII scanners.

use std::fmt;

pub(super) const BATCH_BYTES: usize = 64;
// Boundaries through byte 60 are independent of contractions that begin too
// close to the right edge. Later tokens fall back to the scalar scanner.
const TRUSTED_BITS: u64 = (1_u64 << 61) - 1;

#[derive(Clone, Copy)]
pub(super) struct AsciiMasks {
  pub(super) letters: u64,
  pub(super) digits: u64,
  pub(super) spaces: u64,
  pub(super) whitespace: u64,
  pub(super) apostrophes: u64,
}

#[derive(Clone, Copy)]
struct Classifier;

impl Classifier {
  fn detect() -> Option<Self> {
    #[cfg(target_arch = "aarch64")]
    {
      Some(Self)
    }
    #[cfg(target_arch = "x86_64")]
    {
      crate::avx2::is_available().then_some(Self)
    }
  }

  unsafe fn classify(self, pointer: *const u8) -> Option<AsciiMasks> {
    #[cfg(target_arch = "aarch64")]
    {
      // SAFETY: NEON is a baseline AArch64 feature and the caller supplies 64
      // readable bytes.
      unsafe { crate::neon::ascii_masks(pointer) }
    }
    #[cfg(target_arch = "x86_64")]
    {
      // SAFETY: `Classifier` can be constructed only after AVX2 detection and
      // the caller supplies 64 readable bytes.
      unsafe { crate::avx2::ascii_masks(pointer) }
    }
  }
}

#[derive(Clone)]
pub(super) struct BoundaryState {
  classifier: Option<Classifier>,
  base: usize,
  cursor: Option<usize>,
  starts: u64,
  blocked_until: usize,
}

impl fmt::Debug for BoundaryState {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    formatter
      .debug_struct("BoundaryState")
      .field("enabled", &self.is_enabled())
      .field("base", &self.base)
      .field("cursor", &self.cursor)
      .field("starts", &self.starts)
      .field("blocked_until", &self.blocked_until)
      .finish()
  }
}

impl BoundaryState {
  pub(super) fn for_text(bytes: &[u8], supported_pattern: bool) -> Self {
    let classifier =
      if supported_pattern && bytes.len() >= BATCH_BYTES && bytes[..BATCH_BYTES].is_ascii() {
        Classifier::detect()
      } else {
        None
      };
    Self {
      classifier,
      base: 0,
      cursor: None,
      starts: 0,
      blocked_until: 0,
    }
  }

  #[inline(always)]
  pub(super) fn is_enabled(&self) -> bool {
    self.classifier.is_some()
  }

  /// Return a cached token end, or defer this token to the scalar scanner.
  #[inline(never)]
  pub(super) fn next_end(&mut self, bytes: &[u8], start: usize) -> Option<usize> {
    let classifier = self.classifier?;
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
      // Do not pay for a full-vector ASCII eligibility probe on each block.
      self.blocked_until = start + BATCH_BYTES;
      return None;
    }

    let Some(starts) = token_starts(bytes, start, classifier) else {
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

fn token_starts(bytes: &[u8], start: usize, classifier: Classifier) -> Option<u64> {
  debug_assert!(bytes.len() - start >= BATCH_BYTES);
  // SAFETY: the caller provides 64 readable bytes. `Classifier` instances are
  // constructed only when their platform CPU requirements hold.
  let masks = unsafe { classifier.classify(bytes.as_ptr().add(start)) }?;
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

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn activation_is_conservative() {
    let ascii = [b'a'; BATCH_BYTES];
    let available = Classifier::detect().is_some();
    assert_eq!(
      BoundaryState::for_text(&ascii, true).is_enabled(),
      available
    );
    assert!(!BoundaryState::for_text(&ascii, false).is_enabled());
    assert!(!BoundaryState::for_text(b"short input", true).is_enabled());
    assert!(!BoundaryState::for_text("中文开头的输入不会启用 SIMD".as_bytes(), true).is_enabled());
  }

  #[test]
  fn masks_match_scalar_token_starts() {
    let Some(classifier) = Classifier::detect() else {
      return;
    };
    let text = concat!(
      "The  quick brown fox can't jump 42 times!\n",
      "'s'd'm't'll've're punctuation... words and 12345 tabs\ttoo. ",
      "Another line keeps every test window longer than sixty-four bytes."
    );
    let bytes = text.as_bytes();
    let mut batch_start = 0;
    while bytes.len() - batch_start >= BATCH_BYTES {
      let actual = token_starts(bytes, batch_start, classifier).unwrap() & TRUSTED_BITS;
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
  fn classifies_every_ascii_byte() {
    let Some(classifier) = Classifier::detect() else {
      return;
    };
    let bytes = (0_u8..=127).collect::<Vec<_>>();
    for batch_start in [0, BATCH_BYTES] {
      // SAFETY: each batch contains 64 readable bytes and `Classifier`
      // instances exist only when their platform requirements hold.
      let masks = unsafe { classifier.classify(bytes.as_ptr().add(batch_start)) }.unwrap();
      for offset in 0..BATCH_BYTES {
        let byte = bytes[batch_start + offset];
        let bit = 1_u64 << offset;
        assert_eq!(
          masks.letters & bit != 0,
          byte.is_ascii_alphabetic(),
          "letters byte={byte}"
        );
        assert_eq!(
          masks.digits & bit != 0,
          byte.is_ascii_digit(),
          "digits byte={byte}"
        );
        assert_eq!(masks.spaces & bit != 0, byte == b' ', "spaces byte={byte}");
        assert_eq!(
          masks.whitespace & bit != 0,
          byte == b' ' || (9..=13).contains(&byte),
          "whitespace byte={byte}"
        );
        assert_eq!(
          masks.apostrophes & bit != 0,
          byte == b'\'',
          "apostrophes byte={byte}"
        );
      }
    }
  }

  #[test]
  fn classifier_rejects_non_ascii_windows() {
    let Some(classifier) = Classifier::detect() else {
      return;
    };
    let mut bytes = [b'a'; BATCH_BYTES];
    for offset in 0..BATCH_BYTES {
      bytes[offset] = 0x80;
      // SAFETY: the array provides 64 readable bytes and `Classifier`
      // instances exist only when their platform requirements hold.
      assert!(unsafe { classifier.classify(bytes.as_ptr()) }.is_none());
      bytes[offset] = b'a';
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
    if !state.is_enabled() {
      return;
    }
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
