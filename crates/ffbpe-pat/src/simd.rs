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
  pub(super) newlines: u64,
  pub(super) apostrophes: u64,
}

/// PAT families whose ASCII boundaries can be derived from the shared masks.
#[derive(Clone, Copy, Debug)]
pub(super) enum SimdScheme {
  Gpt2,
  Cl100k,
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
  scheme: Option<SimdScheme>,
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
      .field("scheme", &self.scheme)
      .field("base", &self.base)
      .field("cursor", &self.cursor)
      .field("starts", &self.starts)
      .field("blocked_until", &self.blocked_until)
      .finish()
  }
}

impl BoundaryState {
  pub(super) fn for_text(bytes: &[u8], scheme: Option<SimdScheme>) -> Self {
    let classifier =
      if scheme.is_some() && bytes.len() >= BATCH_BYTES && bytes[..BATCH_BYTES].is_ascii() {
        Classifier::detect()
      } else {
        None
      };
    Self {
      classifier,
      scheme,
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
    let scheme = self.scheme?;
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

    let Some(starts) = token_starts(bytes, start, classifier, scheme) else {
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

fn token_starts(
  bytes: &[u8],
  start: usize,
  classifier: Classifier,
  scheme: SimdScheme,
) -> Option<u64> {
  debug_assert!(bytes.len() - start >= BATCH_BYTES);
  // SAFETY: the caller provides 64 readable bytes. `Classifier` instances are
  // constructed only when their platform CPU requirements hold.
  let window = &bytes[start..start + BATCH_BYTES];
  let masks = unsafe { classifier.classify(window.as_ptr()) }?;
  Some(match scheme {
    SimdScheme::Gpt2 => gpt2_token_starts(window, masks),
    SimdScheme::Cl100k => cl100k_token_starts(window, masks),
  })
}

fn gpt2_token_starts(bytes: &[u8], masks: AsciiMasks) -> u64 {
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
    let suffix_len = match bytes[offset + 1] {
      b's' | b'd' | b'm' | b't' => 2,
      b'l' if bytes[offset + 2] == b'l' => 3,
      b'v' | b'r' if bytes[offset + 2] == b'e' => 3,
      _ => 0,
    };
    if suffix_len != 0 {
      starts &= !(1_u64 << (offset + 1));
      starts |= 1_u64 << (offset + suffix_len);
    }
  }
  starts
}

/// Derive cl100k boundaries from one all-ASCII window.
///
/// This is the pure-ASCII part of GigaToken's cl100k mask algebra, reduced to
/// a token-aligned cache. It omits cross-window carries because the iterator
/// calls it only at known token boundaries. A token that reaches the right
/// edge has no cached end, so the scalar scanner retains all edge lookahead.
#[inline(always)]
fn cl100k_token_starts(bytes: &[u8], masks: AsciiMasks) -> u64 {
  let letters = masks.letters;
  let digits = masks.digits;
  let spaces = masks.spaces;
  let linebreaks = masks.newlines;
  let tab_whitespace = masks.whitespace & !spaces & !linebreaks;
  let other = !(masks.letters | masks.digits | masks.whitespace);

  // `[^\r\n\p{L}\p{N}]?+\p{L}++`: a letter continues a letter run,
  // follows literal space/tab-class whitespace, or is absorbed by a single
  // punctuation prefix whose own predecessor is neither punctuation nor
  // literal space.
  let previous_letters = letters << 1;
  let previous_spaces = spaces << 1;
  let previous_tab_whitespace = tab_whitespace << 1;
  let previous_other = other << 1;
  let two_back_space_or_other = (spaces | other) << 2;
  let letter_starts = letters
    & !previous_letters
    & !previous_spaces
    & !previous_tab_whitespace
    & !(previous_other & !two_back_space_or_other);

  // `\p{N}{1,3}+`: run starts re-arm every three digits.
  let digit_starts = cl100k_digit_starts(digits);

  // ` ?[^\s\p{L}\p{N}]++[\r\n]*+`: punctuation continues through an
  // adjacent punctuation run, and only literal space can prefix it.
  let punctuation_starts = other & !(other << 1) & !(spaces << 1);

  // Newlines directly after punctuation belong to the punctuation token.
  let absorbed_newlines = smear_up(linebreaks & (other << 1), linebreaks);
  let whitespace = masks.whitespace & !absorbed_newlines;
  let mut whitespace_starts = whitespace & (!(whitespace << 1) | (!whitespace >> 1));

  // A whitespace run containing a newline ends at its last newline. Its
  // remaining newline-free tail follows the normal "split before the last
  // whitespace byte" rule when it ends within this window.
  let mut newline_runs = linebreaks & whitespace;
  while newline_runs != 0 {
    let first_newline = newline_runs.trailing_zeros() as usize;
    let start = whitespace_run_start(whitespace, first_newline);
    let end = run_end(whitespace, start);
    let run = bit_range(start, end);
    whitespace_starts &= !run;
    whitespace_starts |= 1_u64 << start;

    let last_newline = 63 - (linebreaks & run).leading_zeros() as usize;
    let tail_start = last_newline + 1;
    if tail_start < end {
      whitespace_starts |= 1_u64 << tail_start;
      if end < BATCH_BYTES && end - tail_start > 1 {
        whitespace_starts |= 1_u64 << (end - 1);
      }
    }
    newline_runs &= !run;
  }

  let mut starts = letter_starts | digit_starts | punctuation_starts | whitespace_starts | 1;
  let mut contractions = masks.apostrophes & starts & TRUSTED_BITS;
  while contractions != 0 {
    let offset = contractions.trailing_zeros() as usize;
    contractions &= contractions - 1;
    if let Some(length) = cl100k_contraction_len(bytes, offset) {
      starts &= !(1_u64 << (offset + 1));
      starts |= 1_u64 << (offset + length);
    }
  }
  starts
}

#[inline(always)]
fn cl100k_contraction_len(bytes: &[u8], offset: usize) -> Option<usize> {
  let first = bytes.get(offset + 1)?.to_ascii_lowercase();
  match first {
    b's' | b'd' | b'm' | b't' => Some(2),
    b'l' if ascii_case_eq(bytes.get(offset + 2), b'l') => Some(3),
    b'v' if ascii_case_eq(bytes.get(offset + 2), b'e') => Some(3),
    b'r' if ascii_case_eq(bytes.get(offset + 2), b'e') => Some(3),
    _ => None,
  }
}

#[inline(always)]
fn ascii_case_eq(actual: Option<&u8>, expected: u8) -> bool {
  actual.is_some_and(|byte| byte.to_ascii_lowercase() == expected)
}

#[inline(always)]
fn cl100k_digit_starts(digits: u64) -> u64 {
  let mut starts = digits & !(digits << 1);
  let mut continues_for_three = digits & (digits >> 1) & (digits >> 2) & (digits >> 3);
  let mut shift = 3;
  while shift < BATCH_BYTES as u32 {
    starts |= (starts & continues_for_three) << shift;
    continues_for_three &= continues_for_three >> shift;
    shift <<= 1;
  }
  starts
}

#[inline(always)]
fn smear_up(seed: u64, within: u64) -> u64 {
  let mut included = seed;
  let mut remaining = within;
  let mut shift = 1;
  while shift < BATCH_BYTES as u32 {
    included |= (included << shift) & remaining;
    remaining &= remaining << shift;
    shift <<= 1;
  }
  included
}

#[inline(always)]
fn whitespace_run_start(whitespace: u64, offset: usize) -> usize {
  let preceding = !whitespace & bit_range(0, offset);
  if preceding == 0 {
    0
  } else {
    64 - preceding.leading_zeros() as usize
  }
}

#[inline(always)]
fn bit_range(start: usize, end: usize) -> u64 {
  debug_assert!(start <= end && end <= BATCH_BYTES);
  let before_end = if end == BATCH_BYTES {
    u64::MAX
  } else {
    (1_u64 << end) - 1
  };
  let before_start = if start == 0 {
    0
  } else if start == BATCH_BYTES {
    u64::MAX
  } else {
    (1_u64 << start) - 1
  };
  before_end & !before_start
}

#[inline(always)]
fn run_end(mask: u64, offset: usize) -> usize {
  debug_assert!(mask & (1_u64 << offset) != 0);
  offset + (mask >> offset).trailing_ones() as usize
}

#[cfg(test)]
mod tests {
  use super::*;

  fn assert_mask_starts_match_scalar(
    text: &str,
    scheme: SimdScheme,
    pretoken_end: fn(&str, usize) -> usize,
  ) {
    let Some(classifier) = Classifier::detect() else {
      return;
    };
    let bytes = text.as_bytes();
    let mut batch_start = 0;
    while bytes.len() - batch_start >= BATCH_BYTES {
      let actual = token_starts(bytes, batch_start, classifier, scheme).unwrap() & TRUSTED_BITS;
      let mut expected = 1_u64;
      let mut position = batch_start;
      loop {
        position = pretoken_end(text, position);
        let offset = position - batch_start;
        if offset >= 61 {
          break;
        }
        expected |= 1_u64 << offset;
      }
      assert_eq!(
        actual, expected,
        "batch_start={batch_start}, scheme={scheme:?}"
      );
      batch_start = pretoken_end(text, batch_start);
    }
  }

  fn assert_boundary_state_matches_scalar(
    text: &str,
    scheme: SimdScheme,
    pretoken_end: fn(&str, usize) -> usize,
  ) -> usize {
    let mut state = BoundaryState::for_text(text.as_bytes(), Some(scheme));
    if !state.is_enabled() {
      return 0;
    }
    let mut position = 0;
    let mut cached = 0;
    while position < text.len() {
      let expected = pretoken_end(text, position);
      let actual = state.next_end(text.as_bytes(), position);
      cached += usize::from(actual.is_some());
      assert_eq!(
        actual.unwrap_or(expected),
        expected,
        "position={position}, scheme={scheme:?}"
      );
      position = expected;
    }
    cached
  }

  fn random_ascii_text(len: usize) -> String {
    const ALPHABET: &[u8] =
      b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '!?_-/\t\n\r\x0b\x0c";
    let mut bytes = Vec::with_capacity(len);
    let mut random = 0x9e37_79b9_7f4a_7c15_u64;
    for _ in 0..len {
      random ^= random >> 12;
      random ^= random << 25;
      random ^= random >> 27;
      bytes.push(ALPHABET[(random as usize) % ALPHABET.len()]);
    }
    String::from_utf8(bytes).unwrap()
  }

  #[test]
  fn activation_is_conservative() {
    let ascii = [b'a'; BATCH_BYTES];
    let available = Classifier::detect().is_some();
    assert_eq!(
      BoundaryState::for_text(&ascii, Some(SimdScheme::Gpt2)).is_enabled(),
      available
    );
    assert_eq!(
      BoundaryState::for_text(&ascii, Some(SimdScheme::Cl100k)).is_enabled(),
      available
    );
    assert!(!BoundaryState::for_text(&ascii, None).is_enabled());
    assert!(!BoundaryState::for_text(b"short input", Some(SimdScheme::Gpt2)).is_enabled());
    assert!(
      !BoundaryState::for_text(
        "中文开头的输入不会启用 SIMD".as_bytes(),
        Some(SimdScheme::Cl100k),
      )
      .is_enabled()
    );
  }

  #[test]
  fn gpt2_masks_match_scalar_token_starts() {
    let text = concat!(
      "The  quick brown fox can't jump 42 times!\n",
      "'s'd'm't'll've're punctuation... words and 12345 tabs\ttoo. ",
      "Another line keeps every test window longer than sixty-four bytes."
    );
    assert_mask_starts_match_scalar(text, SimdScheme::Gpt2, crate::gpt2::pretoken_end);
  }

  #[test]
  fn cl100k_masks_match_scalar_token_starts_at_every_edge() {
    const CASE: &str = concat!(
      "!!word !word \tword \rword \nword 1 12 123 1234567890 ",
      "'s 'S 'd 'D 'm 'M 't 'T 'll 'LL 've 'Ve 're 'RE 'lx 'vX 'rX ",
      "!?\r\nnext ...\n\rnext \t\x0b\x0c  word  \tword \r\n word ",
      "tail \n \t"
    );
    for lead in 0..64 {
      let text = format!(
        "{}{}{}",
        "a\n".repeat(lead),
        CASE,
        " later words 1234?!\r\n".repeat(4)
      );
      assert_mask_starts_match_scalar(&text, SimdScheme::Cl100k, crate::cl100k::pretoken_end);
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
          masks.newlines & bit != 0,
          matches!(byte, b'\r' | b'\n'),
          "newlines byte={byte}"
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
    let text = random_ascii_text(32_768);
    let cached =
      assert_boundary_state_matches_scalar(&text, SimdScheme::Gpt2, crate::gpt2::pretoken_end);
    assert!(cached > 1_000, "boundary cache did not engage often enough");
  }

  #[test]
  fn cl100k_boundary_state_matches_scalar_on_ascii_and_unicode_streams() {
    let mut text = random_ascii_text(32_768);
    text.push_str("中文后继续 ASCII words 123456 !word\r\n");
    text.push_str(&random_ascii_text(4_096));
    let cached =
      assert_boundary_state_matches_scalar(&text, SimdScheme::Cl100k, crate::cl100k::pretoken_end);
    assert!(cached > 1_000, "boundary cache did not engage often enough");
  }

  #[test]
  fn cl100k_boundary_state_matches_scalar_for_every_ascii_transition() {
    let bytes = (0_u8..=127).cycle().take(8_192).collect::<Vec<_>>();
    let text = String::from_utf8(bytes).unwrap();
    let cached =
      assert_boundary_state_matches_scalar(&text, SimdScheme::Cl100k, crate::cl100k::pretoken_end);
    assert!(cached > 100, "boundary cache did not engage often enough");
  }
}
