//! Zero-copy scanners for common tokenizer PAT expressions.
//!
//! This crate specializes a fixed set of well-known pretokenizer patterns. It
//! does not compile arbitrary regular expressions; callers should retain their
//! regex fallback for patterns that [`Pattern::recognize`] does not identify.

mod ascii;
mod cl100k;
mod common;
mod gpt2;
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod neon;
mod o200k;

use std::{iter::FusedIterator, ops::Range};

/// Original GPT-2 pretokenizer pattern.
pub const GPT2_PATTERN: &str = gpt2::PATTERN;
/// Legacy spelling of the GPT-2 pattern with equivalent matching semantics.
pub const GPT2_LEGACY_PATTERN: &str = gpt2::LEGACY_PATTERN;
/// Canonical tiktoken GPT-2/r50k/p50k pretokenizer pattern.
pub const R50K_PATTERN: &str = gpt2::R50K_PATTERN;
/// Canonical tiktoken cl100k pretokenizer pattern.
pub const CL100K_PATTERN: &str = cl100k::PATTERN;
/// Canonical tiktoken o200k pretokenizer pattern.
pub const O200K_PATTERN: &str = o200k::PATTERN;

/// A supported tokenizer pretokenization pattern family.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum Pattern {
  Gpt2,
  R50k,
  Cl100k,
  O200k,
}

impl Pattern {
  /// All currently supported canonical pattern families.
  pub const ALL: [Self; 4] = [Self::Gpt2, Self::R50k, Self::Cl100k, Self::O200k];

  /// Identify an exact known PAT expression.
  pub fn recognize(regex: &str) -> Option<Self> {
    match regex {
      gpt2::PATTERN | gpt2::LEGACY_PATTERN => Some(Self::Gpt2),
      gpt2::R50K_PATTERN => Some(Self::R50k),
      cl100k::PATTERN => Some(Self::Cl100k),
      o200k::PATTERN => Some(Self::O200k),
      _ => None,
    }
  }

  /// Return the canonical regex spelling for this pattern family.
  pub const fn regex(self) -> &'static str {
    match self {
      Self::Gpt2 => GPT2_PATTERN,
      Self::R50k => R50K_PATTERN,
      Self::Cl100k => CL100K_PATTERN,
      Self::O200k => O200K_PATTERN,
    }
  }

  /// Iterate over the byte ranges of pretokens in `text`.
  pub fn offsets(self, text: &str) -> Offsets<'_> {
    Offsets {
      pattern: self,
      text,
      start: 0,
      #[cfg(all(feature = "simd", target_arch = "aarch64"))]
      neon: neon::BoundaryState::for_text(
        text.as_bytes(),
        matches!(self, Pattern::Gpt2 | Pattern::R50k),
      ),
    }
  }

  /// Iterate over borrowed pretokens in `text`.
  pub fn split(self, text: &str) -> Split<'_> {
    Split {
      text,
      offsets: self.offsets(text),
    }
  }

  fn pretoken_end(self, text: &str, start: usize) -> usize {
    match self {
      Self::Gpt2 | Self::R50k => gpt2::pretoken_end(text, start),
      Self::Cl100k => cl100k::pretoken_end(text, start),
      Self::O200k => o200k::pretoken_end(text, start),
    }
  }
}

/// Byte ranges produced by a known [`Pattern`].
#[derive(Clone, Debug)]
pub struct Offsets<'a> {
  pattern: Pattern,
  text: &'a str,
  start: usize,
  #[cfg(all(feature = "simd", target_arch = "aarch64"))]
  neon: neon::BoundaryState,
}

impl Iterator for Offsets<'_> {
  type Item = Range<usize>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.start == self.text.len() {
      return None;
    }
    let start = self.start;
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    let end = if self.neon.is_enabled() {
      self
        .neon
        .next_end(self.text.as_bytes(), start)
        .unwrap_or_else(|| self.pattern.pretoken_end(self.text, start))
    } else {
      self.pattern.pretoken_end(self.text, start)
    };
    #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
    let end = self.pattern.pretoken_end(self.text, start);
    debug_assert!(end > start);
    debug_assert!(self.text.is_char_boundary(end));
    self.start = end;
    Some(start..end)
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let remaining = self.text.len() - self.start;
    (usize::from(remaining > 0), Some(remaining))
  }
}

impl FusedIterator for Offsets<'_> {}

/// Borrowed pretokens produced by a known [`Pattern`].
#[derive(Clone, Debug)]
pub struct Split<'a> {
  text: &'a str,
  offsets: Offsets<'a>,
}

impl<'a> Iterator for Split<'a> {
  type Item = &'a str;

  fn next(&mut self) -> Option<Self::Item> {
    self.offsets.next().map(|range| &self.text[range])
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    self.offsets.size_hint()
  }
}

impl FusedIterator for Split<'_> {}

#[cfg(test)]
mod tests {
  use fancy_regex::Regex;

  use super::*;

  fn assert_regex_parity(pattern: Pattern, text: &str) {
    assert_regex_string_parity(pattern, pattern.regex(), text);
  }

  fn assert_regex_string_parity(pattern: Pattern, regex: &str, text: &str) {
    let regex = Regex::new(regex).unwrap();
    assert_compiled_regex_parity(pattern, &regex, text);
  }

  fn assert_compiled_regex_parity(pattern: Pattern, regex: &Regex, text: &str) {
    let expected = regex
      .find_iter(text)
      .map(|found| found.unwrap().as_str())
      .collect::<Vec<_>>();
    let actual = pattern.split(text).collect::<Vec<_>>();
    assert_eq!(actual, expected, "pattern={pattern:?}, text={text:?}");

    let offsets = pattern.offsets(text).collect::<Vec<_>>();
    assert_eq!(
      offsets
        .iter()
        .map(|range| &text[range.clone()])
        .collect::<Vec<_>>(),
      actual,
    );
  }

  #[test]
  fn recognizes_only_exact_supported_patterns() {
    for pattern in Pattern::ALL {
      assert_eq!(Pattern::recognize(pattern.regex()), Some(pattern));
    }
    assert_eq!(Pattern::recognize(GPT2_LEGACY_PATTERN), Some(Pattern::Gpt2),);
    assert_eq!(Pattern::recognize(r"\p{L}+"), None);
  }

  #[test]
  fn matches_regex_on_edge_cases() {
    for text in [
      "",
      "Hello, world! It's 2024.",
      "'s'd'm't'll've're",
      "'S'LL'Ve'RE'ſ",
      "a  b   c    ",
      "a\t b\r\nc\u{A0}\u{2003}d",
      "你好，世界！Now是2024年。",
      "한글かなカナ mixed العربية १२३",
      "e\u{301} café 👩‍💻🏳️‍🌈",
      "lower UPPER TitleCase HTTPServer ABC中文def",
      "a1 12 123 1234 １２３４",
      "a\r\nb\n\nc\r\r\nd trailing \t  ",
      "²¼ⅠⅫ⑴",
      "<|endoftext|>before<|endoftext|>after",
      " punctuation...?!—–_+=/\\\"'s ",
      " \n \t",
      "a\r\n \u{2003}",
      "A中文B A中文Bc a中文B \u{301}ABC \u{301}ABCdef",
      "\u{301}\u{301}A \u{301}A\u{301}B \u{301}A\u{301}b",
      "x'ſtail 'ſtail 'Lſ 'VÉ 'REmainder",
      "...\r\n//word",
    ] {
      for pattern in Pattern::ALL {
        assert_regex_parity(pattern, text);
      }
      assert_regex_string_parity(Pattern::Gpt2, GPT2_LEGACY_PATTERN, text);
    }
  }

  #[test]
  fn matches_regex_on_short_class_combinations() {
    // Exercise branch priority and optional-prefix backtracking, especially
    // marks (both word and punctuation members) and trailing whitespace.
    const ALPHABET: &[char] = &[
      'A', 'a', '中', '\u{301}', '1', ' ', '\t', '\n', '\r', '\u{2003}', '\'', '/',
    ];
    for pattern in Pattern::ALL {
      let regex = Regex::new(pattern.regex()).unwrap();
      for length in 1..=4 {
        for mut value in 0..ALPHABET.len().pow(length) {
          let mut text = String::new();
          for _ in 0..length {
            text.push(ALPHABET[value % ALPHABET.len()]);
            value /= ALPHABET.len();
          }
          assert_compiled_regex_parity(pattern, &regex, &text);
        }
      }
    }
  }

  #[test]
  fn matches_regex_at_ascii_run_boundaries() {
    for pattern in Pattern::ALL {
      let regex = Regex::new(pattern.regex()).unwrap();
      for length in 0..=33 {
        for word in ["a", "Z", "aZ"] {
          for suffix in ["!", "é", "中", "\u{301}", "👩", "\n ", "'ſtail"] {
            let text = format!(" {}{suffix}ASCII", word.repeat(length));
            for end in text
              .char_indices()
              .map(|(index, _)| index)
              .chain([text.len()])
            {
              assert_compiled_regex_parity(pattern, &regex, &text[..end]);
            }
          }
        }
      }
    }
  }

  #[test]
  fn matches_regex_at_dense_unicode_threshold() {
    for pattern in Pattern::ALL {
      let regex = Regex::new(pattern.regex()).unwrap();
      for scalar in ["é", "中", "𐐷"] {
        for length in 1..=8 {
          for suffix in ["!", "0", " ASCII", "\u{301}", "👩"] {
            let text = format!(" {}{suffix}", scalar.repeat(length));
            assert_compiled_regex_parity(pattern, &regex, &text);
          }
        }
      }
    }
  }

  #[test]
  fn matches_regex_on_deterministic_unicode_mix() {
    const ALPHABET: &[char] = &[
      'a', 'Z', '0', '9', '\'', ' ', '\t', '\n', '\r', ',', '—', '你', '界', '한', '글', 'か',
      'ナ', 'é', '\u{301}', '\u{A0}', '\u{2003}', '²', 'Ⅻ', '👩', '\u{200D}', '💻',
    ];
    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    let mut text = String::new();
    for _ in 0..20_000 {
      state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
      text.push(ALPHABET[(state as usize) % ALPHABET.len()]);
    }
    for pattern in Pattern::ALL {
      assert_regex_parity(pattern, &text);
    }
  }

  #[test]
  fn matches_regex_on_random_unicode_scalars() {
    let mut state = 0xa076_1d64_78bd_642f_u64;
    let mut text = String::new();
    let mut count = 0;
    while count < 50_000 {
      state ^= state >> 12;
      state ^= state << 25;
      state ^= state >> 27;
      let value = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) % 0x11_0000) as u32;
      if let Some(ch) = char::from_u32(value) {
        text.push(ch);
        count += 1;
      }
    }
    for pattern in Pattern::ALL {
      assert_regex_parity(pattern, &text);
    }
  }
}
