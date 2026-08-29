use super::{
  ascii,
  common::{
    CaseClass, case_class, case_insensitive_contraction_end, char_at, is_letter, is_number,
    is_other, is_whitespace, next_boundary, scan_limited_numbers, scan_while,
    scan_whitespace_with_newlines,
  },
};

// The case-state scan is adapted from GigaToken's o200k_family.rs.
// See LICENSE-GIGATOKEN and README.md for attribution.

struct WordEnd {
  end: usize,
  branch_one: bool,
}

pub(super) const PATTERN: &str = concat!(
  r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
  "|",
  r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
  "|",
  r"\p{N}{1,3}",
  "|",
  r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
  "|",
  r"\s*[\r\n]+",
  "|",
  r"\s+(?!\S)",
  "|",
  r"\s+",
);

pub(super) fn pretoken_end(text: &str, start: usize) -> usize {
  if let Some(end) = word_end(text, start) {
    return end;
  }

  let first = char_at(text, start);
  if is_number(first) {
    return scan_limited_numbers(text, start);
  }
  if let Some(end) = punctuation_end(text, start) {
    return end;
  }
  if is_whitespace(first) {
    return scan_whitespace_with_newlines(text, start, false);
  }
  next_boundary(text, start)
}

fn word_end(text: &str, start: usize) -> Option<usize> {
  let first = char_at(text, start);
  if !matches!(first, '\r' | '\n') && !is_letter(first) && !is_number(first) {
    let word_start = start + first.len_utf8();
    // Marks can be the optional prefix OR a word member. Branch one without
    // a prefix wins over branch two with one: "\u{301}ABC" -> mark | ABC.
    let prefix_is_mark = !first.is_ascii() && case_class(first) == CaseClass::Shared;
    if let Some(word) = scan_case_word(text, word_start)
      && (word.branch_one || !prefix_is_mark)
    {
      return Some(with_optional_contraction(text, word.end));
    }
    return prefix_is_mark.then(|| with_optional_contraction(text, word_start));
  }
  scan_case_word(text, start).map(|word| with_optional_contraction(text, word.end))
}

fn scan_case_word(text: &str, start: usize) -> Option<WordEnd> {
  let bytes = text.as_bytes();
  let mut pos = start;
  let mut lower_phase = false;
  let mut last_shared_end = None;
  while pos < bytes.len() {
    let byte = bytes[pos];
    if byte.is_ascii_lowercase() {
      lower_phase = true;
      pos = ascii::scan_lowercase(bytes, pos + 1);
    } else if byte.is_ascii_uppercase() {
      if lower_phase {
        break;
      }
      pos = ascii::scan_uppercase(bytes, pos + 1);
    } else if byte.is_ascii() {
      break;
    } else {
      let ch = char_at(text, pos);
      match case_class(ch) {
        CaseClass::Upper if lower_phase => break,
        CaseClass::Upper => {}
        CaseClass::Lower => lower_phase = true,
        CaseClass::Shared => {
          if !lower_phase {
            last_shared_end = Some(pos + ch.len_utf8());
          }
        }
        CaseClass::Other => break,
      }
      pos += ch.len_utf8();
    }
  }
  if pos == start {
    return None;
  }
  // Before a strict lowercase letter, branch one can give its final shared
  // scalar back to the required lower/shared group. Remember that boundary
  // during the forward scan instead of searching backwards and rescanning.
  Some(WordEnd {
    end: if lower_phase {
      pos
    } else {
      last_shared_end.unwrap_or(pos)
    },
    branch_one: lower_phase || last_shared_end.is_some(),
  })
}

fn with_optional_contraction(text: &str, end: usize) -> usize {
  if end < text.len() {
    case_insensitive_contraction_end(text, end).unwrap_or(end)
  } else {
    end
  }
}

fn punctuation_end(text: &str, start: usize) -> Option<usize> {
  let word_start = if text.as_bytes()[start] == b' ' {
    start + 1
  } else {
    start
  };
  if word_start >= text.len() || !is_other(char_at(text, word_start)) {
    return None;
  }
  let punctuation_end = scan_while(text, word_start, is_other);
  Some(scan_while(text, punctuation_end, |ch| {
    matches!(ch, '\r' | '\n' | '/')
  }))
}
