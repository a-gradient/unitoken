use super::common::{
  case_insensitive_contraction_end, char_at, is_letter, is_number,
  is_o200k_lower_or_shared, is_o200k_upper_or_shared, is_other, is_whitespace,
  next_boundary, scan_through_last_newline, scan_whitespace, scan_while,
};

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
  if let Some(end) = word_branch_one_end(text, start) {
    return end;
  }
  if let Some(end) = word_branch_two_end(text, start) {
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
    if let Some(end) = scan_through_last_newline(text, start) {
      return end;
    }
    return scan_whitespace(text, start);
  }
  next_boundary(text, start)
}

fn word_branch_one_end(text: &str, start: usize) -> Option<usize> {
  match_with_optional_prefix(text, start, word_branch_one_core)
}

fn word_branch_two_end(text: &str, start: usize) -> Option<usize> {
  match_with_optional_prefix(text, start, word_branch_two_core)
}

fn match_with_optional_prefix(
  text: &str,
  start: usize,
  matcher: fn(&str, usize) -> Option<usize>,
) -> Option<usize> {
  let first = char_at(text, start);
  if !matches!(first, '\r' | '\n') && !is_letter(first) && !is_number(first) {
    let word_start = next_boundary(text, start);
    if word_start < text.len() && let Some(end) = matcher(text, word_start) {
      return Some(end);
    }
  }
  matcher(text, start)
}

fn word_branch_one_core(text: &str, start: usize) -> Option<usize> {
  let upper_end = scan_while(text, start, is_o200k_upper_or_shared);
  let lower_follows =
    upper_end < text.len() && is_o200k_lower_or_shared(char_at(text, upper_end));
  let lower_start = if lower_follows {
    upper_end
  } else if upper_end > start {
    text[start..upper_end]
      .char_indices()
      .rev()
      .find_map(|(offset, ch)| {
        is_o200k_lower_or_shared(ch).then_some(start + offset)
      })?
  } else {
    return None;
  };
  let end = scan_while(text, lower_start, is_o200k_lower_or_shared);
  Some(with_optional_contraction(text, end))
}

fn word_branch_two_core(text: &str, start: usize) -> Option<usize> {
  let upper_end = scan_while(text, start, is_o200k_upper_or_shared);
  if upper_end == start {
    return None;
  }
  let end = scan_while(text, upper_end, is_o200k_lower_or_shared);
  Some(with_optional_contraction(text, end))
}

fn with_optional_contraction(text: &str, end: usize) -> usize {
  if end < text.len() {
    case_insensitive_contraction_end(text, end).unwrap_or(end)
  } else {
    end
  }
}

fn scan_limited_numbers(text: &str, start: usize) -> usize {
  let mut end = start;
  for (count, (offset, ch)) in text[start..].char_indices().enumerate() {
    if count == 3 || !is_number(ch) {
      break;
    }
    end = start + offset + ch.len_utf8();
  }
  end
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
