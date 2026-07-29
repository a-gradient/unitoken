mod cl100k;
mod common;
mod gpt2;
mod o200k;

use crate::MyResult;

pub(super) const GPT2_PATTERN: &str = gpt2::PATTERN;
pub(super) const R50K_PATTERN: &str = gpt2::R50K_PATTERN;
pub(super) const CL100K_PATTERN: &str = cl100k::PATTERN;
pub(super) const O200K_PATTERN: &str = o200k::PATTERN;

pub(super) fn for_each_known<'a>(
  text: &'a str,
  pattern: &str,
  emit: impl FnMut(&'a str) -> MyResult<()>,
) -> Option<MyResult<()>> {
  if gpt2::recognizes(pattern) {
    Some(gpt2::for_each(text, emit))
  } else if pattern == cl100k::PATTERN {
    Some(cl100k::for_each(text, emit))
  } else if pattern == o200k::PATTERN {
    Some(o200k::for_each(text, emit))
  } else {
    None
  }
}
