pub(super) mod backend;
mod cl100k;
mod common;
mod engine;
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
    Some(engine::for_each::<gpt2::Gpt2>(text, emit))
  } else if pattern == cl100k::PATTERN {
    Some(engine::for_each::<cl100k::Cl100k>(text, emit))
  } else if pattern == o200k::PATTERN {
    Some(engine::for_each::<o200k::O200k>(text, emit))
  } else {
    None
  }
}

#[cfg(test)]
fn for_each_known_with_backend<
  'a,
  B: backend::Backend,
>(
  text: &'a str,
  pattern: &str,
  emit: impl FnMut(&'a str) -> MyResult<()>,
) -> Option<MyResult<()>> {
  if gpt2::recognizes(pattern) {
    Some(engine::for_each_with_backend::<gpt2::Gpt2, B>(
      text, emit,
    ))
  } else if pattern == cl100k::PATTERN {
    Some(engine::for_each_with_backend::<cl100k::Cl100k, B>(
      text, emit,
    ))
  } else if pattern == o200k::PATTERN {
    Some(engine::for_each_with_backend::<o200k::O200k, B>(
      text, emit,
    ))
  } else {
    None
  }
}

#[cfg(test)]
pub(super) fn for_each_known_scalar<'a>(
  text: &'a str,
  pattern: &str,
  emit: impl FnMut(&'a str) -> MyResult<()>,
) -> Option<MyResult<()>> {
  for_each_known_with_backend::<backend::Scalar>(text, pattern, emit)
}

#[cfg(all(test, target_arch = "aarch64"))]
pub(super) fn for_each_known_neon<'a>(
  text: &'a str,
  pattern: &str,
  emit: impl FnMut(&'a str) -> MyResult<()>,
) -> Option<MyResult<()>> {
  for_each_known_with_backend::<backend::Neon>(text, pattern, emit)
}

#[cfg(all(test, target_arch = "x86_64"))]
pub(super) fn for_each_known_sse2<'a>(
  text: &'a str,
  pattern: &str,
  emit: impl FnMut(&'a str) -> MyResult<()>,
) -> Option<MyResult<()>> {
  for_each_known_with_backend::<backend::Sse2>(text, pattern, emit)
}

#[cfg(all(test, target_arch = "x86_64"))]
pub(super) fn for_each_known_avx2<'a>(
  text: &'a str,
  pattern: &str,
  emit: impl FnMut(&'a str) -> MyResult<()>,
) -> Option<MyResult<()>> {
  for_each_known_with_backend::<backend::Avx2>(text, pattern, emit)
}
