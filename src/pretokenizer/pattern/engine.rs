//! Shared token-emission loop with pattern-specific boundary decisions.

use crate::MyResult;

use super::backend::{Backend, Scalar};

pub(super) trait Pattern {
  fn pretoken_end<B: Backend>(
    text: &str,
    start: usize,
    backend: &mut B,
  ) -> usize;
}

pub(super) fn for_each<'a, P: Pattern>(
  text: &'a str,
  emit: impl FnMut(&'a str) -> MyResult<()>,
) -> MyResult<()> {
  #[cfg(target_arch = "aarch64")]
  {
    use super::backend::Neon;

    if std::arch::is_aarch64_feature_detected!("neon") {
      return for_each_with_backend::<P, Neon>(text, emit);
    }
  }

  #[cfg(target_arch = "x86_64")]
  {
    use super::backend::{Avx2, Sse2};

    if std::arch::is_x86_feature_detected!("avx2") {
      return for_each_with_backend::<P, Avx2>(text, emit);
    }
    return for_each_with_backend::<P, Sse2>(text, emit);
  }

  #[allow(unreachable_code)]
  for_each_with_backend::<P, Scalar>(text, emit)
}

pub(super) fn for_each_with_backend<
  'a,
  P: Pattern,
  B: Backend,
>(
  text: &'a str,
  mut emit: impl FnMut(&'a str) -> MyResult<()>,
) -> MyResult<()> {
  let mut backend = B::default();
  let mut start = 0;
  while start < text.len() {
    let end = P::pretoken_end(text, start, &mut backend);
    debug_assert!(end > start);
    debug_assert!(text.is_char_boundary(end));
    emit(&text[start..end])?;
    start = end;
  }
  Ok(())
}
