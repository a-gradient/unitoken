use crate::bpe::{BpeTrainer, CharToIdx, HasChar, IdxLike, Word, utils::{ToWord, WordDebugExt}};

pub trait CanToWord<C, T>
where
  Self: Imply<T, Is: ToWord<C>>,
{}

impl<C, T> CanToWord<C, T> for C
where
  T: ToWord<C>,
{}

pub trait CanTrain<C, I>
where
  Self: Imply<Word<C>, Is: WordDebugExt>,
  Self: for<'a> Imply<C, Is: CanToWord<C, &'a str>>,
  Self: Imply<C, Is: Clone + Ord + Send + Sync + CharToIdx<I> + CanToWord<C, u8> + 'static>,
  Self: Imply<I, Is: IdxLike + HasChar<C>>,
{}

impl<C, I> CanTrain<C, I> for BpeTrainer<C, I>
where
  Word<C>: WordDebugExt,
  for<'a> &'a str: ToWord<C>,
  C: Clone + Ord + Send + Sync + CharToIdx<I> + 'static,
  I: IdxLike + HasChar<C>,
  u8: ToWord<C>,
{}

// https://docs.rs/imply-hack/latest/imply_hack/
// https://github.com/rust-lang/rust/issues/44491#issuecomment-2496196742
pub trait Imply<T>: ImplyHack<T, Is = T> {}

impl<T, U> Imply<T> for U {}

pub trait ImplyHack<T> {
  type Is;
}

impl<T, U> ImplyHack<T> for U {
  type Is = T;
}
