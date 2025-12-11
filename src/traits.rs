use crate::{MyResult, bpe::{BpeEncoder, BpeTrainer, CharSplit, CharToIdx, Freq, HasChar, Idx, IdxLike, Word, utils::{ToWord, WordDebugExt}}};

pub trait Train {
  fn new(special_tokens: Vec<String>) -> Self;
  fn add_words(&mut self, words: &mut dyn Iterator<Item = (&str, Freq)>);
  fn vocab_size(&self) -> usize;
  fn init_training(&mut self);
  fn step(&mut self) -> MyResult<()>;
  fn train(&mut self, vocab_size: usize) -> MyResult<()> {
    self.init_training();
    loop {
      self.step()?;
      if self.vocab_size() >= vocab_size {
        break;
      }
    }
    Ok(())
  }
}

pub trait CanToWord<C, T>
where
  Self: Imply<T, Is: ToWord<C>>,
{}

impl<C, T> CanToWord<C, T> for C
where
  T: ToWord<C>,
{}

pub trait CanStrToWord<C>
where
  Self: for<'a> Imply<C, Is: CanToWord<C, &'a str>>,
{}

impl<C> CanStrToWord<C> for C
where
  for<'a> &'a str: ToWord<C>,
{}

pub trait CanTrain<C, I>
where
  Self: Imply<Word<C>, Is: WordDebugExt>,
  Self: Imply<C, Is: Clone + Ord + Send + Sync + 'static>,
  Self: Imply<C, Is: CharToIdx<I> + CanToWord<C, u8> + CanStrToWord<C>>,
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

pub trait CanEncode<C, I>
where
  Self: Imply<Word<C>, Is: WordDebugExt>,
  Self: Imply<C, Is: Ord + std::hash::Hash + Clone + Send + Sync + 'static>,
  Self: Imply<C, Is: CharSplit + CanStrToWord<C>>,
  Self: Imply<I, Is: IdxLike>,
{}

impl<C> CanEncode<C, Idx> for BpeEncoder<C>
where
  C: Ord + std::hash::Hash + CharSplit + CanStrToWord<C> + Clone + Send + Sync + 'static,
  Word<C>: WordDebugExt,
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
