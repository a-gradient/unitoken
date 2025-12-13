#[pyo3::pymodule(gil_used = false)]
mod _lib {
use std::{collections::BTreeMap, path::PathBuf};

use ordermap::OrderMap;
use pyo3::{prelude::*, pymethods};

use crate::{MyError, bpe::{BpeTrainer, CharIdx, CharSplit, Character, Idx, IdxLike, Word, utils::ToWord}, spec::{gpt2::Gpt2Spec, uni::UniSpec}, traits::{CanStrToWord, Train as _}};

#[pyclass(subclass)]
pub struct BpeTrainerBase;

#[allow(dead_code)]
/// this is just a reference for impl blocks, not directly used
pub trait BpeTrainerBaseImpl: Sized {
  fn new_py(special_tokens: Vec<String>) -> (Self, BpeTrainerBase);

  fn add_words(&mut self, py: Python, words: Vec<(String, i64)>);
  fn vocab_size(&self) -> usize;
  fn init_training(&mut self, py: Python);
  fn step(&mut self, py: Python) -> PyResult<i64>;
  fn get_vocabs(&self) -> Vocabs;
  fn save_vocab(&self, py: Python, path: PathBuf, spec: &str) -> PyResult<()>;
  fn save_merges_txt(&self, py: Python, path: PathBuf, spec: &str) -> PyResult<()>;
}

// #[pyclass(eq, eq_int)]
// #[derive(PartialEq)]
// pub enum SpecEnum {
//   #[pyo3(name = "gpt2")]
//   Gpt2,
//   #[pyo3(name = "uni")]
//   Uni,
// }

// #[pyclass(eq, eq_int)]
// #[derive(PartialEq)]
// pub enum CharLevel {
//   #[pyo3(name = "u8")]
//   U8,
//   #[pyo3(name = "char")]
//   Char,
// }

#[allow(non_camel_case_types)]
#[pyclass(extends = BpeTrainerBase)]
pub struct BpeTrainer_u8_Idx {
  pub inner: BpeTrainer<u8, Idx>,
}

#[pymethods]
impl BpeTrainer_u8_Idx {
  #[new]
  pub fn new_py(special_tokens: Vec<String>) -> (Self, BpeTrainerBase) {
    (
      Self {
        inner: BpeTrainer::new(vec![], special_tokens),
      },
      BpeTrainerBase {},
    )
  }

  pub fn add_words(&mut self, py: Python, words: Vec<(String, i64)>) {
    py.detach(||
      self.inner.add_words(&mut words.iter().map(|(w, f)| (w.as_str(), *f)))
    )
  }

  pub fn vocab_size(&self) -> usize {
    self.inner.vocab_size()
  }

  pub fn init_training(&mut self, py: Python) {
    py.detach(|| self.inner.init_training())
  }

  pub fn step(&mut self, py: Python) -> PyResult<i64> {
    py.detach(|| self.inner.step()).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(self.inner.vocab_size() as i64)
  }

  pub fn get_vocabs(&self) -> Vocabs {
    Vocabs {
      inner: Box::new(VocabsInner::new(&self.inner.vocab)),
    }
  }

  pub fn save_vocab(&self, py: Python, path: PathBuf, spec: &str) -> PyResult<()> {
    py.detach(|| {
      let mut file = std::fs::File::create(&path)?;
      let mut writer = std::io::BufWriter::new(&mut file);
      match spec {
        "gpt2" => self.inner.save_vocab_json(&Gpt2Spec, &mut writer),
        "uni" => self.inner.save_vocab_json(&UniSpec, &mut writer),
        _ => Err(MyError::SpecError(format!("Unknown spec: {}", spec))),
      }
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
  }

  pub fn save_merges_txt(&self, py: Python, path: PathBuf, spec: &str) -> PyResult<()> {
    py.detach(|| {
      let mut file = std::fs::File::create(&path)?;
      let mut writer = std::io::BufWriter::new(&mut file);
      match spec {
        "gpt2" => self.inner.save_merges_txt(&Gpt2Spec, &mut writer),
        "uni" => self.inner.save_merges_txt(&UniSpec, &mut writer),
        _ => Err(MyError::SpecError(format!("Unknown spec: {}", spec))),
      }
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
  }
}

#[allow(non_camel_case_types)]
#[pyclass(extends = BpeTrainerBase)]
pub struct BpeTrainer_Character_CharIdx {
  pub inner: BpeTrainer<Character, CharIdx>,
}

#[pymethods]
impl BpeTrainer_Character_CharIdx {
  #[new]
  pub fn new_py(special_tokens: Vec<String>) -> (Self, BpeTrainerBase) {
    (
      Self {
        inner: BpeTrainer::new(vec![], special_tokens),
      },
      BpeTrainerBase {},
    )
  }

  pub fn add_words(&mut self, py: Python, words: Vec<(String, i64)>) {
    py.detach(||
      self.inner.add_words(&mut words.iter().map(|(w, f)| (w.as_str(), *f)))
    )
  }

  pub fn vocab_size(&self) -> usize {
    self.inner.vocab_size()
  }

  pub fn init_training(&mut self, py: Python) {
    py.detach(|| self.inner.init_training())
  }

  pub fn step(&mut self, py: Python) -> PyResult<i64> {
    py.detach(|| self.inner.step()).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(self.inner.vocab_size() as i64)
  }

  pub fn get_vocabs(&self) -> Vocabs {
    Vocabs {
      inner: Box::new(VocabsInner::new(&self.inner.vocab)),
    }
  }

  pub fn save_vocab(&self, py: Python, path: PathBuf, spec: &str) -> PyResult<()> {
    py.detach(|| {
      let mut file = std::fs::File::create(&path)?;
      let mut writer = std::io::BufWriter::new(&mut file);
      match spec {
        "gpt2" => Err(MyError::SpecError("gpt2 spec not supported for Character tokenizer".to_string())),
        "uni" => self.inner.save_vocab_json(&UniSpec, &mut writer),
        _ => Err(MyError::SpecError(format!("Unknown spec: {}", spec))),
      }
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
  }

  pub fn save_merges_txt(&self, py: Python, path: PathBuf, spec: &str) -> PyResult<()> {
    py.detach(|| {
      let mut file = std::fs::File::create(&path)?;
      let mut writer = std::io::BufWriter::new(&mut file);
      match spec {
        "gpt2" => Err(MyError::SpecError("gpt2 spec not supported for Character tokenizer".to_string())),
        "uni" => self.inner.save_merges_txt(&UniSpec, &mut writer),
        _ => Err(MyError::SpecError(format!("Unknown spec: {}", spec))),
      }
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
  }
}

pub struct VocabsInner<C, I>(OrderMap<Word<C>, I>);

impl<C: std::hash::Hash + Eq, I: IdxLike> VocabsInner<C, I> {
  pub fn new(vocab: &BTreeMap<I, Word<C>>) -> Self {
    Self(vocab.iter().map(|(i, c)| (c.clone(), i.clone())).collect())
  }
}

trait VocabsImpl {
  fn len(&self) -> usize;
  fn get(&self, word: &str) -> Option<i64>;
  fn items(&self) -> Vec<(Vec<u8>, i64)>;
}

impl<C: CanStrToWord + CharSplit + std::hash::Hash + Eq, I: IdxLike> VocabsImpl for VocabsInner<C, I> {
  fn len(&self) -> usize {
    self.0.len()
  }

  fn get(&self, word: &str) -> Option<i64> {
    self.0.get(&word.to_word()).map(|i| i.to_u64() as i64)
  }

  fn items(&self) -> Vec<(Vec<u8>, i64)> {
    self.0.iter().map(|(w, i)| (CharSplit::to_vec_u8(w), i.to_u64() as i64)).collect()
  }
}

#[pyclass]
pub struct Vocabs {
  inner: Box<dyn VocabsImpl + Send + Sync>,
}

#[pymethods]
impl Vocabs {
  #[getter]
  pub fn len(&self) -> usize {
    self.inner.len()
  }

  pub fn get(&self, word: &str) -> Option<i64> {
    self.inner.get(word)
  }

  pub fn items(&self) -> Vec<(Vec<u8>, i64)> {
    self.inner.items()
  }
}

#[pymodule_export]
pub use crate::pretokenizer::PreTokenizer;

#[pymethods]
impl PreTokenizer {
  #[new]
  pub fn new_py(special_tokens: Vec<String>, eot_token: Option<String>) -> Self {
    Self::new(&special_tokens, eot_token.as_deref())
  }

  #[pyo3(name = "find_chunk_boundaries")]
  pub fn py_find_chunk_boundaries(
    &self, path: PathBuf, desired_num_chunks: usize,
  ) -> PyResult<Vec<(u64, usize)>> {
    self.find_chunk_boundaries(path, desired_num_chunks)
      .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
  }

  #[pyo3(name = "get_words_from_segment")]
  pub fn py_get_words_from_segment(
    &self, path: PathBuf, offset: u64, length: usize,
  ) -> PyResult<BTreeMap<String, i64>> {
    self.get_words_from_segment(path, offset, length)
      .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
  }

  #[pyo3(name = "get_words_from_file")]
  pub fn py_get_words_from_file(
    &self, path: PathBuf, desired_num_chunks: usize,
  ) -> PyResult<BTreeMap<String, i64>> {
    self.get_words_from_file(path, desired_num_chunks)
      .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
  }
}

#[pyclass]
pub struct BpeEncoderBase;

// #[pymodule(gil_used = false)]
// #[pyo3(name="_lib")]
// fn _tiktoken(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
//   m.add_class::<BpeTrainerBase>()?;
//   m.add_class::<BpeTrainer_u8_Idx>()?;
//   m.add_class::<BpeTrainer_Character_CharIdx>()?;
//   Ok(())
// }


}

#[test]
#[ignore = "manual"]
fn generate_py_stubs() {
  println!("test");
  let module = pyo3_introspection::introspect_cdylib(
      "./python/unitoken/_lib.cpython-313-darwin.so",
      "_lib",
  )
  .expect("introspection to succeed");
  let result = pyo3_introspection::module_stub_files(&module);
  println!("{result:?}");
  let value = result.get(&std::path::PathBuf::from("__init__.pyi")).unwrap();
  std::fs::write("./python/unitoken/_lib.pyi", value).unwrap();
}
