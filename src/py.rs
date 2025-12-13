#[pyo3::pymodule(gil_used = false)]
mod _lib {
use std::path::PathBuf;

use pyo3::prelude::*;

use crate::{MyError, bpe::{BpeTrainer, CharIdx, Character, Idx}, spec::{gpt2::Gpt2Spec, uni::UniSpec}, traits::Train as _};

#[pyclass(subclass)]
pub struct BpeTrainerBase;

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

  pub fn step(&mut self, py: Python) -> PyResult<()> {
    py.detach(|| self.inner.step()).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
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
