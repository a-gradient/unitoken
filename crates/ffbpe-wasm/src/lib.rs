use std::collections::BTreeMap;

use ffbpe::{
  MyError, MyResult,
  bpe::{
    BpeEncoder as CoreBpeEncoder, BpeModel as CoreBpeModel,
    BpeTrainer as CoreBpeTrainer, BpeTrainerConfig, CharIdx, CharSplit,
    Character, Idx, IdxLike, InitialAlphabet, TieBreak,
    encoder::BpeBuilder,
  },
  counter::{BigramCounter as CoreBigramCounter, WordCounter as CoreWordCounter},
  pretokenizer::{
    PreTokenizer as CorePreTokenizer, UnicodeBigramMixedBoundary,
    parse_unicode_bigrams, unicode_bigram_to_string,
  },
  spec::{Spec, gpt2::Gpt2Spec, unitoken::UnitokenSpec},
  traits::{CanEncode, Encode as _, Train as _},
};
use js_sys::Uint32Array;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

fn js_error(error: impl ToString) -> JsValue {
  js_sys::Error::new(&error.to_string()).into()
}

fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
  value.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
    .map_err(js_error)
}

fn from_js<T: for<'de> Deserialize<'de>>(value: JsValue) -> Result<T, JsValue> {
  serde_wasm_bindgen::from_value(value).map_err(js_error)
}

fn frequency_from_js(value: f64, name: &str) -> Result<i64, JsValue> {
  if !value.is_finite()
    || value.fract() != 0.0
    || value < i64::MIN as f64
    || value > i64::MAX as f64
  {
    return Err(js_error(format!("{name} must be a finite integer")));
  }
  Ok(value as i64)
}

fn configure_pre_tokenizer(
  mut pre_tokenizer: CorePreTokenizer,
  unicode_bigrams: Option<&[String]>,
  mixed_boundary: &str,
) -> MyResult<CorePreTokenizer> {
  if let Some(bigrams) = unicode_bigrams {
    pre_tokenizer = pre_tokenizer.with_unicode_bigrams(parse_unicode_bigrams(bigrams)?);
  }
  Ok(pre_tokenizer.with_unicode_bigram_mixed_boundary(
    UnicodeBigramMixedBoundary::parse(mixed_boundary)?,
  ))
}

fn configure_encoder<C>(
  mut encoder: CoreBpeEncoder<C>,
  unicode_bigrams: Option<&[String]>,
  mixed_boundary: &str,
) -> MyResult<CoreBpeEncoder<C>>
where
  CoreBpeEncoder<C>: CanEncode<C, Idx>,
  C: Clone,
{
  encoder.pre_tokenizer = configure_pre_tokenizer(
    encoder.pre_tokenizer,
    unicode_bigrams,
    mixed_boundary,
  )?;
  Ok(encoder)
}

#[derive(Clone, Deserialize)]
#[serde(default)]
struct PreTokenizerOptions {
  eot_token: Option<String>,
  pat_str: Option<String>,
  unicode_bigrams: Option<Vec<String>>,
  unicode_bigram_mixed_boundary: String,
}

impl Default for PreTokenizerOptions {
  fn default() -> Self {
    Self {
      eot_token: None,
      pat_str: None,
      unicode_bigrams: None,
      unicode_bigram_mixed_boundary: "keep".to_string(),
    }
  }
}

#[wasm_bindgen]
pub struct WasmPreTokenizer {
  inner: CorePreTokenizer,
}

#[wasm_bindgen]
impl WasmPreTokenizer {
  #[wasm_bindgen(constructor)]
  pub fn new(special_tokens: Vec<String>, options: JsValue) -> Result<WasmPreTokenizer, JsValue> {
    let options = if options.is_null() || options.is_undefined() {
      PreTokenizerOptions::default()
    } else {
      from_js(options)?
    };
    let pre_tokenizer = CorePreTokenizer::try_new(
      &special_tokens,
      options.eot_token.as_deref(),
      options.pat_str.as_deref(),
    ).map_err(js_error)?;
    let inner = configure_pre_tokenizer(
      pre_tokenizer,
      options.unicode_bigrams.as_deref(),
      &options.unicode_bigram_mixed_boundary,
    ).map_err(js_error)?;
    Ok(Self { inner })
  }

  #[wasm_bindgen(js_name = withUnicodeBigrams)]
  pub fn with_unicode_bigrams(&self, bigrams: Vec<String>) -> Result<WasmPreTokenizer, JsValue> {
    let inner = self.inner.clone().with_unicode_bigrams(
      parse_unicode_bigrams(&bigrams).map_err(js_error)?,
    );
    Ok(Self { inner })
  }

  #[wasm_bindgen(js_name = getWords)]
  pub fn get_words(&self, text: &str) -> Result<JsValue, JsValue> {
    to_js(&self.inner.get_words_owned(text).map_err(js_error)?)
  }

  #[wasm_bindgen(js_name = bigramCounter)]
  pub fn bigram_counter(&self) -> WasmBigramCounter {
    WasmBigramCounter {
      inner: CoreBigramCounter::new(self.inner.clone()),
    }
  }

  #[wasm_bindgen(js_name = wordCounter)]
  pub fn word_counter(&self) -> WasmWordCounter {
    WasmWordCounter {
      inner: CoreWordCounter::new(self.inner.clone()),
    }
  }

  #[wasm_bindgen(js_name = loadWordCounter)]
  pub fn load_word_counter(&self, serialized: &str) -> Result<WasmWordCounter, JsValue> {
    let inner = CoreWordCounter::load(self.inner.clone(), serialized.as_bytes())
      .map_err(js_error)?;
    Ok(WasmWordCounter { inner })
  }
}

#[derive(Serialize)]
struct BigramSelection {
  bigrams: Vec<String>,
  cutoff_freq: Option<i64>,
  max_excluded_freq: Option<i64>,
}

fn bigram_selection(counter: &CoreBigramCounter, top_k: usize, min_freq: i64) -> BigramSelection {
  let selection = counter.selection(top_k, min_freq);
  let mut bigrams = selection.bigrams.into_iter()
    .map(unicode_bigram_to_string)
    .collect::<Vec<_>>();
  bigrams.sort_unstable();
  BigramSelection {
    bigrams,
    cutoff_freq: selection.cutoff_freq,
    max_excluded_freq: selection.max_excluded_freq,
  }
}

#[wasm_bindgen]
pub struct WasmBigramCounter {
  inner: CoreBigramCounter,
}

#[wasm_bindgen]
impl WasmBigramCounter {
  #[wasm_bindgen(js_name = addText)]
  pub fn add_text(&mut self, text: &str) -> Result<(), JsValue> {
    self.inner.add_text(text).map_err(js_error)
  }

  #[wasm_bindgen(js_name = addBatch)]
  pub fn add_batch(&mut self, texts: Vec<String>) -> Result<(), JsValue> {
    self.inner.add_batch(&texts).map_err(js_error)
  }

  pub fn merge(&mut self, other: WasmBigramCounter) -> Result<(), JsValue> {
    self.inner.merge(other.inner).map_err(js_error)
  }

  pub fn selected(&self, top_k: usize, min_freq: f64) -> Result<Vec<String>, JsValue> {
    Ok(bigram_selection(
      &self.inner,
      top_k,
      frequency_from_js(min_freq, "min_freq")?,
    ).bigrams)
  }

  pub fn select(&self, top_k: usize, min_freq: f64) -> Result<JsValue, JsValue> {
    to_js(&bigram_selection(
      &self.inner,
      top_k,
      frequency_from_js(min_freq, "min_freq")?,
    ))
  }

  pub fn items(&self) -> Result<JsValue, JsValue> {
    let mut items = self.inner.counts().iter()
      .map(|(bigram, frequency)| (unicode_bigram_to_string(*bigram), *frequency))
      .collect::<Vec<_>>();
    items.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    to_js(&items)
  }
}

#[wasm_bindgen]
pub struct WasmWordCounter {
  inner: CoreWordCounter,
}

#[wasm_bindgen]
impl WasmWordCounter {
  #[wasm_bindgen(js_name = addText)]
  pub fn add_text(&mut self, text: &str) -> Result<(), JsValue> {
    self.inner.add_text(text).map_err(js_error)
  }

  #[wasm_bindgen(js_name = addBatch)]
  pub fn add_batch(&mut self, texts: Vec<String>) -> Result<(), JsValue> {
    self.inner.add_batch(&texts).map_err(js_error)
  }

  pub fn merge(&mut self, other: WasmWordCounter) -> Result<(), JsValue> {
    self.inner.merge(other.inner).map_err(js_error)
  }

  pub fn words(&self) -> Result<JsValue, JsValue> {
    to_js(&self.inner.words())
  }

  #[wasm_bindgen(getter)]
  pub fn len(&self) -> usize {
    self.inner.len()
  }

  #[wasm_bindgen(getter, js_name = isEmpty)]
  pub fn is_empty(&self) -> bool {
    self.inner.is_empty()
  }

  pub fn clear(&mut self) {
    self.inner.clear();
  }

  pub fn serialize(&self) -> Result<String, JsValue> {
    let mut output = Vec::new();
    self.inner.save(&mut output).map_err(js_error)?;
    String::from_utf8(output).map_err(js_error)
  }
}

#[derive(Clone, Deserialize)]
#[serde(default)]
struct TrainerOptions {
  unit: String,
  initial_alphabet: String,
  tie_break: String,
  parallel_merge_min_occurs_in: Option<usize>,
  hot_pair_window_size: Option<usize>,
  bigram_cutoff_freq: Option<i64>,
}

impl Default for TrainerOptions {
  fn default() -> Self {
    Self {
      unit: "byte".to_string(),
      initial_alphabet: "raw".to_string(),
      tie_break: "smallest_pair_id".to_string(),
      parallel_merge_min_occurs_in: None,
      hot_pair_window_size: None,
      bigram_cutoff_freq: None,
    }
  }
}

impl TrainerOptions {
  fn config(&self) -> MyResult<BpeTrainerConfig> {
    let initial_alphabet = match self.initial_alphabet.as_str() {
      "raw" => InitialAlphabet::RawBytes,
      "byte_level" => InitialAlphabet::ByteLevel,
      value => return Err(MyError::SpecError(format!("Unknown initial_alphabet: {value}"))),
    };
    let tie_break = match self.tie_break.as_str() {
      "smallest_pair_id" => TieBreak::SmallestPairId,
      "largest_content" => TieBreak::LargestContent,
      value => return Err(MyError::SpecError(format!("Unknown tie_break: {value}"))),
    };
    if self.hot_pair_window_size == Some(0) {
      return Err(MyError::SpecError("hot_pair_window_size must be positive".to_string()));
    }
    if self.bigram_cutoff_freq.is_some_and(|frequency| frequency <= 0) {
      return Err(MyError::SpecError("bigram_cutoff_freq must be positive".to_string()));
    }
    Ok(BpeTrainerConfig {
      initial_alphabet,
      tie_break,
      parallel_merge_min_occurs_in: self.parallel_merge_min_occurs_in,
      hot_pair_window_size: self.hot_pair_window_size,
      bigram_cutoff_freq: self.bigram_cutoff_freq,
    })
  }
}

enum TrainerInner {
  Byte(CoreBpeTrainer<u8, Idx>),
  Unicode(CoreBpeTrainer<Character, CharIdx>),
}

#[derive(Serialize)]
struct HotPairWindowStats {
  hydration_scans: u64,
  hydrated_word_entries: u64,
  batch_prunes: u64,
  prune_evictions: u64,
  peak_resident_pairs: usize,
  resident_pairs: usize,
  occurrence_capacity: usize,
}

#[derive(Serialize)]
struct TrainerMemoryUsage {
  word_entries: usize,
  word_entry_capacity: usize,
  word_storage_bytes: usize,
  pair_entries: usize,
  pair_table_capacity: usize,
  pair_table_bytes: usize,
  occurrence_set_slots: usize,
  occurrence_set_slot_capacity: usize,
  occurrence_set_header_bytes: usize,
  occurrence_capacity_entries: usize,
  occurrence_capacity_bytes: usize,
  merge_heap_entries: usize,
  merge_heap_capacity: usize,
  merge_heap_bytes: usize,
  merge_entries: usize,
  merge_storage_bytes: usize,
  vocab_entries: usize,
  vocab_token_bytes: usize,
  estimated_persistent_bytes: usize,
}

fn trainer_memory_usage<C, I>(trainer: &CoreBpeTrainer<C, I>) -> TrainerMemoryUsage {
  let usage = trainer.memory_usage();
  TrainerMemoryUsage {
    word_entries: usage.word_entries,
    word_entry_capacity: usage.word_entry_capacity,
    word_storage_bytes: usage.word_storage_bytes,
    pair_entries: usage.pair_entries,
    pair_table_capacity: usage.pair_table_capacity,
    pair_table_bytes: usage.pair_table_bytes,
    occurrence_set_slots: usage.occurrence_set_slots,
    occurrence_set_slot_capacity: usage.occurrence_set_slot_capacity,
    occurrence_set_header_bytes: usage.occurrence_set_header_bytes,
    occurrence_capacity_entries: usage.occurrence_capacity_entries,
    occurrence_capacity_bytes: usage.occurrence_capacity_bytes,
    merge_heap_entries: usage.merge_heap_entries,
    merge_heap_capacity: usage.merge_heap_capacity,
    merge_heap_bytes: usage.merge_heap_bytes,
    merge_entries: usage.merge_entries,
    merge_storage_bytes: usage.merge_storage_bytes,
    vocab_entries: usage.vocab_entries,
    vocab_token_bytes: usage.vocab_token_bytes,
    estimated_persistent_bytes: usage.estimated_persistent_bytes,
  }
}

fn hot_pair_window_stats<C, I>(trainer: &CoreBpeTrainer<C, I>) -> Option<HotPairWindowStats> {
  let stats = trainer.hot_pair_window_stats()?;
  Some(HotPairWindowStats {
    hydration_scans: stats.hydration_scans,
    hydrated_word_entries: stats.hydrated_word_entries,
    batch_prunes: stats.batch_prunes,
    prune_evictions: stats.prune_evictions,
    peak_resident_pairs: stats.peak_resident_pairs,
    resident_pairs: trainer.hot_resident_pairs(),
    occurrence_capacity: trainer.hot_occurrence_capacity(),
  })
}

#[wasm_bindgen]
pub struct WasmBpeTrainer {
  inner: TrainerInner,
}

#[wasm_bindgen]
impl WasmBpeTrainer {
  #[wasm_bindgen(constructor)]
  pub fn new(special_tokens: Vec<String>, options: JsValue) -> Result<WasmBpeTrainer, JsValue> {
    let options = if options.is_null() || options.is_undefined() {
      TrainerOptions::default()
    } else {
      from_js(options)?
    };
    let config = options.config().map_err(js_error)?;
    let inner = match options.unit.as_str() {
      "byte" => TrainerInner::Byte(CoreBpeTrainer::new_with_config(
        vec![], special_tokens, config,
      )),
      "unicode" => TrainerInner::Unicode(CoreBpeTrainer::new_with_config(
        vec![], special_tokens, config,
      )),
      value => return Err(js_error(format!("Unknown unit: {value}"))),
    };
    Ok(Self { inner })
  }

  #[wasm_bindgen(getter)]
  pub fn unit(&self) -> String {
    match self.inner {
      TrainerInner::Byte(_) => "byte",
      TrainerInner::Unicode(_) => "unicode",
    }.to_string()
  }

  #[wasm_bindgen(getter, js_name = vocabSize)]
  pub fn vocab_size(&self) -> usize {
    match &self.inner {
      TrainerInner::Byte(trainer) => trainer.vocab_size(),
      TrainerInner::Unicode(trainer) => trainer.vocab_size(),
    }
  }

  #[wasm_bindgen(getter, js_name = lastMergeFreq)]
  pub fn last_merge_freq(&self) -> Result<JsValue, JsValue> {
    let frequency = match &self.inner {
      TrainerInner::Byte(trainer) => trainer.last_merge_freq(),
      TrainerInner::Unicode(trainer) => trainer.last_merge_freq(),
    };
    to_js(&frequency)
  }

  #[wasm_bindgen(getter, js_name = hotPairWindowStats)]
  pub fn hot_pair_window_stats(&self) -> Result<JsValue, JsValue> {
    match &self.inner {
      TrainerInner::Byte(trainer) => to_js(&hot_pair_window_stats(trainer)),
      TrainerInner::Unicode(trainer) => to_js(&hot_pair_window_stats(trainer)),
    }
  }

  #[wasm_bindgen(getter, js_name = memoryUsage)]
  pub fn memory_usage(&self) -> Result<JsValue, JsValue> {
    match &self.inner {
      TrainerInner::Byte(trainer) => to_js(&trainer_memory_usage(trainer)),
      TrainerInner::Unicode(trainer) => to_js(&trainer_memory_usage(trainer)),
    }
  }

  #[wasm_bindgen(js_name = addWords)]
  pub fn add_words(&mut self, words: JsValue) -> Result<(), JsValue> {
    let words: Vec<(String, i64)> = from_js(words)?;
    match &mut self.inner {
      TrainerInner::Byte(trainer) => {
        trainer.add_words(&mut words.iter().map(|(word, frequency)| (word.as_str(), *frequency)));
      }
      TrainerInner::Unicode(trainer) => {
        trainer.add_words(&mut words.iter().map(|(word, frequency)| (word.as_str(), *frequency)));
      }
    }
    Ok(())
  }

  #[wasm_bindgen(js_name = addWordCounter)]
  pub fn add_word_counter(&mut self, counter: &mut WasmWordCounter) {
    let words = counter.inner.words();
    counter.inner.clear();
    match &mut self.inner {
      TrainerInner::Byte(trainer) => {
        trainer.add_words(&mut words.iter().map(|(word, frequency)| (word.as_str(), *frequency)));
      }
      TrainerInner::Unicode(trainer) => {
        trainer.add_words(&mut words.iter().map(|(word, frequency)| (word.as_str(), *frequency)));
      }
    }
  }

  #[wasm_bindgen(js_name = initTraining)]
  pub fn init_training(&mut self) {
    match &mut self.inner {
      TrainerInner::Byte(trainer) => trainer.init_training(),
      TrainerInner::Unicode(trainer) => trainer.init_training(),
    }
  }

  pub fn train(&mut self, vocab_size: usize) -> Result<(), JsValue> {
    match &mut self.inner {
      TrainerInner::Byte(trainer) => trainer.train_until(vocab_size),
      TrainerInner::Unicode(trainer) => trainer.train_until(vocab_size),
    }.map_err(js_error)
  }

  #[wasm_bindgen(js_name = trainWithBbpeFallback)]
  pub fn train_with_bbpe_fallback(
    &mut self, vocab_size: usize, primary_vocab_ratio: f64,
  ) -> Result<(), JsValue> {
    if !primary_vocab_ratio.is_finite() || !(0.0..=1.0).contains(&primary_vocab_ratio) {
      return Err(js_error("primary_vocab_ratio must be finite and between 0 and 1"));
    }
    match &mut self.inner {
      TrainerInner::Unicode(trainer) => trainer
        .train_until_with_bbpe_fallback(vocab_size, primary_vocab_ratio)
        .map_err(js_error),
      TrainerInner::Byte(_) => Err(js_error("trainWithBbpeFallback requires unit=\"unicode\"")),
    }
  }

  pub fn step(&mut self) -> Result<usize, JsValue> {
    match &mut self.inner {
      TrainerInner::Byte(trainer) => {
        trainer.step().map_err(js_error)?;
        Ok(trainer.vocab_size())
      }
      TrainerInner::Unicode(trainer) => {
        trainer.step().map_err(js_error)?;
        Ok(trainer.vocab_size())
      }
    }
  }

  #[wasm_bindgen(js_name = vocabItems)]
  pub fn vocab_items(&self) -> Result<JsValue, JsValue> {
    let items = match &self.inner {
      TrainerInner::Byte(trainer) => vocab_items(trainer.vocab.iter()),
      TrainerInner::Unicode(trainer) => vocab_items(trainer.vocab.iter()),
    };
    to_js(&items)
  }

  #[wasm_bindgen(js_name = validateModel)]
  pub fn validate_model(&self) -> Result<WasmBpeModel, JsValue> {
    let inner = match &self.inner {
      TrainerInner::Byte(trainer) => {
        ModelInner::Byte(trainer.validate_model().map_err(js_error)?)
      }
      TrainerInner::Unicode(trainer) => {
        ModelInner::Unicode(trainer.validate_model().map_err(js_error)?)
      }
    };
    Ok(WasmBpeModel { inner })
  }
}

fn vocab_items<'a, C: CharSplit + 'a, I: IdxLike + 'a>(
  vocab: impl Iterator<Item = (&'a I, &'a ffbpe::bpe::Word<C>)>,
) -> Vec<(Vec<u8>, i64)> {
  vocab.map(|(index, word)| (C::to_vec_u8(word), index.to_u64() as i64)).collect()
}

enum ModelInner {
  Byte(CoreBpeModel<u8, Idx>),
  Unicode(CoreBpeModel<Character, CharIdx>),
}

#[derive(Clone, Deserialize)]
#[serde(default)]
struct EncoderOptions {
  unit: String,
  format: Option<String>,
  special_tokens: Option<Vec<String>>,
  pat_str: Option<String>,
  unicode_bigrams: Option<Vec<String>>,
  unicode_bigram_mixed_boundary: String,
  split_on_vocab_bigrams: bool,
}

impl Default for EncoderOptions {
  fn default() -> Self {
    Self {
      unit: "byte".to_string(),
      format: None,
      special_tokens: None,
      pat_str: None,
      unicode_bigrams: None,
      unicode_bigram_mixed_boundary: "keep".to_string(),
      split_on_vocab_bigrams: true,
    }
  }
}

impl EncoderOptions {
  fn format(&self) -> MyResult<&str> {
    let format = self.format.as_deref().unwrap_or_else(|| {
      if self.unit == "unicode" { "unitoken" } else { "gpt2" }
    });
    match (self.unit.as_str(), format) {
      ("byte", "gpt2" | "unitoken") | ("unicode", "unitoken") => Ok(format),
      ("byte" | "unicode", _) => Err(MyError::SpecError(format!(
        "format {format} is not compatible with unit {}", self.unit,
      ))),
      _ => Err(MyError::SpecError(format!("Unknown unit: {}", self.unit))),
    }
  }
}

enum EncoderInner {
  Byte(CoreBpeEncoder<u8>),
  Unicode(CoreBpeEncoder<Character>),
}

fn build_encoder_from_data<C>(
  vocab: BTreeMap<Idx, Vec<u8>>,
  merges: Vec<(Vec<u8>, Vec<u8>)>,
  options: &EncoderOptions,
  spec: &dyn Spec<C, Idx>,
) -> MyResult<CoreBpeEncoder<C>>
where
  CoreBpeEncoder<C>: CanEncode<C, Idx>,
  C: Clone,
{
  let builder = BpeBuilder::new()
    .set_vocab(vocab)
    .set_merges_raw(merges)
    .set_special_tokens(options.special_tokens.clone())
    .set_pat_str(options.pat_str.clone())
    .set_split_on_vocab_bigrams(options.split_on_vocab_bigrams);
  configure_encoder(
    builder.build(spec)?,
    options.unicode_bigrams.as_deref(),
    &options.unicode_bigram_mixed_boundary,
  )
}

fn build_encoder_from_files<C>(
  vocab: &str,
  merges: &str,
  options: &EncoderOptions,
  spec: &dyn Spec<C, Idx>,
) -> MyResult<CoreBpeEncoder<C>>
where
  CoreBpeEncoder<C>: CanEncode<C, Idx>,
  C: Clone,
{
  let builder = BpeBuilder::new()
    .load_vocab_reader(vocab.as_bytes(), spec)?
    .load_merges_reader(merges.as_bytes(), spec)?
    .set_special_tokens(options.special_tokens.clone())
    .set_pat_str(options.pat_str.clone())
    .set_split_on_vocab_bigrams(options.split_on_vocab_bigrams);
  configure_encoder(
    builder.build(spec)?,
    options.unicode_bigrams.as_deref(),
    &options.unicode_bigram_mixed_boundary,
  )
}

#[wasm_bindgen]
pub struct WasmBpeEncoder {
  inner: EncoderInner,
}

#[wasm_bindgen]
impl WasmBpeEncoder {
  #[wasm_bindgen(js_name = fromData)]
  pub fn from_data(
    vocab: JsValue, merges: JsValue, options: JsValue,
  ) -> Result<WasmBpeEncoder, JsValue> {
    let vocab: Vec<(Vec<u8>, Idx)> = from_js(vocab)?;
    let merges: Vec<(Vec<u8>, Vec<u8>)> = from_js(merges)?;
    let options: EncoderOptions = from_js(options)?;
    options.format().map_err(js_error)?;
    let vocab = vocab.into_iter().map(|(word, index)| (index, word)).collect();
    let inner = match options.unit.as_str() {
      "byte" => EncoderInner::Byte(build_encoder_from_data(
        vocab, merges, &options,
        if options.format.as_deref() == Some("unitoken") { &UnitokenSpec } else { &Gpt2Spec },
      ).map_err(js_error)?),
      "unicode" => EncoderInner::Unicode(build_encoder_from_data(
        vocab, merges, &options, &UnitokenSpec,
      ).map_err(js_error)?),
      value => return Err(js_error(format!("Unknown unit: {value}"))),
    };
    Ok(Self { inner })
  }

  #[wasm_bindgen(js_name = fromFiles)]
  pub fn from_files(
    vocab: &str, merges: &str, options: JsValue,
  ) -> Result<WasmBpeEncoder, JsValue> {
    let options: EncoderOptions = from_js(options)?;
    let format = options.format().map_err(js_error)?;
    let inner = match (options.unit.as_str(), format) {
      ("byte", "gpt2") => EncoderInner::Byte(build_encoder_from_files(
        vocab, merges, &options, &Gpt2Spec,
      ).map_err(js_error)?),
      ("byte", "unitoken") => EncoderInner::Byte(build_encoder_from_files(
        vocab, merges, &options, &UnitokenSpec,
      ).map_err(js_error)?),
      ("unicode", "unitoken") => EncoderInner::Unicode(build_encoder_from_files(
        vocab, merges, &options, &UnitokenSpec,
      ).map_err(js_error)?),
      _ => unreachable!("validated format and unit"),
    };
    Ok(Self { inner })
  }

  #[wasm_bindgen(getter)]
  pub fn unit(&self) -> String {
    match self.inner {
      EncoderInner::Byte(_) => "byte",
      EncoderInner::Unicode(_) => "unicode",
    }.to_string()
  }

  #[wasm_bindgen(js_name = preTokenizer)]
  pub fn pre_tokenizer(&self) -> WasmPreTokenizer {
    let inner = match &self.inner {
      EncoderInner::Byte(encoder) => encoder.pre_tokenizer.clone(),
      EncoderInner::Unicode(encoder) => encoder.pre_tokenizer.clone(),
    };
    WasmPreTokenizer { inner }
  }

  #[wasm_bindgen(js_name = encodeWord)]
  pub fn encode_word(&self, word: &str) -> Result<Uint32Array, JsValue> {
    let ids = match &self.inner {
      EncoderInner::Byte(encoder) => encoder.encode_word(word),
      EncoderInner::Unicode(encoder) => encoder.encode_word(word),
    }.map_err(js_error)?;
    Ok(Uint32Array::from(ids.as_ref()))
  }

  #[wasm_bindgen(js_name = encodeWords)]
  pub fn encode_words(&self, words: Vec<String>) -> Result<JsValue, JsValue> {
    let words = words.iter().map(String::as_str).collect::<Vec<_>>();
    let encoded = match &self.inner {
      EncoderInner::Byte(encoder) => encoder.encode_words(&words),
      EncoderInner::Unicode(encoder) => encoder.encode_words(&words),
    }.map_err(js_error)?;
    let encoded = encoded.into_iter()
      .map(|ids| ids.iter().copied().collect::<Vec<_>>())
      .collect::<Vec<_>>();
    to_js(&encoded)
  }

  pub fn encode(&self, text: &str) -> Result<Uint32Array, JsValue> {
    let ids = match &self.inner {
      EncoderInner::Byte(encoder) => encoder.encode_string(text),
      EncoderInner::Unicode(encoder) => encoder.encode_string(text),
    }.map_err(js_error)?;
    Ok(Uint32Array::from(ids.as_slice()))
  }

  pub fn decode(&self, ids: Vec<Idx>) -> Result<String, JsValue> {
    match &self.inner {
      EncoderInner::Byte(encoder) => encoder.decode(&ids),
      EncoderInner::Unicode(encoder) => encoder.decode(&ids),
    }.map_err(js_error)
  }
}

#[wasm_bindgen]
pub struct WasmBpeModel {
  inner: ModelInner,
}

#[wasm_bindgen]
impl WasmBpeModel {
  #[wasm_bindgen(getter)]
  pub fn unit(&self) -> String {
    match self.inner {
      ModelInner::Byte(_) => "byte",
      ModelInner::Unicode(_) => "unicode",
    }.to_string()
  }

  #[wasm_bindgen(getter, js_name = lastMergeFreq)]
  pub fn last_merge_freq(&self) -> Result<JsValue, JsValue> {
    let frequency = match &self.inner {
      ModelInner::Byte(model) => model.last_merge_freq(),
      ModelInner::Unicode(model) => model.last_merge_freq(),
    };
    to_js(&frequency)
  }

  #[wasm_bindgen(getter, js_name = specialTokens)]
  pub fn special_tokens(&self) -> Vec<String> {
    match &self.inner {
      ModelInner::Byte(model) => model.special_tokens().to_vec(),
      ModelInner::Unicode(model) => model.special_tokens().to_vec(),
    }
  }

  #[wasm_bindgen(js_name = vocabItems)]
  pub fn vocab_items(&self) -> Result<JsValue, JsValue> {
    let items = match &self.inner {
      ModelInner::Byte(model) => vocab_items(model.vocab().iter()),
      ModelInner::Unicode(model) => vocab_items(model.vocab().iter()),
    };
    to_js(&items)
  }

  pub fn encoder(&self, options: JsValue) -> Result<WasmBpeEncoder, JsValue> {
    let options = if options.is_null() || options.is_undefined() {
      EncoderOptions::default()
    } else {
      from_js(options)?
    };
    let inner = match &self.inner {
      ModelInner::Byte(model) => EncoderInner::Byte(configure_encoder(
        model.to_encoder_with_options(
          options.pat_str.as_deref(), options.split_on_vocab_bigrams,
        ).map_err(js_error)?,
        options.unicode_bigrams.as_deref(),
        &options.unicode_bigram_mixed_boundary,
      ).map_err(js_error)?),
      ModelInner::Unicode(model) => EncoderInner::Unicode(configure_encoder(
        model.to_encoder_with_options(
          options.pat_str.as_deref(), options.split_on_vocab_bigrams,
        ).map_err(js_error)?,
        options.unicode_bigrams.as_deref(),
        &options.unicode_bigram_mixed_boundary,
      ).map_err(js_error)?),
    };
    Ok(WasmBpeEncoder { inner })
  }

  #[wasm_bindgen(js_name = serializeVocab)]
  pub fn serialize_vocab(&self, format: &str) -> Result<String, JsValue> {
    let mut output = Vec::new();
    match (&self.inner, format) {
      (ModelInner::Byte(model), "gpt2") => model.save_vocab_json(&Gpt2Spec, &mut output),
      (ModelInner::Byte(model), "unitoken") => model.save_vocab_json(&UnitokenSpec, &mut output),
      (ModelInner::Unicode(model), "unitoken") => model.save_vocab_json(&UnitokenSpec, &mut output),
      _ => Err(MyError::SpecError(format!(
        "format {format} is not compatible with unit {}", self.unit(),
      ))),
    }.map_err(js_error)?;
    String::from_utf8(output).map_err(js_error)
  }

  #[wasm_bindgen(js_name = serializeMerges)]
  pub fn serialize_merges(&self, format: &str) -> Result<String, JsValue> {
    let mut output = Vec::new();
    match (&self.inner, format) {
      (ModelInner::Byte(model), "gpt2") => model.save_merges_txt(&Gpt2Spec, &mut output),
      (ModelInner::Byte(model), "unitoken") => model.save_merges_txt(&UnitokenSpec, &mut output),
      (ModelInner::Unicode(model), "unitoken") => model.save_merges_txt(&UnitokenSpec, &mut output),
      _ => Err(MyError::SpecError(format!(
        "format {format} is not compatible with unit {}", self.unit(),
      ))),
    }.map_err(js_error)?;
    String::from_utf8(output).map_err(js_error)
  }
}
