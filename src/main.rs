use clap::{Parser, arg};
use std::{
  fs,
  path::{Path, PathBuf},
};

use unitoken::{
  bpe::BpeTrainer,
  pretokenizer::{get_words_from_file, save_words, sort_words},
};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
  #[arg(long = "special-tokens")]
  special_tokens_path: Option<PathBuf>,
  #[arg(short = 'c', long = "chunks", default_value = "1024")]
  num_chunks: u32,
  #[arg(short, long = "out", default_value = "out")]
  out_dir: PathBuf,
  #[arg(short='s', long, default_value = "10000")]
  vocab_size: u32,
  #[arg(value_parser = clap::value_parser!(PathBuf))]
  input_file: PathBuf,
}

fn train_bpe<P: AsRef<Path>>(
  path: P, vocab_size: u32, num_chunks: u32, special_tokens: &Vec<String>, out_dir: &PathBuf,
) {
  let file_stem = path
    .as_ref()
    .file_stem()
    .expect("Failed to get file stem")
    .to_str()
    .expect("Failed to convert file stem to str");
  // use first special_token as split_special_token
  let split_special_token = special_tokens.get(0).map(String::as_str);
  let words = get_words_from_file(&path, num_chunks, special_tokens, split_special_token).unwrap();
  let words_sorted = sort_words(&words);

  let mut bpe = BpeTrainer::from_words(words, special_tokens);
  let start_vocab_idx = bpe.start_vocab_idx.load(std::sync::atomic::Ordering::Acquire) as usize;
  bpe.init_training();
  for _ in start_vocab_idx..vocab_size as usize {
    bpe.step();
  }

  fs::create_dir_all(out_dir).expect("Failed to create output directory");
  let vocab_filename = format!("vocab.{file_stem}.json");
  let merges_filename = format!("merges.{file_stem}.txt");
  let words_filename = format!("_words.{file_stem}.json");
  let mut open_options = fs::OpenOptions::new();
  open_options.write(true).create(true).truncate(true);
  let vocab_file = open_options.open(out_dir.join(vocab_filename)).unwrap();
  let merges_file = open_options.open(out_dir.join(merges_filename)).unwrap();
  let words_file = open_options.open(out_dir.join(words_filename)).unwrap();

  bpe.save_vocab_json(&vocab_file).unwrap();
  bpe.save_merges_txt(merges_file).unwrap();
  save_words(words_file, &words_sorted).unwrap();
}

fn lines_of(s: &str) -> Vec<String> {
  s.lines().filter(|line| !line.is_empty()).map(|line| line.to_string()).collect()
}

fn main() {
  let cli = Cli::parse();
  let special_tokens: Vec<String> = match &cli.special_tokens_path {
    Some(path) => {
      let content = fs::read_to_string(path).expect("Failed to read special tokens file");
      lines_of(&content)
    }
    None => lines_of(include_str!("../fixtures/default_special_tokens.txt")),
  };
  train_bpe(
    &cli.input_file,
    cli.vocab_size,
    cli.num_chunks,
    &special_tokens,
    &cli.out_dir,
  );
}
