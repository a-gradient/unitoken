#[macro_use]
extern crate tracing;

use clap::{Parser, Subcommand};
use std::{
  collections::BTreeMap, fs, path::{Path, PathBuf}
};

use unitoken::{
  bpe::BpeTrainer,
  pretokenizer::{get_words_from_file, save_words, sort_words},
};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
  #[command(subcommand)]
  command: Commands,
  #[arg(short, long, action = clap::ArgAction::Count)]
  verbose: u8,
}

#[derive(Subcommand)]
enum Commands {
  Train(TrainArgs),
  Encode(EncodeArgs),
}

impl Commands {
  fn verbose(&self) -> u8 {
    match self {
      Commands::Train(args) => args.verbose,
      Commands::Encode(args) => args.verbose,
    }
  }
}

#[derive(Parser)]
struct TrainArgs {
  #[arg(short, long, action = clap::ArgAction::Count)]
  verbose: u8,
  #[arg(short, long = "out", default_value = "out")]
  out_dir: PathBuf,
  #[arg(short='s', long, default_value = "10000")]
  vocab_size: u32,
  #[arg(short = 'c', long = "chunks", default_value = "1024")]
  num_chunks: u32,
  #[arg(long = "special-tokens")]
  special_tokens_path: Option<PathBuf>,
  #[arg(value_parser = clap::value_parser!(PathBuf))]
  input_file: PathBuf,
}

#[derive(Parser)]
struct EncodeArgs {
  #[arg(short, long, action = clap::ArgAction::Count)]
  verbose: u8,
  #[arg(short, long = "out", default_value = "out")]
  out_dir: PathBuf,
  #[arg(value_parser = clap::value_parser!(PathBuf))]
  input_file: PathBuf,
}

fn _pretokenize<P1: AsRef<Path>, P2: AsRef<Path>>(output: P1, input: P2, num_chunks: u32, special_tokens: Vec<String>) -> BTreeMap<String, i64> {
  if output.as_ref().exists() {
    info!("pretokenize file already exists, loading from {}", output.as_ref().display());
    let result = serde_json::from_reader(fs::File::open(output).expect("open _words file")).expect("read _words file");
    return result;
  }
  let split_special_token = special_tokens.get(0).cloned();

  let words = get_words_from_file(&input, num_chunks, special_tokens, split_special_token.as_deref()).unwrap();

  let words_file = fs::File::create(output).unwrap();
  save_words(words_file, &sort_words(&words)).unwrap();
  words
}

fn train_bpe<P: AsRef<Path>>(
  path: P, vocab_size: u32, num_chunks: u32, special_tokens: &Vec<String>, out_dir: &PathBuf,
) {
  fs::create_dir_all(out_dir).expect("Failed to create output directory");

  let file_stem = path
    .as_ref()
    .file_stem()
    .expect("Failed to get file stem")
    .to_str()
    .expect("Failed to convert file stem to str");
  // use first special_token as split_special_token

  let words = _pretokenize(
    out_dir.join(format!("_words.{file_stem}.json")),
    &path,
    num_chunks,
    special_tokens.clone(),
  );

  let mut bpe = BpeTrainer::from_words(words, special_tokens);
  let start_vocab_idx = bpe.start_vocab_idx.load(std::sync::atomic::Ordering::Acquire) as usize;
  bpe.init_training();
  for i in start_vocab_idx..vocab_size as usize {
    if bpe.step().is_none() {
      warn!(vocab_size=i, "No more merges can be made, stopping training early");
      break;
    }
  }

  let vocab_filename = format!("vocab.{file_stem}.json");
  let merges_filename = format!("merges.{file_stem}.txt");
  let vocab_file = fs::File::create(out_dir.join(vocab_filename)).unwrap();
  let merges_file = fs::File::create(out_dir.join(merges_filename)).unwrap();

  bpe.save_vocab_json(vocab_file).unwrap();
  bpe.save_merges_txt(merges_file).unwrap();
}

fn lines_of(s: &str) -> Vec<String> {
  s.lines().filter(|line| !line.is_empty()).map(|line| line.to_string()).collect()
}

fn run_train(args: TrainArgs) {
  let special_tokens = if let Some(special_tokens_path) = args.special_tokens_path {
    let content = fs::read_to_string(special_tokens_path).expect("Failed to read special tokens file");
    lines_of(&content)
  } else {
    lines_of(include_str!("../fixtures/default_special_tokens.txt"))
  };
  debug!("Special tokens: {:?}", special_tokens);
  debug!("Vocabulary size: {}", args.vocab_size);
  debug!("Number of chunks: {}", args.num_chunks);
  debug!("Input file: {}", args.input_file.display());
  debug!("Output directory: {}", args.out_dir.display());
  info!("Training BPE model...");
  train_bpe(
    args.input_file,
    args.vocab_size,
    args.num_chunks,
    &special_tokens,
    &args.out_dir,
  );
}

fn main() {
  let cli = Cli::parse();
  let verbose = cli.verbose + cli.command.verbose();
  match verbose {
    0 => tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init(),
    1 => tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).init(),
    _ => tracing_subscriber::fmt().with_max_level(tracing::Level::TRACE).init(),
  }
  debug!("Verbosity level: {}", verbose);
  match cli.command {
    Commands::Train(train_args) => {
      run_train(train_args);
    }
    Commands::Encode(_encode_args) => {
      unimplemented!("Encode command is not implemented yet");
    }
  }
}
