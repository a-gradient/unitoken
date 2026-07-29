use std::fs;

use criterion::{
  black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use fancy_regex::Regex;

use ffbpe::{
  bpe::{encoder::BpeBuilder, Idx},
  pretokenizer::PreTokenizer,
  spec::gpt2::Gpt2Spec,
  traits::Encode as _,
};

mod pretokenizer_patterns;

use pretokenizer_patterns::{DATASETS, PATTERNS};

fn build_gpt2_encoder_from_fixtures(name: &str) -> ffbpe::bpe::BpeEncoder<u8> {
  BpeBuilder::new()
    .load_merges_file(format!("fixtures/merges.{name}.txt"), &Gpt2Spec)
    .unwrap()
    .load_vocab_file(format!("fixtures/vocab.{name}.json"), &Gpt2Spec)
    .unwrap()
    .build(&Gpt2Spec)
    .unwrap()
}

fn bench_pretokenizer(c: &mut Criterion) {
  let special_tokens = vec![ffbpe::pretokenizer::DEFAULT_EOT.to_string()];
  let pre = PreTokenizer::new(&special_tokens, Some(ffbpe::pretokenizer::DEFAULT_EOT));

  let base = "Once upon a time, in a small village, there lived a cat named Mango.";
  let input = base.repeat(200);

  c.bench_function("pretokenizer/get_words", |b| {
    b.iter(|| {
      let words = pre.get_words(black_box(&input)).unwrap();
      black_box(words.len())
    })
  });
}

fn bench_pretokenizer_scan(c: &mut Criterion) {
  let datasets = DATASETS.map(|(name, path)| (name, fixture_prefix(path, 1 << 20)));

  for (pattern_name, pattern) in PATTERNS {
    let pretokenizer = PreTokenizer::try_new(&[], None, Some(pattern)).unwrap();
    let reference = Regex::new(pattern).unwrap();
    let mut group = c.benchmark_group(format!("pretokenizer/scan/{pattern_name}"));
    for (dataset_name, input) in &datasets {
      group.throughput(Throughput::Bytes(input.len() as u64));
      group.bench_with_input(
        BenchmarkId::new("dispatch", dataset_name),
        input,
        |b, input| {
          b.iter(|| {
            let mut count = 0_usize;
            let mut bytes = 0_usize;
            pretokenizer
              .for_each_pretoken(black_box(input), |token| {
                count += 1;
                bytes += token.len();
              })
              .unwrap();
            black_box((count, bytes))
          })
        },
      );
      group.bench_with_input(
        BenchmarkId::new("fancy_regex", dataset_name),
        input,
        |b, input| {
          b.iter(|| {
            let mut count = 0_usize;
            let mut bytes = 0_usize;
            for found in reference.find_iter(black_box(input)) {
              let token = found.unwrap().as_str();
              count += 1;
              bytes += token.len();
            }
            black_box((count, bytes))
          })
        },
      );
    }
    group.finish();
  }
}

fn fixture_prefix(path: &str, max_bytes: usize) -> String {
  let input = fs::read_to_string(path).unwrap();
  let mut end = input.len().min(max_bytes);
  while !input.is_char_boundary(end) {
    end -= 1;
  }
  input[..end].to_string()
}

fn bench_bpe_encode_decode(c: &mut Criterion) {
  const FIXTURE: &str = "tinystories_sample_5M";
  let bpe = build_gpt2_encoder_from_fixtures(FIXTURE);

  let base = "Once upon a time, there was a little robot who loved to read books.";
  let input = base.repeat(200);

  let mut group = c.benchmark_group("bpe");

  group.bench_with_input(BenchmarkId::new("encode_string", FIXTURE), &input, |b, s| {
    b.iter(|| {
      let out = bpe.encode_string(black_box(s)).unwrap();
      black_box(out)
    })
  });

  let encoded: Vec<Idx> = bpe.encode_string(&input).unwrap();
  group.bench_with_input(BenchmarkId::new("decode", FIXTURE), &encoded, |b, ids| {
    b.iter(|| {
      let out = bpe.decode(black_box(ids)).unwrap();
      black_box(out)
    })
  });

  group.finish();
}

criterion_group!(
  benches,
  bench_pretokenizer,
  bench_pretokenizer_scan,
  bench_bpe_encode_decode,
);
criterion_main!(benches);
