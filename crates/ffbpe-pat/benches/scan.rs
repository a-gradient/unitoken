use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use fancy_regex::Regex;
use ffbpe_pat::Pattern;

fn bench_scan(c: &mut Criterion) {
  let datasets = [
    (
      "english",
      "Once upon a time, a small tokenizer split this sentence. It's 2026!\n"
        .repeat(16_384),
    ),
    (
      "chinese",
      "从前有一个小型分词器，它会处理中文、数字123和标点。\n".repeat(16_384),
    ),
    (
      "mixed",
      "Hello 世界! 한글かな العربية १२३ 👩‍💻\r\n".repeat(16_384),
    ),
    (
      "long_ascii",
      format!("{} {} ", "a".repeat(256), "Z".repeat(256)).repeat(2_048),
    ),
    (
      "code",
      "  fn parseHTTPResponse(input: &str) -> Result<()> {\n    let camelCase = 123456; // comment\n  }\n"
        .repeat(16_384),
    ),
    (
      "case_mix",
      "ABC中文DEF A\u{301}BCdef a中文B HTTPResponse can'ts 'ſtail\r\n\t  "
        .repeat(16_384),
    ),
  ];

  for pattern in Pattern::ALL {
    let regex = Regex::new(pattern.regex()).unwrap();
    let mut group = c.benchmark_group(format!("scan/{pattern:?}"));
    for (dataset_name, input) in &datasets {
      group.throughput(Throughput::Bytes(input.len() as u64));
      group.bench_with_input(
        BenchmarkId::new("specialized", dataset_name),
        input,
        |b, input| {
          b.iter(|| {
            let mut token_count = 0_usize;
            let mut token_bytes = 0_usize;
            for token in pattern.split(black_box(input)) {
              token_count += 1;
              token_bytes += token.len();
            }
            black_box((token_count, token_bytes))
          })
        },
      );
      group.bench_with_input(
        BenchmarkId::new("fancy_regex", dataset_name),
        input,
        |b, input| {
          b.iter(|| {
            let mut token_count = 0_usize;
            let mut token_bytes = 0_usize;
            for found in regex.find_iter(black_box(input)) {
              let token = found.unwrap().as_str();
              token_count += 1;
              token_bytes += token.len();
            }
            black_box((token_count, token_bytes))
          })
        },
      );
    }
    group.finish();
  }
}

criterion_group!(benches, bench_scan);
criterion_main!(benches);
