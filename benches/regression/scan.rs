use std::{
  fs,
  hint::black_box,
  path::{Path, PathBuf},
  time::Instant,
};

use clap::Args as ClapArgs;
use fancy_regex::Regex;
use ffbpe::pretokenizer::PreTokenizer;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
  common::{
    environment::{default_suite_report_path, environment_report},
    fingerprint::{sha256_hex, to_hex},
    process::write_json_atomic,
    report::{EnvironmentReport, RunFailure, RunStatus},
    util::{
      duration_ms, duration_ns, now_seconds, resolve_path_for_comparison,
      throughput_mib,
    },
  },
  pretokenizer_patterns::{DATASETS, PATTERNS},
};

pub const CONTRACT: &str = "unitoken_pretokenizer_scan_regression_v1";
const SCHEMA_VERSION: u64 = 1;

#[derive(Clone, Debug, ClapArgs)]
pub struct Args {
  /// Number of paired dispatch/reference timing samples per pattern and corpus.
  #[arg(long, default_value_t = 5)]
  repeats: usize,
  /// Maximum UTF-8 input bytes read from each checked-in fixture.
  #[arg(long, default_value_t = 1024 * 1024)]
  max_bytes: usize,
  /// JSON report path. Defaults below out/benchmarks/regression/.
  #[arg(long)]
  output: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
struct ScanRequest {
  pattern_name: String,
  pattern: String,
  dataset_name: String,
  input_path: PathBuf,
  input_bytes: u64,
  input_sha256: String,
  sample_index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct TokenFingerprint {
  token_count: usize,
  token_bytes: u64,
  token_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct ScanMeasurement {
  dispatch_ns: u64,
  #[serde(skip_serializing_if = "Option::is_none")]
  scalar_ns: Option<u64>,
  fancy_regex_ns: u64,
  tokens: TokenFingerprint,
}

#[derive(Clone, Debug, Serialize)]
struct ScanSample {
  request: ScanRequest,
  status: RunStatus,
  measurement: Option<ScanMeasurement>,
  error: Option<RunFailure>,
}

#[derive(Clone, Debug, Serialize)]
struct ScanGates {
  passed: bool,
  failures: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct ScanReport {
  schema_version: u64,
  contract: String,
  generated_at_unix_seconds: u64,
  environment: EnvironmentReport,
  samples: Vec<ScanSample>,
  gates: ScanGates,
}

pub fn run(args: Args) -> Result<(), String> {
  if args.repeats == 0 {
    return Err("--repeats must be positive".to_string());
  }
  if args.max_bytes == 0 {
    return Err("--max-bytes must be positive".to_string());
  }

  let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
  let mut samples = Vec::new();
  let mut failures = Vec::new();

  for (pattern_name, pattern) in PATTERNS {
    let pretokenizer = PreTokenizer::try_new(&[], None, Some(pattern))
      .map_err(|error| format!("cannot configure {pattern_name}: {error}"))?;
    let reference = Regex::new(pattern)
      .map_err(|error| format!("cannot compile {pattern_name}: {error}"))?;

    for (dataset_name, relative_path) in DATASETS {
      let input_path = manifest_dir.join(relative_path);
      let input = fixture_prefix(&input_path, args.max_bytes)?;
      let input_sha256 = sha256_hex(input.as_bytes());
      let dispatch_fingerprint = fingerprint_dispatch(&pretokenizer, &input)?;
      let reference_fingerprint = fingerprint_reference(&reference, &input)?;

      if dispatch_fingerprint != reference_fingerprint {
        let message = format!(
          "{pattern_name}/{dataset_name} dispatch token boundaries differ from fancy-regex"
        );
        failures.push(message.clone());
        samples.push(ScanSample {
          request: request(
            pattern_name,
            pattern,
            dataset_name,
            &input_path,
            &input,
            &input_sha256,
            0,
          ),
          status: RunStatus::Failed,
          measurement: None,
          error: Some(RunFailure {
            phase: "correctness".to_string(),
            message,
          }),
        });
        continue;
      }

      #[cfg(feature = "benchmark-internals")]
      {
        let scalar_fingerprint =
          fingerprint_scalar(&pretokenizer, &input)?;
        if dispatch_fingerprint != scalar_fingerprint {
          let message = format!(
            "{pattern_name}/{dataset_name} native and scalar token boundaries differ"
          );
          failures.push(message.clone());
          samples.push(ScanSample {
            request: request(
              pattern_name,
              pattern,
              dataset_name,
              &input_path,
              &input,
              &input_sha256,
              0,
            ),
            status: RunStatus::Failed,
            measurement: None,
            error: Some(RunFailure {
              phase: "correctness".to_string(),
              message,
            }),
          });
          continue;
        }
      }

      time_dispatch(&pretokenizer, &input)?;
      #[cfg(feature = "benchmark-internals")]
      time_scalar(&pretokenizer, &input)?;
      time_reference(&reference, &input)?;

      for sample_index in 0..args.repeats {
        #[cfg(feature = "benchmark-internals")]
        let (dispatch_ns, scalar_ns, fancy_regex_ns) =
          match sample_index % 3 {
            0 => (
              time_dispatch(&pretokenizer, &input)?,
              time_scalar(&pretokenizer, &input)?,
              time_reference(&reference, &input)?,
            ),
            1 => {
              let scalar_ns = time_scalar(&pretokenizer, &input)?;
              let fancy_regex_ns =
                time_reference(&reference, &input)?;
              let dispatch_ns = time_dispatch(&pretokenizer, &input)?;
              (dispatch_ns, scalar_ns, fancy_regex_ns)
            }
            _ => {
              let fancy_regex_ns =
                time_reference(&reference, &input)?;
              let dispatch_ns = time_dispatch(&pretokenizer, &input)?;
              let scalar_ns = time_scalar(&pretokenizer, &input)?;
              (dispatch_ns, scalar_ns, fancy_regex_ns)
            }
          };
        #[cfg(not(feature = "benchmark-internals"))]
        let (dispatch_ns, fancy_regex_ns) = if sample_index % 2 == 0 {
          (
            time_dispatch(&pretokenizer, &input)?,
            time_reference(&reference, &input)?,
          )
        } else {
          let fancy_regex_ns = time_reference(&reference, &input)?;
          let dispatch_ns = time_dispatch(&pretokenizer, &input)?;
          (dispatch_ns, fancy_regex_ns)
        };
        #[cfg(not(feature = "benchmark-internals"))]
        let scalar_ns = None;
        #[cfg(feature = "benchmark-internals")]
        let scalar_ns = Some(scalar_ns);
        samples.push(ScanSample {
          request: request(
            pattern_name,
            pattern,
            dataset_name,
            &input_path,
            &input,
            &input_sha256,
            sample_index,
          ),
          status: RunStatus::Completed,
          measurement: Some(ScanMeasurement {
            dispatch_ns,
            scalar_ns,
            fancy_regex_ns,
            tokens: dispatch_fingerprint.clone(),
          }),
          error: None,
        });
      }
    }
  }

  let environment = environment_report();
  let report = ScanReport {
    schema_version: SCHEMA_VERSION,
    contract: CONTRACT.to_string(),
    generated_at_unix_seconds: now_seconds(),
    environment,
    samples,
    gates: ScanGates {
      passed: failures.is_empty(),
      failures,
    },
  };
  let output = args.output.unwrap_or_else(|| {
    default_suite_report_path("pretokenizer-scan", &report.environment)
  });
  validate_output_path(&output, manifest_dir)?;
  write_json_atomic(&output, &report)?;
  print_summary(&output, &report);

  if report.gates.passed {
    Ok(())
  } else {
    Err(format!(
      "pretokenizer scan correctness gates failed; inspect {}",
      output.display()
    ))
  }
}

fn validate_output_path(output: &Path, manifest_dir: &Path) -> Result<(), String> {
  let resolved_output = resolve_path_for_comparison(output)?;
  for (_, relative_path) in DATASETS {
    let input = resolve_path_for_comparison(&manifest_dir.join(relative_path))?;
    if resolved_output == input {
      return Err(format!(
        "report output cannot overwrite the input corpus {}",
        input.display()
      ));
    }
  }
  Ok(())
}

fn request(
  pattern_name: &str,
  pattern: &str,
  dataset_name: &str,
  input_path: &Path,
  input: &str,
  input_sha256: &str,
  sample_index: usize,
) -> ScanRequest {
  ScanRequest {
    pattern_name: pattern_name.to_string(),
    pattern: pattern.to_string(),
    dataset_name: dataset_name.to_string(),
    input_path: input_path.to_path_buf(),
    input_bytes: input.len() as u64,
    input_sha256: input_sha256.to_string(),
    sample_index,
  }
}

fn fixture_prefix(path: &Path, max_bytes: usize) -> Result<String, String> {
  let input = fs::read_to_string(path)
    .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
  let mut end = input.len().min(max_bytes);
  while !input.is_char_boundary(end) {
    end -= 1;
  }
  Ok(input[..end].to_string())
}

fn fingerprint_dispatch(
  pretokenizer: &PreTokenizer,
  input: &str,
) -> Result<TokenFingerprint, String> {
  let mut fingerprint = TokenFingerprintBuilder::new();
  pretokenizer
    .for_each_pretoken(input, |token| fingerprint.update(token))
    .map_err(|error| format!("dispatch scan failed: {error}"))?;
  Ok(fingerprint.finish())
}

fn fingerprint_reference(
  reference: &Regex,
  input: &str,
) -> Result<TokenFingerprint, String> {
  let mut fingerprint = TokenFingerprintBuilder::new();
  for found in reference.find_iter(input) {
    fingerprint.update(
      found
        .map_err(|error| format!("fancy-regex scan failed: {error}"))?
        .as_str(),
    );
  }
  Ok(fingerprint.finish())
}

#[cfg(feature = "benchmark-internals")]
fn fingerprint_scalar(
  pretokenizer: &PreTokenizer,
  input: &str,
) -> Result<TokenFingerprint, String> {
  let mut fingerprint = TokenFingerprintBuilder::new();
  pretokenizer
    .for_each_pretoken_scalar(input, |token| fingerprint.update(token))
    .map_err(|error| format!("scalar scan failed: {error}"))?;
  Ok(fingerprint.finish())
}

fn time_dispatch(pretokenizer: &PreTokenizer, input: &str) -> Result<u64, String> {
  let started = Instant::now();
  let mut token_count = 0_usize;
  let mut token_bytes = 0_usize;
  pretokenizer
    .for_each_pretoken(black_box(input), |token| {
      token_count += 1;
      token_bytes += token.len();
    })
    .map_err(|error| format!("dispatch scan failed: {error}"))?;
  black_box((token_count, token_bytes));
  Ok(duration_ns(started.elapsed()))
}

fn time_reference(reference: &Regex, input: &str) -> Result<u64, String> {
  let started = Instant::now();
  let mut token_count = 0_usize;
  let mut token_bytes = 0_usize;
  for found in reference.find_iter(black_box(input)) {
    let token = found
      .map_err(|error| format!("fancy-regex scan failed: {error}"))?
      .as_str();
    token_count += 1;
    token_bytes += token.len();
  }
  black_box((token_count, token_bytes));
  Ok(duration_ns(started.elapsed()))
}

#[cfg(feature = "benchmark-internals")]
fn time_scalar(
  pretokenizer: &PreTokenizer,
  input: &str,
) -> Result<u64, String> {
  let started = Instant::now();
  let mut token_count = 0_usize;
  let mut token_bytes = 0_usize;
  pretokenizer
    .for_each_pretoken_scalar(black_box(input), |token| {
      token_count += 1;
      token_bytes += token.len();
    })
    .map_err(|error| format!("scalar scan failed: {error}"))?;
  black_box((token_count, token_bytes));
  Ok(duration_ns(started.elapsed()))
}

struct TokenFingerprintBuilder {
  digest: Sha256,
  token_count: usize,
  token_bytes: u64,
}

impl TokenFingerprintBuilder {
  fn new() -> Self {
    Self {
      digest: Sha256::new(),
      token_count: 0,
      token_bytes: 0,
    }
  }

  fn update(&mut self, token: &str) {
    self.digest.update((token.len() as u64).to_le_bytes());
    self.digest.update(token.as_bytes());
    self.token_count += 1;
    self.token_bytes += token.len() as u64;
  }

  fn finish(self) -> TokenFingerprint {
    TokenFingerprint {
      token_count: self.token_count,
      token_bytes: self.token_bytes,
      token_sha256: to_hex(&self.digest.finalize()),
    }
  }
}

fn print_summary(path: &Path, report: &ScanReport) {
  println!("pretokenizer scan report: {}", path.display());
  for (pattern_name, _) in PATTERNS {
    for (dataset_name, _) in DATASETS {
      let matching = report.samples.iter().filter(|sample| {
        sample.request.pattern_name == pattern_name
          && sample.request.dataset_name == dataset_name
      });
      let input_bytes = matching.clone().next().map(|sample| sample.request.input_bytes);
      let mut dispatch = matching
        .clone()
        .filter_map(|sample| sample.measurement.as_ref())
        .map(|measurement| measurement.dispatch_ns)
        .collect::<Vec<_>>();
      let mut fancy_regex = matching
        .clone()
        .filter_map(|sample| sample.measurement.as_ref())
        .map(|measurement| measurement.fancy_regex_ns)
        .collect::<Vec<_>>();
      let mut scalar = matching
        .filter_map(|sample| sample.measurement.as_ref())
        .filter_map(|measurement| measurement.scalar_ns)
        .collect::<Vec<_>>();
      let (Some(input_bytes), Some(dispatch_ns), Some(fancy_regex_ns)) = (
        input_bytes,
        median(&mut dispatch),
        median(&mut fancy_regex),
      ) else {
        println!("  {pattern_name}/{dataset_name} failed");
        continue;
      };
      println!(
        "  {pattern_name}/{dataset_name} median dispatch={:.3} ms ({:.1} MiB/s) fancy-regex={:.3} ms ({:.1} MiB/s)",
        duration_ms(dispatch_ns),
        throughput_mib(input_bytes, dispatch_ns),
        duration_ms(fancy_regex_ns),
        throughput_mib(input_bytes, fancy_regex_ns),
      );
      if let Some(scalar_ns) = median(&mut scalar) {
        println!(
          "    forced scalar={:.3} ms ({:.1} MiB/s), native speedup={:.2}x",
          duration_ms(scalar_ns),
          throughput_mib(input_bytes, scalar_ns),
          scalar_ns as f64 / dispatch_ns as f64,
        );
      }
    }
  }
}

fn median(values: &mut [u64]) -> Option<u64> {
  if values.is_empty() {
    return None;
  }
  values.sort_unstable();
  let middle = values.len() / 2;
  if values.len() % 2 == 0 {
    Some(values[middle - 1].saturating_add(values[middle]) / 2)
  } else {
    Some(values[middle])
  }
}
