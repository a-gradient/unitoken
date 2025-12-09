use std::{collections::HashMap, sync::{Arc, Mutex, OnceLock}};

use metrics::{Counter, CounterFn, Gauge, GaugeFn, Histogram, HistogramFn, Recorder, SetRecorderError};

pub struct Frame<T> {
  value: T,
  timestamp: std::time::Instant,
}

impl<T> Frame<T> {
  pub fn new(value: T) -> Self {
    Self {
      value,
      timestamp: std::time::Instant::now(),
    }
  }
}

#[derive(Default)]
pub struct Block<T> {
  frames: Vec<Frame<T>>,
  current: T,
}

pub struct GlobalStore {
  started_at: std::time::Instant,
  counters: Mutex<HashMap<String, Block<u64>>>,
  gauges: Mutex<HashMap<String, Block<f64>>>,
  histograms: Mutex<HashMap<String, Block<f64>>>,
}

static GLOBAL_STORE: OnceLock<Arc<GlobalStore>> = OnceLock::new();

fn global_store() -> &'static Arc<GlobalStore> {
  GLOBAL_STORE.get_or_init(|| {
    Arc::new(GlobalStore {
      started_at: std::time::Instant::now(),
      counters: Mutex::new(HashMap::new()),
      gauges: Mutex::new(HashMap::new()),
      histograms: Mutex::new(HashMap::new()),
    })
  })
}

struct CounterHandle {
  key: String,
  store: Arc<GlobalStore>,
}

struct GaugeHandle {
  key: String,
  store: Arc<GlobalStore>,
}

struct HistogramHandle {
  key: String,
  store: Arc<GlobalStore>,
}


impl CounterFn for CounterHandle {
  fn increment(&self, value: u64) {
    let mut map = self.store.counters.lock().unwrap();
    let entry = map.entry(self.key.clone()).or_default();
    entry.current += value;
    entry.frames.push(Frame::new(entry.current));
  }

  fn absolute(&self, value: u64) {
    let mut map = self.store.counters.lock().unwrap();
    let entry = map.entry(self.key.clone()).or_default();
    entry.current = value;
    entry.frames.push(Frame::new(entry.current));
  }
}

impl GaugeFn for GaugeHandle {
  fn increment(&self, value: f64) {
    let mut map = self.store.gauges.lock().unwrap();
    let entry = map.entry(self.key.clone()).or_default();
    entry.current += value;
    entry.frames.push(Frame::new(entry.current));
  }

  fn decrement(&self, value: f64) {
    let mut map = self.store.gauges.lock().unwrap();
    let entry = map.entry(self.key.clone()).or_default();
    entry.current -= value;
    entry.frames.push(Frame::new(entry.current));
  }

  fn set(&self, value: f64) {
    let mut map = self.store.gauges.lock().unwrap();
    let entry = map.entry(self.key.clone()).or_default();
    entry.current = value;
    entry.frames.push(Frame::new(entry.current));
  }
}

impl HistogramFn for HistogramHandle {
  fn record(&self, value: f64) {
    let mut map = self.store.histograms.lock().unwrap();
    let entry = map.entry(self.key.clone()).or_default();
    entry.current = value;
    entry.frames.push(Frame::new(entry.current));
  }
}

pub struct MetricsRecorder;
impl Recorder for MetricsRecorder {
  fn describe_counter(&self, _key: metrics::KeyName, _unit: Option<metrics::Unit>, _description: metrics::SharedString) { }

  fn describe_gauge(&self, _key: metrics::KeyName, _unit: Option<metrics::Unit>, _description: metrics::SharedString) { }

  fn describe_histogram(&self, _key: metrics::KeyName, _unit: Option<metrics::Unit>, _description: metrics::SharedString) { }

  fn register_counter(&self, key: &metrics::Key, _metadata: &metrics::Metadata<'_>) -> Counter {
    let store = global_store().clone();
    let handle = CounterHandle {
      key: key.name().to_string(),
      store,
    };
    Counter::from_arc(Arc::new(handle))
  }

  fn register_gauge(&self, key: &metrics::Key, _metadata: &metrics::Metadata<'_>) -> Gauge {
    let store = global_store().clone();
    let handle = GaugeHandle {
      key: key.name().to_string(),
      store,
    };
    Gauge::from_arc(Arc::new(handle))
  }

  fn register_histogram(&self, key: &metrics::Key, _metadata: &metrics::Metadata<'_>) -> Histogram {
    let store = global_store().clone();
    let handle = HistogramHandle {
      key: key.name().to_string(),
      store,
    };
    Histogram::from_arc(Arc::new(handle))
  }
}

pub fn init_metrics() -> Result<(), SetRecorderError<MetricsRecorder>> {
  let recorder = MetricsRecorder;
  metrics::set_global_recorder(recorder)?;
  Ok(())
}
