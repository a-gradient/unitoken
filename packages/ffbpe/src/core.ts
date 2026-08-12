import { modelOptions, resolveFormat, validateModelConfig } from "./config.js"
import type {
  BpeEncoderOptions,
  BpeTrainerOptions,
  FileFormat,
  HotPairWindowStats,
  ModelConfig,
  PretrainedFiles,
  PreTokenSpan,
  PreTokenizerOptions,
  SourceBatchOptions,
  TextFile,
  TrainerMemoryUsage,
  Runtime,
  TrainBpeOptions,
  UnicodeBigramSelection,
  Unit,
  VocabularyEntry,
  WordEntry,
  WordFrequencies,
} from "./types.js"
import { initWasm, wasm } from "./wasm.js"
import type {
  RawBigramCounter,
  RawBpeEncoder,
  RawBpeModel,
  RawBpeTrainer,
  RawPreTokenizer,
  RawWordCounter,
} from "./wasm-types.js"

let active_runtime: Runtime | undefined
const raw_word_counters = new WeakMap<WordCounter, RawWordCounter>()

export class FFBPE {
  private constructor() {}

  static async init(): Promise<void> {
    await initWasm(await runtime().wasm_input())
  }

  /** @internal */
  static configureRuntime(runtime: Runtime): void {
    active_runtime = runtime
  }
}

function runtime(): Runtime {
  if (active_runtime === undefined) {
    throw new Error("No FFBPE runtime configured. Import `@tokn-ai/ffbpe/browser` or `@tokn-ai/ffbpe/node`.")
  }
  return active_runtime
}

function preTokenizerOptions(options: PreTokenizerOptions): object {
  return {
    eot_token: options.eot_token ?? null,
    pat_str: options.pat_str ?? null,
    unicode_bigrams: options.unicode_bigrams === null || options.unicode_bigrams === undefined
      ? null
      : [...options.unicode_bigrams],
    unicode_bigram_mixed_boundary: options.unicode_bigram_mixed_boundary ?? "keep",
  }
}

function encoderOptions(options: BpeEncoderOptions): object {
  const unit = options.unit ?? "byte"
  return {
    ...preTokenizerOptions(options),
    unit,
    format: resolveFormat(unit, options.format),
    special_tokens: options.special_tokens === undefined ? null : [...options.special_tokens],
    split_on_vocab_bigrams: options.split_on_vocab_bigrams ?? true,
  }
}

function normalizeWordEntries(words: WordFrequencies | readonly WordEntry[]): Array<[string, number]> {
  const entries = Array.isArray(words) ? words : Object.entries(words)
  return entries.map(([word, frequency]) => {
    safeInteger(frequency, `frequency for ${JSON.stringify(word)}`)
    return [word, frequency]
  })
}

function normalizeVocabulary(vocab: ReadonlyMap<Uint8Array, number> | readonly VocabularyEntry[]): Array<[number[], number]> {
  const entries = vocab instanceof Map ? [...vocab] : [...vocab]
  return entries.map(([token, id]) => {
    unsignedInteger(id, "vocabulary id", 0xffff_ffff)
    return [[...token], id]
  })
}

function normalizeMerges(
  merges: readonly (readonly [Uint8Array, Uint8Array])[],
): Array<[number[], number[]]> {
  return merges.map(([left, right]) => [[...left], [...right]])
}

function vocabularyItems(items: Array<[number[], number]>): VocabularyEntry[] {
  return items.map(([token, id]) => [Uint8Array.from(token), id])
}

function nullableFrequency(value: unknown): number | null {
  if (value === null || value === undefined) return null
  if (typeof value !== "number") throw new TypeError("Expected a numeric merge frequency")
  return value
}

function safeInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value)) throw new RangeError(`${name} must be a safe integer`)
}

function unsignedInteger(value: number, name: string, maximum = Number.MAX_SAFE_INTEGER): void {
  safeInteger(value, name)
  if (value < 0 || value > maximum) {
    throw new RangeError(`${name} must be between 0 and ${maximum}`)
  }
}

function positiveInteger(value: number, name: string): void {
  unsignedInteger(value, name)
  if (value === 0) throw new RangeError(`${name} must be positive`)
}

const text_encoder = new TextEncoder()

function addSourceBatches(
  source: Iterable<string>,
  options: SourceBatchOptions,
  add_batch: (batch: string[]) => void,
): void {
  const max_records = options.max_records ?? 4096
  const max_bytes = options.max_bytes ?? 64 * 1024 * 1024
  positiveInteger(max_records, "max_records")
  positiveInteger(max_bytes, "max_bytes")

  let batch: string[] = []
  let batch_bytes = 0
  for (const text of source) {
    if (typeof text !== "string") throw new TypeError("source must yield strings")
    const text_bytes = text_encoder.encode(text).byteLength
    if (batch.length > 0 && (
      batch.length >= max_records
      || batch_bytes + text_bytes > max_bytes
    )) {
      add_batch(batch)
      batch = []
      batch_bytes = 0
    }
    batch.push(text)
    batch_bytes += text_bytes
  }
  if (batch.length > 0) add_batch(batch)
}

export class PreTokenizer {
  private inner: RawPreTokenizer

  constructor(special_tokens: readonly string[] = [], options: PreTokenizerOptions = {}) {
    const WasmPreTokenizer = wasm().WasmPreTokenizer
    this.inner = new WasmPreTokenizer([...special_tokens], preTokenizerOptions(options))
  }

  /** @internal */
  static fromRaw(inner: RawPreTokenizer): PreTokenizer {
    const result = Object.create(PreTokenizer.prototype) as { inner: RawPreTokenizer }
    result.inner = inner
    return result as unknown as PreTokenizer
  }

  withUnicodeBigrams(bigrams: readonly string[]): PreTokenizer {
    return PreTokenizer.fromRaw(this.inner.withUnicodeBigrams([...bigrams]))
  }

  getWords(text: string): WordFrequencies {
    return this.inner.getWords(text)
  }

  /** Return ordered logical pretokens with UTF-8 byte offsets. */
  split(text: string): PreTokenSpan[] {
    return this.inner.split(text)
  }

  bigramCounter(): BigramCounter {
    return new BigramCounter(this)
  }

  wordCounter(): WordCounter {
    return new WordCounter(this)
  }

  loadWordCounterData(serialized: string): WordCounter {
    return WordCounter.fromRaw(this.inner.loadWordCounter(serialized))
  }

  async loadWordCounter(file: string | URL | Blob): Promise<WordCounter> {
    return this.loadWordCounterData(await readTextFile(file))
  }

  async getWordsFromFile(file: string | URL | Blob): Promise<WordFrequencies> {
    return this.getWords(await readTextFile(file))
  }

  async buildUnicodeBigramsFromFile(
    file: TextFile,
    top_k = 100_000,
    min_freq = 16,
  ): Promise<string[]> {
    return (await this.selectUnicodeBigramsFromFile(file, top_k, min_freq)).bigrams
  }

  async selectUnicodeBigramsFromFile(
    file: TextFile,
    top_k = 100_000,
    min_freq = 16,
  ): Promise<UnicodeBigramSelection> {
    const counter = this.bigramCounter()
    counter.addText(await readTextFile(file))
    return counter.select(top_k, min_freq)
  }

  /** @internal */
  raw(): RawPreTokenizer {
    return this.inner
  }
}

export class BigramCounter {
  private readonly inner: RawBigramCounter

  constructor(pre_tokenizer: PreTokenizer) {
    this.inner = pre_tokenizer.raw().bigramCounter()
  }

  addText(text: string): void {
    this.inner.addText(text)
  }

  addBatch(texts: readonly string[]): void {
    this.inner.addBatch([...texts])
  }

  addSource(source: Iterable<string>, options: SourceBatchOptions = {}): void {
    addSourceBatches(source, options, batch => this.inner.addBatch(batch))
  }

  merge(other: BigramCounter): void {
    this.inner.merge(other.inner)
  }

  selected(top_k: number, min_freq: number): string[] {
    unsignedInteger(top_k, "top_k")
    safeInteger(min_freq, "min_freq")
    return this.inner.selected(top_k, min_freq)
  }

  select(top_k: number, min_freq: number): UnicodeBigramSelection {
    unsignedInteger(top_k, "top_k")
    safeInteger(min_freq, "min_freq")
    return this.inner.select(top_k, min_freq) as UnicodeBigramSelection
  }

  items(): Array<[string, number]> {
    return this.inner.items()
  }
}

export class WordCounter {
  constructor(pre_tokenizer: PreTokenizer) {
    raw_word_counters.set(this, pre_tokenizer.raw().wordCounter())
  }

  /** @internal */
  static fromRaw(inner: RawWordCounter): WordCounter {
    const result = Object.create(WordCounter.prototype) as WordCounter
    raw_word_counters.set(result, inner)
    return result
  }

  private get inner(): RawWordCounter {
    const inner = raw_word_counters.get(this)
    if (inner === undefined) throw new Error("Invalid WordCounter")
    return inner
  }

  addText(text: string): void {
    this.inner.addText(text)
  }

  addBatch(texts: readonly string[]): void {
    this.inner.addBatch([...texts])
  }

  addSource(source: Iterable<string>, options: SourceBatchOptions = {}): void {
    addSourceBatches(source, options, batch => this.inner.addBatch(batch))
  }

  merge(other: WordCounter): void {
    this.inner.merge(other.inner)
  }

  words(): WordFrequencies {
    return this.inner.words()
  }

  get len(): number {
    return this.inner.len
  }

  get isEmpty(): boolean {
    return this.inner.isEmpty
  }

  clear(): void {
    this.inner.clear()
  }

  serialize(): string {
    return this.inner.serialize()
  }

  async save(file: string | URL): Promise<void> {
    await writeTextFile(file, this.serialize(), "WordCounter.save")
  }
}

export class BpeTrainer {
  readonly #inner: RawBpeTrainer

  constructor(special_tokens: readonly string[], options: BpeTrainerOptions = {}) {
    if (options.parallel_merge_min_occurs_in != null) {
      unsignedInteger(options.parallel_merge_min_occurs_in, "parallel_merge_min_occurs_in")
    }
    if (options.hot_pair_window_size != null) {
      positiveInteger(options.hot_pair_window_size, "hot_pair_window_size")
    }
    if (options.bigram_cutoff_freq != null) {
      positiveInteger(options.bigram_cutoff_freq, "bigram_cutoff_freq")
    }
    const WasmBpeTrainer = wasm().WasmBpeTrainer
    this.#inner = new WasmBpeTrainer([...special_tokens], {
      unit: options.unit ?? "byte",
      initial_alphabet: options.initial_alphabet ?? "raw",
      tie_break: options.tie_break ?? "smallest_pair_id",
      parallel_merge_min_occurs_in: options.parallel_merge_min_occurs_in ?? null,
      hot_pair_window_size: options.hot_pair_window_size ?? null,
      bigram_cutoff_freq: options.bigram_cutoff_freq ?? null,
    })
  }

  get unit(): Unit {
    return this.#inner.unit as Unit
  }

  get vocabSize(): number {
    return this.#inner.vocabSize
  }

  get vocab(): VocabularyEntry[] {
    return vocabularyItems(this.#inner.vocabItems())
  }

  get lastMergeFreq(): number | null {
    return nullableFrequency(this.#inner.lastMergeFreq)
  }

  get hotPairWindowStats(): HotPairWindowStats | null {
    return this.#inner.hotPairWindowStats as HotPairWindowStats | null
  }

  get memoryUsage(): TrainerMemoryUsage {
    return this.#inner.memoryUsage as TrainerMemoryUsage
  }

  addWords(words: WordFrequencies | readonly WordEntry[]): void {
    this.#inner.addWords(normalizeWordEntries(words))
  }

  addWordCounter(counter: WordCounter): void {
    const inner = raw_word_counters.get(counter)
    if (inner === undefined) throw new Error("Invalid WordCounter")
    this.#inner.addWordCounter(inner)
  }

  initTraining(): void {
    this.#inner.initTraining()
  }

  train(vocab_size: number): void {
    unsignedInteger(vocab_size, "vocab_size")
    this.#inner.train(vocab_size)
  }

  trainWithBbpeFallback(vocab_size: number, primary_vocab_ratio = 0.9): void {
    unsignedInteger(vocab_size, "vocab_size")
    this.#inner.trainWithBbpeFallback(vocab_size, primary_vocab_ratio)
  }

  step(): number {
    return this.#inner.step()
  }

  validateModel(): BpeModel {
    return BpeModel.fromRaw(this.#inner.validateModel())
  }

  async saveFiles(
    vocab_file: string | URL,
    merges_file: string | URL,
    format?: FileFormat | null,
  ): Promise<void> {
    await this.validateModel().saveFiles(vocab_file, merges_file, format)
  }
}

export class BpeModel {
  #default_encoder: BpeEncoder | undefined

  private constructor(private readonly inner: RawBpeModel) {}

  /** @internal */
  static fromRaw(inner: RawBpeModel): BpeModel {
    return new BpeModel(inner)
  }

  get unit(): Unit {
    return this.inner.unit as Unit
  }

  get vocab(): VocabularyEntry[] {
    return vocabularyItems(this.inner.vocabItems())
  }

  get lastMergeFreq(): number | null {
    return nullableFrequency(this.inner.lastMergeFreq)
  }

  get specialTokens(): string[] {
    return [...this.inner.specialTokens]
  }

  encoder(options: PreTokenizerOptions & { split_on_vocab_bigrams?: boolean } = {}): BpeEncoder {
    const use_cache = (
      options.pat_str == null
      && options.unicode_bigrams == null
      && (options.unicode_bigram_mixed_boundary ?? "keep") === "keep"
      && (options.split_on_vocab_bigrams ?? true)
    )
    if (use_cache && this.#default_encoder !== undefined) return this.#default_encoder
    const encoder = BpeEncoder.fromRaw(this.inner.encoder({
      ...preTokenizerOptions(options),
      split_on_vocab_bigrams: options.split_on_vocab_bigrams ?? true,
    }))
    if (use_cache) this.#default_encoder = encoder
    return encoder
  }

  encode(text: string): Uint32Array {
    return this.encoder().encode(text)
  }

  decode(ids: Iterable<number>): string {
    return this.encoder().decode(ids)
  }

  serializeVocab(format?: FileFormat | null): string {
    return this.inner.serializeVocab(resolveFormat(this.unit, format))
  }

  serializeMerges(format?: FileFormat | null): string {
    return this.inner.serializeMerges(resolveFormat(this.unit, format))
  }

  async saveVocabJson(file: string | URL, format?: FileFormat | null): Promise<void> {
    await writeTextFile(file, this.serializeVocab(format), "saveVocabJson")
  }

  async saveMergesTxt(file: string | URL, format?: FileFormat | null): Promise<void> {
    await writeTextFile(file, this.serializeMerges(format), "saveMergesTxt")
  }

  async saveFiles(
    vocab_file: string | URL,
    merges_file: string | URL,
    format?: FileFormat | null,
  ): Promise<void> {
    await Promise.all([
      this.saveVocabJson(vocab_file, format),
      this.saveMergesTxt(merges_file, format),
    ])
  }

  toPretrainedFiles(options: BpeEncoderOptions = {}): PretrainedFiles {
    const format = resolveFormat(this.unit, options.format)
    const encoder_options = {
      ...options,
      unit: this.unit,
      special_tokens: this.specialTokens,
    }
    this.encoder(encoder_options)
    const config: ModelConfig = {
      version: 1,
      unit: this.unit,
      format,
      vocab_file: "vocab.json",
      merges_file: "merges.txt",
      special_tokens: this.specialTokens,
      pat_str: options.pat_str ?? null,
      unicode_bigrams: options.unicode_bigrams === null || options.unicode_bigrams === undefined
        ? null
        : [...options.unicode_bigrams],
      unicode_bigram_mixed_boundary: options.unicode_bigram_mixed_boundary ?? "keep",
      split_on_vocab_bigrams: options.split_on_vocab_bigrams ?? true,
    }
    return {
      "ffbpe.json": `${JSON.stringify(config, null, 2)}\n`,
      "vocab.json": this.serializeVocab(format),
      "merges.txt": this.serializeMerges(format),
    }
  }

  async savePretrained(directory: string | URL, options: BpeEncoderOptions = {}): Promise<void> {
    const selected_runtime = runtime()
    if (selected_runtime.kind !== "node") {
      throw new Error("savePretrained is only available in the Node runtime; use toPretrainedFiles in browsers")
    }
    await selected_runtime.write_model_files(directory, this.toPretrainedFiles(options))
  }
}

export class BpeEncoder {
  private constructor(private readonly inner: RawBpeEncoder) {}

  /** @internal */
  static fromRaw(inner: RawBpeEncoder): BpeEncoder {
    return new BpeEncoder(inner)
  }

  static fromData(
    vocab: ReadonlyMap<Uint8Array, number> | readonly VocabularyEntry[],
    merges: readonly (readonly [Uint8Array, Uint8Array])[],
    options: BpeEncoderOptions = {},
  ): BpeEncoder {
    return BpeEncoder.fromRaw(wasm().WasmBpeEncoder.fromData(
      normalizeVocabulary(vocab),
      normalizeMerges(merges),
      encoderOptions(options),
    ))
  }

  static fromSerialized(vocab: string, merges: string, options: BpeEncoderOptions = {}): BpeEncoder {
    return BpeEncoder.fromRaw(wasm().WasmBpeEncoder.fromFiles(vocab, merges, encoderOptions(options)))
  }

  static async load(
    vocab_file: TextFile,
    merges_file: TextFile,
    options: BpeEncoderOptions = {},
  ): Promise<BpeEncoder> {
    const [vocab, merges] = await Promise.all([
      readTextFile(vocab_file),
      readTextFile(merges_file),
    ])
    return BpeEncoder.fromSerialized(vocab, merges, options)
  }

  static async fromPretrained(directory: string | URL): Promise<BpeEncoder> {
    const selected_runtime = runtime()
    let config_text: string
    try {
      config_text = await selected_runtime.read_model_file(directory, "ffbpe.json")
    } catch (error) {
      try {
        config_text = await selected_runtime.read_model_file(directory, "unitoken.json")
      } catch {
        throw error
      }
    }
    const config = validateModelConfig(JSON.parse(config_text) as unknown)
    const [vocab, merges] = await Promise.all([
      selected_runtime.read_model_file(directory, config.vocab_file),
      selected_runtime.read_model_file(directory, config.merges_file),
    ])
    return BpeEncoder.fromSerialized(vocab, merges, modelOptions(config))
  }

  get unit(): Unit {
    return this.inner.unit as Unit
  }

  preTokenizer(): PreTokenizer {
    return PreTokenizer.fromRaw(this.inner.preTokenizer())
  }

  encodeWord(word: string): Uint32Array {
    return this.inner.encodeWord(word)
  }

  encodeWords(words: readonly string[]): Uint32Array[] {
    return this.inner.encodeWords([...words]).map(ids => Uint32Array.from(ids))
  }

  encode(text: string): Uint32Array {
    return this.inner.encode(text)
  }

  /** Return the exact vocabulary bytes represented by a token id. */
  tokenBytes(id: number): Uint8Array {
    unsignedInteger(id, "token id", 0xffff_ffff)
    return this.inner.tokenBytes(id)
  }

  async encodeFile(file: string | URL | Blob): Promise<Uint32Array> {
    return this.encode(await readTextFile(file))
  }

  decode(ids: Iterable<number>): string {
    return this.inner.decode([...ids])
  }
}

export function trainBpe(texts: string | Iterable<string>, options: TrainBpeOptions): BpeModel {
  const special_tokens = [...(options.special_tokens ?? [])]
  const pre_tokenizer = new PreTokenizer(special_tokens)
  const counter = pre_tokenizer.wordCounter()
  counter.addSource(typeof texts === "string" ? [texts] : texts)
  const trainer = new BpeTrainer(special_tokens, options)
  trainer.addWordCounter(counter)
  trainer.train(options.vocab_size)
  return trainer.validateModel()
}

async function readTextFile(file: string | URL | Blob): Promise<string> {
  const selected_runtime = runtime()
  if (selected_runtime.kind === "browser") {
    if (!(file instanceof Blob)) {
      throw new TypeError("Browser file operations expect a File or Blob")
    }
    return selected_runtime.read_text_file(file)
  }
  if (typeof file !== "string" && !(file instanceof URL)) {
    throw new TypeError("Node file operations expect a filesystem path or file URL")
  }
  return selected_runtime.read_text_file(file)
}

async function writeTextFile(file: string | URL, content: string, operation: string): Promise<void> {
  const selected_runtime = runtime()
  if (selected_runtime.kind !== "node") {
    throw new Error(`${operation} is only available in the Node runtime; use the serialize method in browsers`)
  }
  await selected_runtime.write_text_file(file, content)
}

export type { FileFormat }
