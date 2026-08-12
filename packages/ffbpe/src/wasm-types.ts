export interface RawPreTokenizer {
  free(): void
  withUnicodeBigrams(bigrams: string[]): RawPreTokenizer
  getWords(text: string): Record<string, number>
  split(text: string): Array<{
    kind: "word" | "special"
    text: string
    start_byte: number
    end_byte: number
  }>
  bigramCounter(): RawBigramCounter
  wordCounter(): RawWordCounter
  loadWordCounter(serialized: string): RawWordCounter
}

export interface RawBigramCounter {
  free(): void
  addText(text: string): void
  addBatch(texts: string[]): void
  merge(other: RawBigramCounter): void
  selected(top_k: number, min_freq: number): string[]
  select(top_k: number, min_freq: number): unknown
  items(): Array<[string, number]>
}

export interface RawWordCounter {
  free(): void
  addText(text: string): void
  addBatch(texts: string[]): void
  merge(other: RawWordCounter): void
  words(): Record<string, number>
  readonly len: number
  readonly isEmpty: boolean
  clear(): void
  serialize(): string
}

export interface RawBpeEncoder {
  free(): void
  readonly unit: string
  preTokenizer(): RawPreTokenizer
  encodeWord(word: string): Uint32Array
  encodeWords(words: string[]): number[][]
  encode(text: string): Uint32Array
  tokenBytes(id: number): Uint8Array
  decode(ids: number[]): string
}

export interface RawBpeModel {
  free(): void
  readonly unit: string
  readonly lastMergeFreq: unknown
  readonly specialTokens: string[]
  vocabItems(): Array<[number[], number]>
  encoder(options: object): RawBpeEncoder
  serializeVocab(format: string): string
  serializeMerges(format: string): string
}

export interface RawBpeTrainer {
  free(): void
  readonly unit: string
  readonly vocabSize: number
  readonly lastMergeFreq: unknown
  readonly hotPairWindowStats: unknown
  readonly memoryUsage: unknown
  addWords(words: Array<[string, number]>): void
  addWordCounter(counter: RawWordCounter): void
  initTraining(): void
  train(vocab_size: number): void
  trainWithBbpeFallback(vocab_size: number, primary_vocab_ratio: number): void
  step(): number
  vocabItems(): Array<[number[], number]>
  validateModel(): RawBpeModel
}

export interface WasmModule {
  default(input?: unknown): Promise<unknown>
  WasmPreTokenizer: new (special_tokens: string[], options: object) => RawPreTokenizer
  WasmBpeTrainer: new (special_tokens: string[], options: object) => RawBpeTrainer
  WasmBpeEncoder: {
    fromData(vocab: Array<[number[], number]>, merges: Array<[number[], number[]]>, options: object): RawBpeEncoder
    fromFiles(vocab: string, merges: string, options: object): RawBpeEncoder
    fromTiktoken(model: string, special_tokens: object[], options: object): RawBpeEncoder
  }
}
