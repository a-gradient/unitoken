export type Unit = "byte" | "unicode"
export type FileFormat = "gpt2" | "unitoken"
export type InitialAlphabet = "raw" | "byte_level"
export type TieBreak = "smallest_pair_id" | "largest_content"
export type UnicodeBigramMixedBoundary = "keep" | "split"

export interface PreTokenizerOptions {
  eot_token?: string | null
  pat_str?: string | null
  unicode_bigrams?: readonly string[] | null
  unicode_bigram_mixed_boundary?: UnicodeBigramMixedBoundary
}

export interface BpeEncoderOptions extends PreTokenizerOptions {
  unit?: Unit
  format?: FileFormat | null
  special_tokens?: readonly string[]
  split_on_vocab_bigrams?: boolean
}

export interface BpeTrainerOptions {
  unit?: Unit
  initial_alphabet?: InitialAlphabet
  tie_break?: TieBreak
  parallel_merge_min_occurs_in?: number | null
  hot_pair_window_size?: number | null
  bigram_cutoff_freq?: number | null
}

export interface TrainBpeOptions extends BpeTrainerOptions {
  vocab_size: number
  special_tokens?: readonly string[]
}

export interface SourceBatchOptions {
  max_records?: number
  max_bytes?: number
}

export interface ModelConfig {
  version: 1
  unit: Unit
  format: FileFormat
  vocab_file: string
  merges_file: string
  special_tokens: string[]
  pat_str: string | null
  unicode_bigrams: string[] | null
  unicode_bigram_mixed_boundary: UnicodeBigramMixedBoundary
  split_on_vocab_bigrams: boolean
}

export interface PretrainedFiles {
  "ffbpe.json": string
  "vocab.json": string
  "merges.txt": string
}

export interface UnicodeBigramSelection {
  bigrams: string[]
  cutoff_freq: number | null
  max_excluded_freq: number | null
}

export interface HotPairWindowStats {
  hydration_scans: number
  hydrated_word_entries: number
  batch_prunes: number
  prune_evictions: number
  peak_resident_pairs: number
  resident_pairs: number
  occurrence_capacity: number
}

export interface TrainerMemoryUsage {
  word_entries: number
  word_entry_capacity: number
  word_storage_bytes: number
  pair_entries: number
  pair_table_capacity: number
  pair_table_bytes: number
  occurrence_set_slots: number
  occurrence_set_slot_capacity: number
  occurrence_set_header_bytes: number
  occurrence_capacity_entries: number
  occurrence_capacity_bytes: number
  merge_heap_entries: number
  merge_heap_capacity: number
  merge_heap_bytes: number
  merge_entries: number
  merge_storage_bytes: number
  vocab_entries: number
  vocab_token_bytes: number
  estimated_persistent_bytes: number
}

export type TextFile = string | URL | Blob

export type WordFrequencies = Readonly<Record<string, number>>
export type WordEntry = readonly [word: string, frequency: number]
export type VocabularyEntry = readonly [token: Uint8Array, id: number]

export interface BrowserRuntime {
  kind: "browser"
  wasm_input(): Promise<undefined>
  read_model_file(base: string | URL, file_name: string): Promise<string>
  read_text_file(file: Blob): Promise<string>
}

export interface NodeRuntime {
  kind: "node"
  wasm_input(): Promise<Uint8Array>
  read_model_file(base: string | URL, file_name: string): Promise<string>
  read_text_file(file: string | URL): Promise<string>
  write_model_files(directory: string | URL, files: PretrainedFiles): Promise<void>
  write_text_file(file: string | URL, content: string): Promise<void>
}

export type Runtime = BrowserRuntime | NodeRuntime
