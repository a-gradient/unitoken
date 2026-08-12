import { mkdir, readFile, writeFile } from "node:fs/promises"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

import {
  BigramCounter,
  BpeEncoder,
  BpeModel,
  BpeTrainer,
  FFBPE,
  PreTokenizer,
  WordCounter,
  trainBpe,
} from "./core.js"
import type { NodeRuntime, PretrainedFiles } from "./types.js"

function filePath(value: string | URL): string {
  return value instanceof URL ? fileURLToPath(value) : value
}

function modelPath(base: string | URL, file_name: string): string {
  return join(filePath(base), file_name)
}

const node_runtime: NodeRuntime = {
  kind: "node",
  async wasm_input() {
    const wasm_url = new URL("../wasm/ffbpe_wasm_bg.wasm", import.meta.url)
    return new Uint8Array(await readFile(wasm_url))
  },
  read_model_file(base, file_name) {
    return readFile(modelPath(base, file_name), "utf8")
  },
  read_text_file(file) {
    return readFile(filePath(file), "utf8")
  },
  async write_model_files(directory, files) {
    const directory_path = filePath(directory)
    await mkdir(directory_path, { recursive: true })
    await Promise.all(Object.entries(files).map(([file_name, content]) => {
      const path = join(directory_path, file_name)
      return mkdir(dirname(path), { recursive: true }).then(() => writeFile(path, content, "utf8"))
    }))
  },
  async write_text_file(file, content) {
    const path = filePath(file)
    await mkdir(dirname(path), { recursive: true })
    await writeFile(path, content, "utf8")
  },
}

FFBPE.configureRuntime(node_runtime)

export {
  BigramCounter,
  BpeEncoder,
  BpeModel,
  BpeTrainer,
  FFBPE,
  PreTokenizer,
  WordCounter,
  trainBpe,
}
export type * from "./types.js"
export type { PretrainedFiles }
