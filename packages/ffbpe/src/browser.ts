import {
  BigramCounter,
  BpeEncoder,
  BpeModel,
  BpeTrainer,
  FFBPE as CoreFFBPE,
  PreTokenizer,
  WordCounter,
  trainBpe,
} from "./core.js"
import type { BrowserRuntime, PretrainedFiles } from "./types.js"

function modelFileUrl(base: string | URL, file_name: string): URL {
  const base_url = base instanceof URL ? new URL(base.href) : new URL(base, globalThis.location?.href)
  if (!base_url.pathname.endsWith("/")) base_url.pathname += "/"
  return new URL(file_name, base_url)
}

const browser_runtime: BrowserRuntime = {
  kind: "browser",
  async wasm_input() {
    return undefined
  },
  async read_model_file(base, file_name) {
    const url = modelFileUrl(base, file_name)
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`Cannot read FFBPE model file ${url.href}: ${response.status} ${response.statusText}`)
    }
    return response.text()
  },
  read_text_file(file) {
    return file.text()
  },
}

class FFBPE {
  private constructor() {}

  static async init(): Promise<void> {
    CoreFFBPE.configureRuntime(browser_runtime)
    await CoreFFBPE.init()
  }
}

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
