import type { WasmModule } from "./wasm-types.js"

let wasm_module: WasmModule | undefined
let init_promise: Promise<void> | undefined

export async function initWasm(input?: unknown): Promise<void> {
  if (wasm_module !== undefined) return
  init_promise ??= import("../wasm/ffbpe_wasm.js")
    .then(async (module: unknown) => {
      const typed_module = module as WasmModule
      await typed_module.default(input === undefined ? undefined : { module_or_path: input })
      wasm_module = typed_module
    })
  await init_promise
}

export function wasm(): WasmModule {
  if (wasm_module === undefined) {
    throw new Error("FFBPE is not initialized. Call `await FFBPE.init()` first.")
  }
  return wasm_module
}
