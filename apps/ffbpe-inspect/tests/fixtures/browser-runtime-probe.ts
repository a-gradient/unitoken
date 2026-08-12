import { FFBPE } from "@tokn-ai/ffbpe/browser"

export async function initializeBrowserRuntime(): Promise<void> {
  await FFBPE.init()
}
