import { rm } from "node:fs/promises"

await rm(new URL("../wasm/.gitignore", import.meta.url), { force: true })
