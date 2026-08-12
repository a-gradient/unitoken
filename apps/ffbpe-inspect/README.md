# FFBPE Inspect website

Interactive browser UI for comparing FFBPE pretokenizer boundaries with final
BPE tokens. It consumes the local `@tokn-ai/ffbpe`,
`@tokn-ai/ffbpe-inspect`, and `@tokn-ai/ffbpe-presets` packages and runs the
tokenizer entirely in WASM.

The website includes verified copies of the official GPT-2/r50k, p50k,
cl100k, and o200k ranked vocabularies. Only the selected model is fetched by
the browser; input text never leaves the tab.

```sh
npm --prefix ../../packages/ffbpe run build
npm --prefix ../../packages/ffbpe-inspect run build
npm --prefix ../../packages/ffbpe-presets run build
npm install
npm run dev
```

Use `npm test` to run the production build and server-rendering smoke test.
