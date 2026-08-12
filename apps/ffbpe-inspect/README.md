# FFBPE Inspect website

Interactive browser UI for comparing FFBPE pretokenizer boundaries with final
BPE tokens. It consumes the local `@tokn-ai/ffbpe` and
`@tokn-ai/ffbpe-inspect` packages and runs the tokenizer entirely in WASM.

```sh
npm --prefix ../../packages/ffbpe run build
npm --prefix ../../packages/ffbpe-inspect run build
npm install
npm run dev
```

Use `npm test` to run the production build and server-rendering smoke test.
