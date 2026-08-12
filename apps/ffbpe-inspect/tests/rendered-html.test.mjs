import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    {
      ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) },
    },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the inspector shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /FFBPE Inspect/);
  assert.match(html, /FF<\/b><i>\/<\/i><b>BPE/);
  assert.match(html, /class="site-header"/);
  assert.match(html, /href="\.\/" aria-current="page">Inspect/);
  assert.match(html, /href="\.\.\/docs\/">Docs/);
  assert.match(html, /TOKENIZER PRESET/);
  assert.match(html, /Downloading and verifying .*cl100k_base/);
  assert.doesNotMatch(html, /See what your|WHY TWO STEPS|Boundaries first|RUNS ENTIRELY/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("landing and inspector use the same site header contract", async () => {
  const [response, landing] = await Promise.all([
    render(),
    readFile(new URL("../../../landing/index.html", import.meta.url), "utf8"),
  ]);
  const inspector = await response.text();

  for (const html of [landing, inspector]) {
    assert.match(html, /class="site-header"/);
    assert.match(html, /class="site-header-inner"/);
    assert.match(html, /class="site-brand-mark"/);
    assert.match(html, /class="site-nav"/);
    assert.match(html, />Why FFBPE<\/a>/);
    assert.match(html, />Benchmarks<\/a>/);
    assert.match(html, />Inspect<\/a>/);
    assert.match(html, />Docs<\/a>/);
    assert.match(html, />GitHub<\/a>/);
  }
});

test("builds a relocatable GitHub Pages app", async () => {
  const html = await readFile(new URL("../dist-pages/index.html", import.meta.url), "utf8");

  assert.match(html, /FFBPE Inspect/);
  assert.match(html, /(?:src|href)="\.\/assets\//);
  assert.doesNotMatch(html, /(?:src|href)="\/(?:assets|models)\//);
});

test("static app bundles one configured FFBPE runtime", async () => {
  const manifest = JSON.parse(
    await readFile(new URL("../dist-pages/.vite/manifest.json", import.meta.url), "utf8"),
  );
  const entry = Object.values(manifest).find(chunk => chunk.isEntry === true);

  assert.ok(entry, "missing static app entry in Vite manifest");
  const source = await readFile(resolve("dist-pages", entry.file), "utf8");
  assert.equal(source.match(/No FFBPE runtime configured/g)?.length, 1);
  assert.equal(source.match(/wasm_input/g)?.length, 1);
});
