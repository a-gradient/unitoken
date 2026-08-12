import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";
import { build } from "vite";

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

test("optimized browser entry initializes its FFBPE runtime", async () => {
  const output_directory = await mkdtemp(join(tmpdir(), "ffbpe-browser-runtime-"));
  const repository_root = resolve("../..");
  const browser_entry = resolve(repository_root, "packages/ffbpe/dist/browser.js");

  await build({
    configFile: false,
    logLevel: "silent",
    build: {
      emptyOutDir: true,
      lib: {
        entry: resolve("tests/fixtures/browser-runtime-probe.ts"),
        fileName: () => "runtime-probe.mjs",
        formats: ["es"],
      },
      outDir: output_directory,
    },
    resolve: {
      alias: [
        { find: /^@tokn-ai\/ffbpe(?:\/browser)?$/, replacement: browser_entry },
      ],
    },
  });

  const native_fetch = globalThis.fetch;
  globalThis.fetch = async input => {
    const url = input instanceof Request ? new URL(input.url) : new URL(input);
    if (url.protocol !== "file:") return native_fetch(input);
    const body = await readFile(url);
    return new Response(body, { headers: { "Content-Type": "application/wasm" } });
  };

  try {
    const probe_url = pathToFileURL(join(output_directory, "runtime-probe.mjs"));
    probe_url.searchParams.set("test", `${process.pid}-${Date.now()}`);
    const probe = await import(probe_url.href);
    await probe.initializeBrowserRuntime();
  } finally {
    globalThis.fetch = native_fetch;
  }
});
