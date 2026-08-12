import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import test from "node:test";

import {
  TOKENIZER_PRESETS,
  createPresetEncoder,
} from "@tokn-ai/ffbpe-presets";

const presetRequire = createRequire(new URL(
  "../../../packages/ffbpe-presets/package.json",
  import.meta.url,
));
const presetFfbpeUrl = pathToFileURL(presetRequire.resolve("@tokn-ai/ffbpe"));
const { FFBPE } = await import(presetFfbpeUrl.href);
await FFBPE.init();

const EXPECTED_IDS = {
  gpt2: [31_373, 995, 220, 19_526, 254, 25_001, 121, 50_169, 233],
  r50k_base: [31_373, 995, 220, 19_526, 254, 25_001, 121, 50_169, 233],
  p50k_base: [31_373, 995, 220, 19_526, 254, 25_001, 121, 50_169, 233],
  cl100k_base: [15_339, 1_917, 220, 57_668, 53_901, 62_904, 233],
  o200k_base: [24_912, 2_375, 220, 177_519, 61_138, 233],
};

test("vendored presets match their official hashes and token IDs", async () => {
  const input = "hello world 你好 👋";

  for (const preset of TOKENIZER_PRESETS) {
    const model = await readFile(new URL(
      `../public/models/${preset.name}.tiktoken`,
      import.meta.url,
    ));
    assert.equal(
      createHash("sha256").update(model).digest("hex"),
      preset.model_sha256,
      `${preset.name} model hash`,
    );

    const encoder = createPresetEncoder(preset.name, model);
    assert.deepEqual([...encoder.encode(input)], EXPECTED_IDS[preset.name]);
    assert.equal(encoder.decode(EXPECTED_IDS[preset.name]), input);
  }
});
