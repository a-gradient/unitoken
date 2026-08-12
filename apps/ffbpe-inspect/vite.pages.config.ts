import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { defineConfig } from "vite";

const repositoryRoot = fileURLToPath(new URL("../..", import.meta.url));
const ffbpeBrowserEntry = resolve(repositoryRoot, "packages/ffbpe/dist/browser.js");
const ffbpePackagePattern = /^@tokn-ai\/ffbpe(?:\/browser)?$/;

export default defineConfig({
  base: "./",
  build: {
    manifest: true,
    outDir: "dist-pages",
    emptyOutDir: true,
  },
  resolve: {
    alias: [
      { find: ffbpePackagePattern, replacement: ffbpeBrowserEntry },
    ],
  },
  server: {
    fs: { allow: [repositoryRoot] },
  },
  plugins: [react()],
});
