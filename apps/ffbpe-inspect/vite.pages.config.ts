import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const repositoryRoot = fileURLToPath(new URL("../..", import.meta.url));

export default defineConfig({
  base: "./",
  build: {
    outDir: "dist-pages",
    emptyOutDir: true,
  },
  resolve: {
    dedupe: ["@tokn-ai/ffbpe"],
  },
  server: {
    fs: { allow: [repositoryRoot] },
  },
  plugins: [react()],
});
