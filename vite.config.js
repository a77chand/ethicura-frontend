import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { createReadStream, existsSync } from "node:fs";
import { join } from "node:path";

// ONNX Runtime Web loads its wasm glue (public/ort/*.mjs) with a dynamic
// import(). Vite's dev server refuses to serve public/ files as modules, so
// hand them over untouched. Production builds copy public/ as-is and don't
// need this.
const serveOrtRuntime = {
  name: "serve-ort-runtime",
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const path = req.url?.split("?")[0] ?? "";
      const file = join(server.config.publicDir, path);
      if (!path.startsWith("/ort/") || !existsSync(file)) return next();
      res.setHeader("Content-Type", path.endsWith(".wasm") ? "application/wasm" : "text/javascript");
      createReadStream(file).pipe(res);
    });
  },
};

export default defineConfig({
  plugins: [react(), serveOrtRuntime],
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
