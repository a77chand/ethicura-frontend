// Copy the WebAssembly runtimes for ONNX Runtime Web and MediaPipe into
// public/, so the app serves them itself instead of depending on a CDN.
import { cpSync, mkdirSync, readdirSync } from "node:fs";
import { join } from "node:path";

const copy = (from, to, keep) => {
  mkdirSync(to, { recursive: true });
  for (const f of readdirSync(from).filter(keep)) cpSync(join(from, f), join(to, f));
};

// Only the builds the app actually loads: ORT's WebGPU entry uses the
// asyncify build for both its WebGPU and wasm backends.
copy("node_modules/onnxruntime-web/dist", "public/ort", (f) => f.startsWith("ort-wasm-simd-threaded.asyncify."));
copy("node_modules/@mediapipe/tasks-vision/wasm", "public/mediapipe/wasm", (f) => f.startsWith("vision_wasm_internal."));
console.log("copied ONNX Runtime + MediaPipe wasm into public/");
