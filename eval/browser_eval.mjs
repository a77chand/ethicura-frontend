// Score images with the *live demo's* pipeline (src/inference.js: BlazeFace →
// landmark crop map → OpenCV-style resize → ONNX Runtime Web) in headless
// Chrome. Usage: node browser_eval.mjs <app url> <items.json> <out.jsonl>
// items.json: [{id, url}] where url is served by the app's dev server.
import puppeteer from "puppeteer-core";
import fs from "node:fs";

const [, , appUrl, itemsPath, outPath] = process.argv;
const items = JSON.parse(fs.readFileSync(itemsPath));
const done = new Set(
  fs.existsSync(outPath) ? fs.readFileSync(outPath, "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l).id) : []
);
const todo = items.filter((it) => !done.has(it.id));
console.log(`items ${items.length}, already scored ${done.size}, to do ${todo.length}`);

const browser = await puppeteer.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: "new",
  args: ["--enable-unsafe-webgpu"],
  protocolTimeout: 600000,
});
const page = await browser.newPage();
page.on("console", (m) => m.type() === "error" && !m.text().includes("CONNECTION_REFUSED") && console.error("[page]", m.text()));
await page.goto(appUrl, { waitUntil: "networkidle0" });
const backend = await page.evaluate(async () => {
  const inf = await import("/src/inference.js");
  window.__inf = inf;
  window.__det = await inf.createFaceDetector("IMAGE");
  window.__clf = await inf.createClassifier();
  return window.__clf.backend;
});
console.log("backend:", backend);

const out = fs.createWriteStream(outPath, { flags: "a" });
const started = Date.now();
const BATCH = 200;
for (let i = 0; i < todo.length; i += BATCH) {
  const batch = todo.slice(i, i + BATCH);
  const rows = await page.evaluate(async (batch) => {
    const res = [];
    for (const it of batch) {
      try {
        const img = new Image();
        img.src = it.url;
        await img.decode();
        const W = img.naturalWidth, H = img.naturalHeight;
        const face = window.__inf.bestFace(window.__det.detect(img), W, H);
        if (!face) { res.push({ id: it.id, prob: null }); continue; }
        const box = window.__inf.cropBox(face, W, H);
        const { prob } = await window.__clf.classify(img, box);
        res.push({ id: it.id, prob, box_w: box.w });
      } catch (e) {
        res.push({ id: it.id, prob: null, error: String(e).slice(0, 120) });
      }
    }
    return res;
  }, batch);
  for (const r of rows) out.write(JSON.stringify(r) + "\n");
  const n = Math.min(i + BATCH, todo.length);
  const rate = n / ((Date.now() - started) / 1000);
  if ((i / BATCH) % 10 === 0 || n === todo.length)
    console.log(`${n}/${todo.length} (${rate.toFixed(1)} img/s, ~${((todo.length - n) / rate / 60).toFixed(1)} min left)`);
}
out.end();
await browser.close();
