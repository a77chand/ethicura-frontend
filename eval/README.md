# Evaluating the in-browser pipeline

These scripts score images with **exactly the code the live demo runs**: `src/inference.js`, meaning MediaPipe BlazeFace, the landmark crop map, the OpenCV-style resize and ONNX Runtime Web. They run in headless Chrome, on the same DeepFakeFace validation and test images as the Python evaluation in [ethicura-ai](https://github.com/a77chand/ethicura-ai).

```bash
npm i --no-save puppeteer-core
ln -s /path/to/DeepFakeFace public/__eval     # wiki/, inpainting/, insight/, text2img/ (gitignored; remove before deploying)
npm run dev
node eval/browser_eval.mjs http://localhost:5173 items.json scores.jsonl   # resumable
python eval/analyse.py ../ethicura-ai items.json scores.jsonl report.json
```

`items.json` lists `{id, url, split, source, label, group}` for each image. It's built from the face-crop manifest so the image set and filters match the Python evaluation. The latest report is in [`ethicura-ai/results/browser_pipeline.json`](https://github.com/a77chand/ethicura-ai/blob/main/results/browser_pipeline.json).
