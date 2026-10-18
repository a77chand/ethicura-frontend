# Ethicura AI: in-browser deepfake detection

[![live demo](https://img.shields.io/badge/demo-live-brightgreen)](https://ethicura-frontend.vercel.app)

**[ethicura-frontend.vercel.app](https://ethicura-frontend.vercel.app)**: point your webcam at a face and get a live fake-probability score with a heatmap of the regions that drove it. Everything runs **on your device**, and no video is uploaded.

- **Face tracking:** MediaPipe BlazeFace on every video frame. A ridge regression on its 6 landmarks maps the box onto the MTCNN-style crop the model was trained on.
- **Classifier:** Ethicura's own EfficientNetB0, trained on 120k images and exported to ONNX, running on ONNX Runtime Web (WebGPU, with a WebAssembly fallback). About 30 analyses per second on a laptop.
- **Heatmap without a backward pass:** for a GAP → dense head, Grad-CAM reduces to classic CAM, so the exported graph outputs the heatmap directly.
- **Same preprocessing as training:** a JavaScript port of OpenCV's bilinear resize, because the canvas's own smoothing shifts predictions.

**Measured end to end in the browser** on 8,057 test faces of people never seen in training, using [`eval/`](eval/):

| | |
|---|---|
| AUC | **0.982** (0.980–0.985), vs 0.986 for the Python/MTCNN pipeline |
| Balanced accuracy | 93.4% |
| Calibration error (ECE) | 0.014 |
| Faces BlazeFace missed | 2.2% |

Model training, evaluation and the Python API are in **[a77chand/ethicura-ai](https://github.com/a77chand/ethicura-ai)**.

```bash
npm install
npm run dev      # copies the ONNX Runtime + MediaPipe wasm into public/, then starts Vite
npm run build
```
