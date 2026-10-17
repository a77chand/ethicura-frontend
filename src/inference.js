// On-device inference: MediaPipe BlazeFace finds the face, an ONNX export of
// the detector classifies the crop and returns a Grad-CAM-equivalent heatmap.
// Nothing leaves the browser.
import * as ort from "onnxruntime-web/webgpu";
import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";

const BASE = import.meta.env.BASE_URL;
const INPUT_SIZE = 224;

// The classifier was trained on MTCNN face boxes with a 20% margin (see
// ethicura-ai/src/detector.py). BlazeFace boxes sit differently on the face,
// so they're mapped onto MTCNN's. Fitted on 73 frames where both detectors
// found the face:
//  - BOX_CALIBRATION: median centre offset and size ratio, in units of the
//    BlazeFace box (fallback when landmarks are missing);
//  - LANDMARK_MAP: ridge regression from BlazeFace's 6 landmarks (relative to
//    its box) to MTCNN's box centre and size. Leave-one-out crop IoU against
//    the API's crop: median 0.915 (vs 0.907), worst 0.57 (vs 0.46).
export const BOX_CALIBRATION = { dx: -0.0032, dy: -0.1318, sw: 0.9061, sh: 1.1834 };
// rows: [dcx, dcy, w, h]; columns: 6 landmark x, 6 landmark y, h/w, bias
const LANDMARK_MAP = [
  [-0.03976, 0.06436, 0.05576, 0.13168, 0.1591, 0.44171, 0.05488, 0.4074, -0.12611, -0.29621, -0.16872, -0.06694, -0.00779, -0.01067],
  [0.08287, 0.11949, -0.03711, -0.04907, 0.10505, -0.05541, -0.06594, -0.43414, 0.43621, 0.21778, -0.02801, 0.01913, -0.12935, -0.09213],
  [-0.48462, 0.35418, -0.12118, 0.01525, 0.06268, 0.27204, -0.18911, -0.10109, 0.06836, -0.51797, -0.12609, -0.08182, 0.33211, 0.3158],
  [-0.83854, 1.11806, -0.3034, -0.35398, 0.13504, 0.38837, -0.65275, -0.75247, 0.2078, -0.50037, -0.76156, -0.54262, 0.15064, 0.13635],
];
const MARGIN = 0.2;

export async function createFaceDetector(runningMode = "VIDEO") {
  const fileset = await FilesetResolver.forVisionTasks(`${BASE}mediapipe/wasm`);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: `${BASE}models/blaze_face_short_range.tflite`, delegate },
    runningMode,
    minDetectionConfidence: 0.5,
  });
  try {
    return await FaceDetector.createFromOptions(fileset, options("GPU"));
  } catch {
    return FaceDetector.createFromOptions(fileset, options("CPU"));
  }
}

// Largest-confidence face → {x, y, w, h, keypoints: [[x, y] ×6]} in source
// pixels, or null. width/height convert MediaPipe's normalised landmarks.
export function bestFace(result, width, height) {
  const best = result?.detections?.reduce(
    (a, d) => (!a || d.categories[0].score > a.categories[0].score ? d : a),
    null
  );
  if (!best) return null;
  const b = best.boundingBox;
  const keypoints = width && best.keypoints?.length === 6
    ? best.keypoints.map((k) => [k.x * width, k.y * height])
    : null;
  return { x: b.originX, y: b.originY, w: b.width, h: b.height, keypoints };
}

// BlazeFace box (+ landmarks) → the crop the model was trained on.
export function cropBox(face, width, height, calibration = BOX_CALIBRATION) {
  let cx, cy, w, h;
  if (face.keypoints) {
    const fcx = face.x + face.w / 2;
    const fcy = face.y + face.h / 2;
    const features = [
      ...face.keypoints.map(([x]) => (x - fcx) / face.w),
      ...face.keypoints.map(([, y]) => (y - fcy) / face.w),
      face.h / face.w,
      1,
    ];
    const [dcx, dcy, sw, sh] = LANDMARK_MAP.map((row) => row.reduce((s, wgt, i) => s + wgt * features[i], 0));
    cx = fcx + dcx * face.w;
    cy = fcy + dcy * face.w;
    w = sw * face.w;
    h = sh * face.w;
  } else {
    cx = face.x + face.w / 2 + calibration.dx * face.w;
    cy = face.y + face.h / 2 + calibration.dy * face.h;
    w = face.w * calibration.sw;
    h = face.h * calibration.sh;
  }
  const margin = MARGIN * Math.max(w, h);
  const x1 = Math.max(0, Math.round(cx - w / 2 - margin));
  const y1 = Math.max(0, Math.round(cy - h / 2 - margin));
  const x2 = Math.min(width, Math.round(cx + w / 2 + margin));
  const y2 = Math.min(height, Math.round(cy + h / 2 + margin));
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}

export async function createClassifier() {
  ort.env.wasm.wasmPaths = `${BASE}ort/`;
  const url = `${BASE}models/detector.onnx`;
  let session = null;
  let backend = null;
  for (const provider of ["webgpu", "wasm"]) {
    try {
      session = await ort.InferenceSession.create(url, { executionProviders: [provider] });
      backend = provider;
      break;
    } catch (err) {
      console.warn(`ONNX Runtime: ${provider} unavailable`, err);
    }
  }
  if (!session) throw new Error("Could not start the model in this browser.");

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const input = new Float32Array(INPUT_SIZE * INPUT_SIZE * 3);

  // Crop `box` out of `source` (video/image/canvas), resize to 224x224 and
  // scale to [0, 1] — the same input the Python API builds.
  async function classify(source, box) {
    // Copy the crop 1:1, then resize in JS. The browser's own smoothing
    // filters differ from OpenCV's, and the model is sensitive to that.
    canvas.width = box.w;
    canvas.height = box.h;
    ctx.drawImage(source, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h);
    resizeBilinearLikeOpenCV(ctx.getImageData(0, 0, box.w, box.h), input);
    const feeds = { face: new ort.Tensor("float32", input, [1, INPUT_SIZE, INPUT_SIZE, 3]) };
    const out = await session.run(feeds);
    const [, h, w] = out.cam.dims;
    const camData = out.cam.data;
    const cam = Array.from({ length: h }, (_, y) => Array.from(camData.subarray(y * w, (y + 1) * w)));
    return { prob: out.prob.data[0], cam };
  }

  return { backend, classify };
}

// cv2.resize(..., INTER_LINEAR) — what the training pipeline and the Python
// API use — into a 224x224x3 float array in [0, 1]. Same half-pixel-centre
// mapping and edge clamping as OpenCV, and results rounded to uint8 first
// like OpenCV's output, so the model sees the pixels it was trained on.
function resizeBilinearLikeOpenCV(image, out) {
  const { data, width: w, height: h } = image;
  const sx = w / INPUT_SIZE;
  const sy = h / INPUT_SIZE;
  const axis = (d, scale, size) => {
    let f = (d + 0.5) * scale - 0.5;
    let i = Math.floor(f);
    f -= i;
    if (i < 0) { i = 0; f = 0; }
    if (i >= size - 1) { i = size - 1; f = 0; }
    return [i, Math.min(i + 1, size - 1), f];
  };
  const xs = Array.from({ length: INPUT_SIZE }, (_, d) => axis(d, sx, w));
  let j = 0;
  for (let dy = 0; dy < INPUT_SIZE; dy++) {
    const [y0, y1, fy] = axis(dy, sy, h);
    const r0 = y0 * w * 4;
    const r1 = y1 * w * 4;
    for (let dx = 0; dx < INPUT_SIZE; dx++) {
      const [x0, x1, fx] = xs[dx];
      const a = r0 + x0 * 4, b = r0 + x1 * 4, c = r1 + x0 * 4, e = r1 + x1 * 4;
      for (let ch = 0; ch < 3; ch++) {
        const top = data[a + ch] + (data[b + ch] - data[a + ch]) * fx;
        const bottom = data[c + ch] + (data[e + ch] - data[c + ch]) * fx;
        out[j++] = Math.round(top + (bottom - top) * fy) / 255;
      }
    }
  }
}
