import { useEffect, useRef, useState, useCallback } from "react";
import { bestFace, createClassifier, createFaceDetector, cropBox } from "./inference.js";

// Everything runs on this device: MediaPipe finds the face on every video
// frame (so the box tracks it live), and the classifier runs on the newest
// frame whenever the previous result is back — no fixed rate, no queue.

// The displayed verdict averages the last few classified frames so one
// borderline frame doesn't flip the label back and forth.
const SMOOTHING_WINDOW = 8;
// Consecutive video frames without a face before the banner shows (~⅓ s),
// so a single missed detection doesn't flash it.
const NO_FACE_FRAMES = 10;
// Exponential smoothing of the face box between frames: removes detector
// jitter without visible lag.
const BOX_SMOOTHING = 0.5;
// The sparkline samples the score at most this often, so it spans ~10 s.
const HISTORY_INTERVAL_MS = 330;

export default function App() {
  const videoRef = useRef(null);
  const overlayCanvasRef = useRef(null); // heatmap + face box, drawn over the live video
  const streamRef = useRef(null);
  const modelsRef = useRef(null); // { detector, classifier }
  const rafRef = useRef(null);
  const busyRef = useRef(false);
  const boxRef = useRef(null); // smoothed crop box, in video pixels
  const heatRef = useRef(null); // { canvas, color } from the latest classification
  const showHeatmapRef = useRef(true);
  const noFaceFramesRef = useRef(0);
  const recentRef = useRef([]);
  const timingsRef = useRef([]);
  const lastHistoryAtRef = useRef(0);

  const [modelStatus, setModelStatus] = useState("loading"); // loading | ready | error
  const [modelError, setModelError] = useState(null);
  const [backend, setBackend] = useState(null);
  const [cameraStatus, setCameraStatus] = useState("idle"); // idle | requesting | live | denied | error
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [verdict, setVerdict] = useState(null); // { label, confidence } smoothed over SMOOTHING_WINDOW
  const [noFace, setNoFace] = useState(false);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null); // { fps, modelMs }

  // Load the face detector and classifier as soon as the page opens; the
  // model files are cached by the browser after the first visit.
  useEffect(() => {
    let cancelled = false;
    Promise.all([createFaceDetector("VIDEO"), createClassifier()])
      .then(([detector, classifier]) => {
        if (cancelled) return;
        modelsRef.current = { detector, classifier };
        setBackend(classifier.backend);
        setModelStatus("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        console.error(err);
        setModelError(err?.message ?? String(err));
        setModelStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    showHeatmapRef.current = showHeatmap;
  }, [showHeatmap]);

  const startCamera = useCallback(async () => {
    setCameraStatus("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraStatus("live");
    } catch (err) {
      setCameraStatus(err?.name === "NotAllowedError" ? "denied" : "error");
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    boxRef.current = null;
    heatRef.current = null;
    recentRef.current = [];
    timingsRef.current = [];
    noFaceFramesRef.current = 0;
    setCameraStatus("idle");
    setVerdict(null);
    setNoFace(false);
    setHistory([]);
    setStats(null);
  }, []);

  const classify = useCallback(async (video, box) => {
    busyRef.current = true;
    try {
      const startedAt = performance.now();
      const { prob, cam } = await modelsRef.current.classifier.classify(video, box);
      const now = performance.now();
      if (!streamRef.current) return; // stopped meanwhile

      const timings = [...timingsRef.current.slice(-19), { at: now, ms: now - startedAt }];
      timingsRef.current = timings;
      const span = timings[timings.length - 1].at - timings[0].at;
      setStats({
        fps: span > 0 ? ((timings.length - 1) * 1000) / span : null,
        modelMs: timings.reduce((sum, t) => sum + t.ms, 0) / timings.length,
      });

      const confidence = prob * 100;
      const recent = [...recentRef.current.slice(-(SMOOTHING_WINDOW - 1)), confidence];
      recentRef.current = recent;
      const avg = recent.reduce((sum, c) => sum + c, 0) / recent.length;
      const label = avg >= 50 ? "FAKE" : "REAL";
      setVerdict({ label, confidence: avg });
      heatRef.current = { canvas: heatmapToCanvas(cam), color: label === "FAKE" ? "#ff5c5c" : "#4ade80" };
      if (now - lastHistoryAtRef.current >= HISTORY_INTERVAL_MS) {
        lastHistoryAtRef.current = now;
        setHistory((prev) => [...prev.slice(-29), confidence]);
      }
    } catch (err) {
      console.error("classification failed", err);
    } finally {
      busyRef.current = false;
    }
  }, []);

  // While live: detect on every new video frame, classify when free, redraw.
  useEffect(() => {
    if (cameraStatus !== "live" || modelStatus !== "ready") return;
    const { detector } = modelsRef.current;
    let lastVideoTime = -1;

    const tick = () => {
      const video = videoRef.current;
      if (video && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const face = bestFace(detector.detectForVideo(video, performance.now()), video.videoWidth, video.videoHeight);
        if (face) {
          if (noFaceFramesRef.current >= NO_FACE_FRAMES) setNoFace(false);
          noFaceFramesRef.current = 0;
          const target = cropBox(face, video.videoWidth, video.videoHeight);
          boxRef.current = smoothBox(boxRef.current, target);
          if (!busyRef.current) classify(video, roundBox(boxRef.current));
        } else {
          noFaceFramesRef.current += 1;
          if (noFaceFramesRef.current === NO_FACE_FRAMES) {
            setNoFace(true);
            setVerdict(null);
            recentRef.current = [];
            boxRef.current = null;
            heatRef.current = null;
          }
        }
      }
      drawOverlay(overlayCanvasRef.current, video, boxRef.current, heatRef.current, showHeatmapRef.current);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [cameraStatus, modelStatus, classify]);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return (
    <div style={styles.page}>
      <div style={styles.scanline} />

      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <span style={styles.mark}>ETHICURA</span>
          <span style={styles.markSub}>AI // live analysis · on-device</span>
        </div>
        <div style={styles.headerRight}>
          <StatusPill cameraStatus={cameraStatus} modelStatus={modelStatus} />
        </div>
      </header>

      <main className="layout" style={styles.main}>
        <div style={styles.stage}>
          <div style={styles.videoWrap}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ ...styles.video, opacity: cameraStatus === "live" ? 1 : 0.15 }}
            />
            <canvas ref={overlayCanvasRef} style={styles.overlayCanvas} />

            {cameraStatus !== "live" && (
              <div style={styles.placeholder}>
                {cameraStatus === "idle" && (
                  <>
                    <p style={styles.placeholderTitle}>Camera is off</p>
                    <p style={styles.placeholderSub}>
                      Start analysis to scan for facial manipulation artifacts. Video is analysed in your
                      browser and never leaves this device.
                    </p>
                    <button style={styles.primaryBtn} onClick={startCamera}>
                      Start analysis
                    </button>
                    {modelStatus === "loading" && (
                      <p style={styles.placeholderNote}>Loading the model (16 MB, cached after the first visit)…</p>
                    )}
                  </>
                )}
                {cameraStatus === "requesting" && <p style={styles.placeholderTitle}>Requesting camera access…</p>}
                {cameraStatus === "denied" && (
                  <>
                    <p style={styles.placeholderTitle}>Camera access denied</p>
                    <p style={styles.placeholderSub}>Allow camera access in your browser settings, then try again.</p>
                    <button style={styles.primaryBtn} onClick={startCamera}>
                      Try again
                    </button>
                  </>
                )}
                {cameraStatus === "error" && (
                  <>
                    <p style={styles.placeholderTitle}>Couldn't access camera</p>
                    <button style={styles.primaryBtn} onClick={startCamera}>
                      Retry
                    </button>
                  </>
                )}
              </div>
            )}

            {cameraStatus === "live" && modelStatus === "ready" && noFace && (
              <div style={styles.noFaceBanner}>No face detected — center your face in frame</div>
            )}

            {cameraStatus === "live" && modelStatus === "loading" && (
              <div style={styles.noFaceBanner}>Loading the model…</div>
            )}

            {modelStatus === "error" && (
              <div style={styles.errorBanner}>The model couldn't start in this browser: {modelError}</div>
            )}
          </div>

          {cameraStatus === "live" && (
            <div style={styles.controls}>
              <button style={styles.toggleBtn} onClick={() => setShowHeatmap((v) => !v)}>
                {showHeatmap ? "Hide heatmap" : "Show heatmap"}
              </button>
              <button style={styles.stopBtn} onClick={stopCamera}>
                Stop
              </button>
            </div>
          )}
        </div>

        <aside style={styles.sidebar}>
          <Panel title="Verdict">
            {verdict ? (
              <div style={styles.verdictRow}>
                <span
                  style={{
                    ...styles.verdictLabel,
                    color: verdict.label === "FAKE" ? "#ff5c5c" : "#4ade80",
                  }}
                >
                  {verdict.label}
                </span>
                <span style={styles.confidenceValue}>{verdict.confidence.toFixed(1)}%</span>
              </div>
            ) : (
              <p style={styles.mutedText}>
                {cameraStatus === "live" ? "Waiting for a face…" : "Not scanning"}
              </p>
            )}
            <p style={styles.captionText}>Fake probability, averaged over the last {SMOOTHING_WINDOW} frames</p>
          </Panel>

          <Panel title="Confidence over time">
            <Sparkline values={history} />
          </Panel>

          <Panel title="Performance">
            {stats ? (
              <div style={styles.statsGrid}>
                <Stat label="analyses / s" value={stats.fps != null ? stats.fps.toFixed(0) : "—"} />
                <Stat label="model" value={`${Math.round(stats.modelMs)} ms`} />
                <Stat label="runs on" value={backend === "webgpu" ? "GPU" : "CPU"} />
              </div>
            ) : (
              <p style={styles.mutedText}>
                {cameraStatus === "live" ? "Measuring…" : backend ? `Ready on ${backend === "webgpu" ? "WebGPU" : "WebAssembly"}` : "—"}
              </p>
            )}
          </Panel>

          <Panel title="How this works">
            <ol style={styles.stepsList}>
              <li>MediaPipe BlazeFace tracks the face on every frame.</li>
              <li>An EfficientNetB0 classifier (ONNX, on your GPU) scores the face crop real vs. manipulated.</li>
              <li>A class activation map highlights the regions that drove the score.</li>
            </ol>
            <p style={styles.captionText}>Nothing is uploaded — all inference runs in this browser tab.</p>
          </Panel>
        </aside>
      </main>
    </div>
  );
}

function StatusPill({ cameraStatus, modelStatus }) {
  let text = "ready";
  let color = "#6b7280";
  if (modelStatus === "error") {
    text = "model error";
    color = "#ff5c5c";
  } else if (modelStatus === "loading") {
    text = "loading model";
    color = "#facc15";
  } else if (cameraStatus === "live") {
    text = "live";
    color = "#4ade80";
  } else if (cameraStatus === "requesting") {
    text = "connecting";
    color = "#facc15";
  }
  return (
    <div style={styles.statusPill}>
      <span style={{ ...styles.statusDot, background: color, boxShadow: `0 0 8px ${color}` }} />
      <span style={{ color }}>{text}</span>
    </div>
  );
}

function smoothBox(prev, next) {
  if (!prev) return next;
  const mix = (a, b) => a + (b - a) * BOX_SMOOTHING;
  return { x: mix(prev.x, next.x), y: mix(prev.y, next.y), w: mix(prev.w, next.w), h: mix(prev.h, next.h) };
}

function roundBox(b) {
  return { x: Math.round(b.x), y: Math.round(b.y), w: Math.max(1, Math.round(b.w)), h: Math.max(1, Math.round(b.h)) };
}

function drawOverlay(canvas, video, box, heat, showHeatmap) {
  if (!canvas || !video) return;
  if (video.videoWidth && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!box || !heat) return;
  if (showHeatmap) {
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(heat.canvas, box.x, box.y, box.w, box.h);
  }
  ctx.strokeStyle = heat.color;
  ctx.lineWidth = 2;
  ctx.strokeRect(box.x, box.y, box.w, box.h);
}

function Panel({ title, children }) {
  return (
    <div style={styles.panel}>
      <p style={styles.panelTitle}>{title}</p>
      {children}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div>
      <div style={styles.statValue}>{value}</div>
      <div style={styles.captionText}>{label}</div>
    </div>
  );
}

// Turn the raw Grad-CAM grid (e.g. 7x7, values 0-1) into a tiny RGBA canvas.
// Drawing it scaled up with smoothing gives the usual soft heatmap. Constant
// 40% opacity matches the backend's own overlay (addWeighted 0.6 / 0.4).
function heatmapToCanvas(grid) {
  const h = grid.length;
  const w = grid[0]?.length ?? 0;
  if (!w) return null;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  grid.forEach((row, y) =>
    row.forEach((v, x) => {
      const [r, g, b] = jetColor(v);
      const i = (y * w + x) * 4;
      img.data[i] = r;
      img.data[i + 1] = g;
      img.data[i + 2] = b;
      img.data[i + 3] = 102;
    })
  );
  ctx.putImageData(img, 0, 0);
  return canvas;
}

// Same palette as OpenCV's COLORMAP_JET used by the backend's PNG overlays.
function jetColor(v) {
  const clamp = (n) => Math.max(0, Math.min(1, n));
  return [
    Math.round(255 * clamp(1.5 - Math.abs(4 * v - 3))),
    Math.round(255 * clamp(1.5 - Math.abs(4 * v - 2))),
    Math.round(255 * clamp(1.5 - Math.abs(4 * v - 1))),
  ];
}

function Sparkline({ values }) {
  const width = 240;
  const height = 60;
  if (values.length < 2) {
    return <p style={styles.mutedText}>Collecting data…</p>;
  }
  const max = 100;
  const min = 0;
  const step = width / (values.length - 1);
  const points = values
    .map((v, i) => {
      const x = i * step;
      const y = height - ((v - min) / (max - min)) * height;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <polyline
        points={points}
        fill="none"
        stroke="#ff8a5c"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "#0a0b0d",
    color: "#e8e6e1",
    fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
    position: "relative",
    overflow: "hidden",
  },
  scanline: {
    position: "fixed",
    inset: 0,
    pointerEvents: "none",
    background:
      "repeating-linear-gradient(0deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px, transparent 1px, transparent 3px)",
    zIndex: 1,
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "20px 32px",
    borderBottom: "1px solid #1f2125",
    position: "relative",
    zIndex: 2,
  },
  headerLeft: { display: "flex", flexDirection: "column", gap: 2 },
  mark: { fontSize: 20, fontWeight: 700, letterSpacing: "0.08em" },
  markSub: { fontSize: 11, color: "#7a7f87", letterSpacing: "0.05em" },
  headerRight: {},
  statusPill: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    fontSize: 12,
    letterSpacing: "0.05em",
    textTransform: "uppercase",
    padding: "6px 12px",
    border: "1px solid #23262b",
    borderRadius: 20,
  },
  statusDot: { width: 8, height: 8, borderRadius: "50%" },
  main: {
    display: "grid",
    gridTemplateColumns: "1fr 320px",
    gap: 24,
    padding: 32,
    maxWidth: 1200,
    margin: "0 auto",
    position: "relative",
    zIndex: 2,
  },
  stage: { display: "flex", flexDirection: "column", gap: 16 },
  videoWrap: {
    position: "relative",
    width: "100%",
    aspectRatio: "4 / 3",
    background: "#111318",
    border: "1px solid #23262b",
    borderRadius: 4,
    overflow: "hidden",
  },
  video: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
    transform: "scaleX(-1)",
    transition: "opacity 0.2s ease",
  },
  // Same box + mirroring as the video, and sized to the captured frame, so
  // face_bbox coordinates line up with the face on screen.
  overlayCanvas: {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
    objectFit: "cover",
    transform: "scaleX(-1)",
    pointerEvents: "none",
  },
  placeholder: {
    position: "absolute",
    inset: 0,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: 12,
    textAlign: "center",
    padding: 32,
  },
  placeholderTitle: { fontSize: 16, fontWeight: 600, margin: 0 },
  placeholderSub: { fontSize: 13, color: "#7a7f87", margin: 0, maxWidth: 320 },
  placeholderNote: { fontSize: 11, color: "#facc15", margin: 0, maxWidth: 320 },
  primaryBtn: {
    marginTop: 8,
    padding: "10px 20px",
    background: "#ff8a5c",
    color: "#0a0b0d",
    border: "none",
    borderRadius: 3,
    fontFamily: "inherit",
    fontWeight: 600,
    fontSize: 13,
    letterSpacing: "0.03em",
    cursor: "pointer",
  },
  noFaceBanner: {
    position: "absolute",
    bottom: 12,
    left: "50%",
    transform: "translateX(-50%)",
    background: "rgba(10,11,13,0.85)",
    border: "1px solid #3a3d43",
    padding: "6px 14px",
    borderRadius: 20,
    fontSize: 12,
    color: "#c9cdd3",
  },
  errorBanner: {
    position: "absolute",
    top: 12,
    left: 12,
    right: 12,
    background: "rgba(255,92,92,0.12)",
    border: "1px solid rgba(255,92,92,0.4)",
    padding: "8px 12px",
    borderRadius: 3,
    fontSize: 12,
    color: "#ff9a9a",
  },
  controls: { display: "flex", gap: 10 },
  toggleBtn: {
    padding: "8px 16px",
    background: "transparent",
    color: "#e8e6e1",
    border: "1px solid #33363c",
    borderRadius: 3,
    fontFamily: "inherit",
    fontSize: 12,
    letterSpacing: "0.03em",
    cursor: "pointer",
  },
  stopBtn: {
    padding: "8px 16px",
    background: "transparent",
    color: "#ff8a8a",
    border: "1px solid #4a2a2a",
    borderRadius: 3,
    fontFamily: "inherit",
    fontSize: 12,
    letterSpacing: "0.03em",
    cursor: "pointer",
  },
  sidebar: { display: "flex", flexDirection: "column", gap: 16 },
  panel: {
    border: "1px solid #23262b",
    borderRadius: 4,
    padding: 16,
    background: "#111318",
  },
  panelTitle: {
    margin: "0 0 10px 0",
    fontSize: 11,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    color: "#7a7f87",
  },
  verdictRow: { display: "flex", alignItems: "baseline", justifyContent: "space-between" },
  verdictLabel: { fontSize: 24, fontWeight: 700, letterSpacing: "0.05em" },
  confidenceValue: { fontSize: 18, color: "#c9cdd3" },
  captionText: { margin: "8px 0 0 0", fontSize: 11, color: "#5c6068" },
  mutedText: { fontSize: 13, color: "#7a7f87", margin: 0 },
  stepsList: { margin: 0, paddingLeft: 18, fontSize: 12, color: "#c9cdd3", lineHeight: 1.7 },
  statsGrid: { display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 },
  statValue: { fontSize: 15, color: "#e8e6e1", fontVariantNumeric: "tabular-nums" },
};
