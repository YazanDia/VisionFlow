import { useState, useEffect, useRef, useCallback } from "react";

/*
  VisionFlow — Frontend (Auto-Crop Cutout Version)
  ================================================
  UPDATED:
  • Clipboard now copies ONLY the extracted object pixels.
  • The background is completely transparent.
  • The final image is automatically cropped tightly to the 
    boundaries of the segmented object.
*/

const WS_URL = "ws://localhost:8000/ws/video";

const FRAME_W = 640;
const FRAME_H = 480;

const JPEG_QUALITY = 0.75;
const SEND_EVERY = 2;

// Used to render the visible overlay in the UI (green tint)
const OVERLAY_RGBA = [0, 255, 120, 110];

const uuid = () =>
  "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });

async function decodeMask(b64, w, h) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const oc = document.createElement("canvas");
      oc.width = w;
      oc.height = h;
      const ctx = oc.getContext("2d");
      ctx.drawImage(img, 0, 0, w, h);
      resolve(ctx.getImageData(0, 0, w, h));
    };
    img.src = "data:image/png;base64," + b64;
  });
}

export default function VisionFlow() {
  const videoRef = useRef(null);

  // Visible overlay canvas (shows the green mask)
  const canvasRef = useRef(null);

  // Hidden canvas to capture frames for the backend
  const hiddenRef = useRef(null);

  // Hidden canvas to build the transparent cutout
  const compositeRef = useRef(null);

  const wsRef = useRef(null);
  const animRef = useRef(null);
  const frameCount = useRef(0);
  const clickRef = useRef(null);
  const sessionId = useRef(uuid());
  const wasLive = useRef(false);

  const [status, setStatus] = useState("idle");
  const [latencyMs, setLatencyMs] = useState(null);
  const [maskOn, setMaskOn] = useState(false);
  const [copyMsg, setCopyMsg] = useState("");
  const [crossPos, setCrossPos] = useState({ x: 0, y: 0 });
  const [showCross, setShowCross] = useState(false);

  /*
    ─────────────────────────────────────────────────────────────
    COMPOSITE BUILDER (AUTO-CROP CUTOUT MODE)
    Extracts the raw pixels and dynamically crops the canvas 
    to exactly fit the segmented object with a transparent background.
    ─────────────────────────────────────────────────────────────
  */
  const buildComposite = useCallback(() => {
    const video = videoRef.current;
    const overlayCanvas = canvasRef.current;
    const composite = compositeRef.current;

    if (!video || !overlayCanvas || !composite) return null;

    const ctx = composite.getContext("2d");
    const overlayCtx = overlayCanvas.getContext("2d");

    // 1. Draw raw video to extract its real colors
    ctx.clearRect(0, 0, FRAME_W, FRAME_H);
    ctx.drawImage(video, 0, 0, FRAME_W, FRAME_H);

    const videoData = ctx.getImageData(0, 0, FRAME_W, FRAME_H);
    const maskData = overlayCtx.getImageData(0, 0, FRAME_W, FRAME_H);
    const cutoutData = new ImageData(FRAME_W, FRAME_H);

    // 2. Track the boundaries of the object for cropping
    let minX = FRAME_W,
      minY = FRAME_H,
      maxX = 0,
      maxY = 0;
    let hasVisiblePixels = false;

    // 3. Loop through pixels using X and Y coordinates
    for (let y = 0; y < FRAME_H; y++) {
      for (let x = 0; x < FRAME_W; x++) {
        const i = (y * FRAME_W + x) * 4;

        if (maskData.data[i + 3] > 0) {
          // If mask is present
          // Copy video colors directly
          cutoutData.data[i] = videoData.data[i];
          cutoutData.data[i + 1] = videoData.data[i + 1];
          cutoutData.data[i + 2] = videoData.data[i + 2];
          cutoutData.data[i + 3] = 255; // Make pixel fully opaque

          // Update bounding box
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;

          hasVisiblePixels = true;
        } else {
          // Transparent background
          cutoutData.data[i + 3] = 0;
        }
      }
    }

    if (!hasVisiblePixels) return null; // Nothing was segmented

    // 4. Put the full uncropped cutout back onto the hidden canvas temporarily
    ctx.putImageData(cutoutData, 0, 0);

    // 5. Calculate the exact width and height of the isolated object
    const cropW = maxX - minX + 1;
    const cropH = maxY - minY + 1;

    // 6. Create a final, dynamically sized canvas just for the exported image
    const finalCanvas = document.createElement("canvas");
    finalCanvas.width = cropW;
    finalCanvas.height = cropH;
    const finalCtx = finalCanvas.getContext("2d");

    // 7. Draw ONLY the cropped region to the final canvas
    finalCtx.drawImage(
      composite,
      minX,
      minY,
      cropW,
      cropH, // Source coordinates (bounding box)
      0,
      0,
      cropW,
      cropH, // Destination coordinates
    );

    return finalCanvas;
  }, []);

  /*
    ─────────────────────────────────────────────────────────────
    COPY TO CLIPBOARD
    ─────────────────────────────────────────────────────────────
  */
  const copyFrame = useCallback(async () => {
    try {
      const composite = buildComposite();
      if (!composite) return;

      const blob = await new Promise((resolve) =>
        composite.toBlob(resolve, "image/png"),
      );

      await navigator.clipboard.write([
        new ClipboardItem({
          "image/png": blob,
        }),
      ]);

      setCopyMsg("COPIED!");
      setTimeout(() => setCopyMsg(""), 2000);
    } catch (err) {
      console.error(err);
      setCopyMsg("COPY FAILED");
      setTimeout(() => setCopyMsg(""), 3000);
    }
  }, [buildComposite]);

  /*
    ─────────────────────────────────────────────────────────────
    SAVE PNG
    ─────────────────────────────────────────────────────────────
  */
  const saveFrame = useCallback(() => {
    const composite = buildComposite();
    if (!composite) return;

    const a = document.createElement("a");
    a.href = composite.toDataURL("image/png");
    a.download = `visionflow_cutout_${Date.now()}.png`;
    a.click();
  }, [buildComposite]);

  /*
    ─────────────────────────────────────────────────────────────
    CTRL+C HANDLER
    ─────────────────────────────────────────────────────────────
  */
  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "c") {
        if (document.activeElement.tagName === "INPUT") return;
        e.preventDefault();
        copyFrame();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [copyFrame]);

  /*
    ─────────────────────────────────────────────────────────────
    WEBSOCKET
    ─────────────────────────────────────────────────────────────
  */
  const connectWS = useCallback(() => {
    setStatus("connecting");
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      ws.send(JSON.stringify({ session_id: sessionId.current }));
      wasLive.current = true;
      setStatus("live");
    };

    ws.onmessage = async (evt) => {
      const data = JSON.parse(evt.data);

      if (data.latency_ms !== undefined) {
        setLatencyMs(data.latency_ms.toFixed(1));
      }

      if (!data.mask || !canvasRef.current) return;

      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      const [H, W] = data.shape;

      const imgData = await decodeMask(data.mask, W, H);
      const px = imgData.data;

      let hasMask = false;

      for (let i = 0; i < px.length; i += 4) {
        if (px[i] > 128) {
          px[i] = OVERLAY_RGBA[0];
          px[i + 1] = OVERLAY_RGBA[1];
          px[i + 2] = OVERLAY_RGBA[2];
          px[i + 3] = OVERLAY_RGBA[3];
          hasMask = true;
        } else {
          px[i + 3] = 0;
        }
      }

      ctx.clearRect(0, 0, W, H);
      ctx.putImageData(imgData, 0, 0);
      setMaskOn(hasMask);
    };

    ws.onerror = () => setStatus("error");

    ws.onclose = () => {
      setStatus("idle");
      if (wasLive.current) {
        setTimeout(() => connectWS(), 2000);
      }
    };

    wsRef.current = ws;
  }, []);

  /*
    ─────────────────────────────────────────────────────────────
    CAMERA / VIDEO INPUT
    (Note: Change 'srcObject' to 'src' if using a local MP4 file)
    ─────────────────────────────────────────────────────────────
  */
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: FRAME_W, height: FRAME_H, facingMode: "user" },
        audio: false,
      });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    } catch (err) {
      console.error(err);
      setStatus("error");
    }
  }, []);

  /*
    ─────────────────────────────────────────────────────────────
    CAPTURE LOOP
    ─────────────────────────────────────────────────────────────
  */
  const captureLoop = useCallback(() => {
    const video = videoRef.current;
    const hidden = hiddenRef.current;
    const ws = wsRef.current;

    if (video && hidden && ws && ws.readyState === WebSocket.OPEN) {
      frameCount.current++;

      if (frameCount.current % SEND_EVERY === 0) {
        const ctx = hidden.getContext("2d");
        ctx.drawImage(video, 0, 0, FRAME_W, FRAME_H);

        hidden.toBlob(
          (blob) => {
            const reader = new FileReader();
            reader.onloadend = () => {
              const b64 = reader.result.split(",")[1];
              const click = clickRef.current;
              clickRef.current = null;

              ws.send(
                JSON.stringify({
                  frame: b64,
                  click,
                  session_id: sessionId.current,
                }),
              );
            };
            reader.readAsDataURL(blob);
          },
          "image/jpeg",
          JPEG_QUALITY,
        );
      }
    }

    animRef.current = requestAnimationFrame(captureLoop);
  }, []);

  /*
    ─────────────────────────────────────────────────────────────
    STARTUP
    ─────────────────────────────────────────────────────────────
  */
  useEffect(() => {
    startCamera();
    connectWS();
    animRef.current = requestAnimationFrame(captureLoop);

    return () => {
      cancelAnimationFrame(animRef.current);
      wsRef.current?.close();
      videoRef.current?.srcObject?.getTracks().forEach((t) => t.stop());
    };
  }, [startCamera, connectWS, captureLoop]);

  /*
    ─────────────────────────────────────────────────────────────
    CLICK
    ─────────────────────────────────────────────────────────────
  */
  const handleClick = useCallback((e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const scaleX = FRAME_W / rect.width;
    const scaleY = FRAME_H / rect.height;

    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    clickRef.current = [x, y];
    setMaskOn(false);
  }, []);

  /*
    ─────────────────────────────────────────────────────────────
    STATUS UI
    ─────────────────────────────────────────────────────────────
  */
  const STATUS = {
    idle: { label: "STANDBY", dot: "#71717a", text: "#a1a1aa" },
    connecting: { label: "CONNECTING", dot: "#f59e0b", text: "#f59e0b" },
    live: { label: "LIVE", dot: "#10b981", text: "#10b981" },
    error: { label: "ERROR", dot: "#ef4444", text: "#f87171" },
  }[status];

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        background: "#09090b",
        fontFamily: "JetBrains Mono, monospace",
        color: "white",
      }}
    >
      <div style={{ width: FRAME_W }}>
        {/* HEADER */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: 12,
          }}
        >
          <div>
            VISION<span style={{ color: "#22c55e" }}>FLOW</span>
          </div>

          <div style={{ color: STATUS.text }}>
            ● {STATUS.label}
            {latencyMs && (
              <span style={{ marginLeft: 12 }}>{latencyMs} ms</span>
            )}
          </div>
        </div>

        {/* VIDEO STACK */}
        <div
          style={{
            position: "relative",
            width: FRAME_W,
            height: FRAME_H,
            border: "1px solid #27272a",
            overflow: "hidden",
          }}
        >
          <video
            ref={videoRef}
            muted
            playsInline
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "cover",
            }}
          />

          <canvas
            ref={canvasRef}
            width={FRAME_W}
            height={FRAME_H}
            onClick={handleClick}
            onMouseMove={(e) => {
              const r = e.currentTarget.getBoundingClientRect();
              setCrossPos({
                x: e.clientX - r.left,
                y: e.clientY - r.top,
              });
              setShowCross(true);
            }}
            onMouseLeave={() => setShowCross(false)}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              cursor: "crosshair",
            }}
          />

          {showCross && (
            <div
              style={{
                position: "absolute",
                left: crossPos.x - 10,
                top: crossPos.y - 10,
                width: 20,
                height: 20,
                border: "1px solid #22c55e",
                pointerEvents: "none",
              }}
            />
          )}
        </div>

        {/* BUTTONS */}
        <div
          style={{
            display: "flex",
            gap: 10,
            marginTop: 12,
          }}
        >
          <button
            onClick={copyFrame}
            style={{
              flex: 1,
              padding: 10,
              background: "#18181b",
              border: "1px solid #27272a",
              color: "#22c55e",
              cursor: "pointer",
            }}
          >
            {copyMsg || "COPY CUTOUT (CTRL+C)"}
          </button>

          <button
            onClick={saveFrame}
            style={{
              flex: 1,
              padding: 10,
              background: "#18181b",
              border: "1px solid #27272a",
              color: "#22c55e",
              cursor: "pointer",
            }}
          >
            SAVE CUTOUT AS PNG
          </button>
        </div>

        {/* HIDDEN CANVASES */}
        <canvas
          ref={hiddenRef}
          width={FRAME_W}
          height={FRAME_H}
          style={{ display: "none" }}
        />

        <canvas
          ref={compositeRef}
          width={FRAME_W}
          height={FRAME_H}
          style={{ display: "none" }}
        />
      </div>
    </div>
  );
}
