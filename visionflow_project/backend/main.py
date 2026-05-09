"""
VisionFlow — server.py (Fixed v2)
==================================
What changed from the original:
  1. _run_fastsam() was a stub returning empty tensors — NOW FULLY IMPLEMENTED
     using Ultralytics FastSAM directly. No TensorRT required at all.
     This also eliminates the tensorrt_cu13 vs torch+cu118 runtime conflict.
  2. multiprocessing replaced with ThreadPoolExecutor.
     multiprocessing has spawn/fork issues in WSL; ThreadPool achieves the
     same GPU-inference isolation without the spawn overhead.
  3. Warm-up pass on startup so first inference isn't slow.
  4. /health now returns GPU name for quick sanity checking.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import FastSAM as FastSAMModel

from fast_sam_prompter import FastSAMPrompter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("VisionFlow")

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH       = "FastSAM-s.pt"
FRAME_W, FRAME_H = 640, 480
RERUN_INTERVAL   = 16        # Run full FastSAM every N frames; optical flow in between
FASTSAM_IMGSZ    = 640
FASTSAM_CONF     = 0.4
FASTSAM_IOU      = 0.9
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

LK_PARAMS = dict(
    winSize         = (21, 21),
    maxLevel        = 3,
    criteria        = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    flags           = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
    minEigThreshold = 1e-4,
)


# ─── Tracker state ────────────────────────────────────────────────────────────
@dataclass
class TrackerState:
    prev_gray   : np.ndarray
    prev_mask   : np.ndarray   # uint8  0/255
    frame_count : int = 1


# ══════════════════════════════════════════════════════════════════════════════
#  VisionEngine — FastSAM inference via Ultralytics
# ══════════════════════════════════════════════════════════════════════════════

class VisionEngine:
    """
    Wraps Ultralytics FastSAM + FastSAMPrompter.
    Lives inside a ThreadPoolExecutor thread so GPU inference never stalls
    the asyncio event loop.
    """

    def __init__(self):
        log.info("Loading FastSAM-s on %s …", DEVICE)
        self.model    = FastSAMModel(MODEL_PATH)
        self.prompter = FastSAMPrompter(device=DEVICE)
        # Warm-up: eliminates JIT compilation delay on the first real frame
        dummy = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self._infer(dummy)
        log.info("FastSAM ready ✓  (device=%s)", DEVICE)

    # ── Point-guided segmentation ─────────────────────────────────────────────
    def segment_at_point(
        self,
        frame_bgr : np.ndarray,
        px        : int,
        py        : int,
    ) -> np.ndarray:
        """
        Full FastSAM forward pass + prompt selection.
        Returns a (FRAME_H, FRAME_W) uint8 mask (255 = selected, 0 = background).
        """
        boxes, masks = self._infer(frame_bgr)

        if masks is None or len(masks) == 0:
            log.debug("No instances detected.")
            return np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

        result = self.prompter.select_mask_at_point(boxes, masks, px, py)

        if result.mask is None:
            log.debug("Click (%d,%d) hit no mask.", px, py)
            return np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)

        log.info(
            "Mask selected — idx=%d  area=%d px²  candidates=%d",
            result.mask_index, result.area, result.candidates,
        )
        return FastSAMPrompter.mask_to_numpy(result.mask)

    # ── Raw inference ─────────────────────────────────────────────────────────
    def _infer(
        self,
        frame_bgr : np.ndarray,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        One FastSAM forward pass via Ultralytics.

        Returns
        -------
        boxes : (N, 4) float tensor [x1, y1, x2, y2] in pixel coords
        masks : (N, FRAME_H, FRAME_W) bool tensor
        """
        results = self.model(
            frame_bgr,
            device       = DEVICE,
            retina_masks = True,
            imgsz        = FASTSAM_IMGSZ,
            conf         = FASTSAM_CONF,
            iou          = FASTSAM_IOU,
            verbose      = False,
        )

        r = results[0]

        if r.masks is None or len(r.masks) == 0:
            empty_b = torch.zeros((0, 4), device=DEVICE)
            empty_m = torch.zeros((0, FRAME_H, FRAME_W), dtype=torch.bool, device=DEVICE)
            return empty_b, empty_m

        # boxes are already in original-image pixel space
        boxes = r.boxes.xyxy.to(DEVICE)          # (N, 4) float

        # masks from retina_masks=True are already at input resolution
        masks_raw = r.masks.data.to(DEVICE)      # (N, H_m, W_m) float 0/1

        # Resize to display resolution if they differ
        if masks_raw.shape[1:] != (FRAME_H, FRAME_W):
            masks_raw = F.interpolate(
                masks_raw.unsqueeze(1),            # (N, 1, H_m, W_m)
                size  = (FRAME_H, FRAME_W),
                mode  = "nearest",
            ).squeeze(1)                           # (N, FRAME_H, FRAME_W)

        return boxes, masks_raw.bool()


# ══════════════════════════════════════════════════════════════════════════════
#  Optical flow tracker
# ══════════════════════════════════════════════════════════════════════════════

def track_mask_optflow(
    prev_gray : np.ndarray,
    curr_gray : np.ndarray,
    prev_mask : np.ndarray,   # uint8 0/255
) -> np.ndarray:
    """
    Lucas-Kanade sparse optical flow mask propagation.
    Runs for the 15 frames between full FastSAM inference passes.

    Algorithm:
      1. Shi-Tomasi corners inside the mask → feature points P_t
      2. Pyramidal LK: P_t → P_{t+1}  (minimize SSD in local window)
      3. RANSAC affine from (P_t, P_{t+1}) → transform T
      4. cv2.warpAffine(prev_mask, T) → next mask estimate
    """
    pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners   = 200,
        qualityLevel = 0.01,
        minDistance  = 5,
        mask         = prev_mask,
    )

    if pts is None or len(pts) < 4:
        return prev_mask   # not enough features — keep prev mask

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts, None, **LK_PARAMS
    )

    good_prev = pts[status.ravel() == 1]
    good_next = next_pts[status.ravel() == 1]

    if len(good_prev) < 4:
        return prev_mask

    T, _ = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC)
    if T is None:
        return prev_mask

    return cv2.warpAffine(
        prev_mask, T, (FRAME_W, FRAME_H),
        flags      = cv2.INTER_NEAREST,
        borderMode = cv2.BORDER_CONSTANT,
        borderValue= 0,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  FrameProcessor — stateful per-session, runs in thread pool
# ══════════════════════════════════════════════════════════════════════════════

class FrameProcessor:
    def __init__(self, engine: VisionEngine):
        self.engine   = engine
        self.trackers : Dict[str, TrackerState] = {}

    def process(
        self,
        session_id  : str,
        frame_bytes : bytes,
        click       : Optional[list],
    ) -> dict:
        t0 = time.perf_counter()

        # Decode JPEG
        arr   = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"mask": "", "shape": [FRAME_H, FRAME_W], "latency_ms": 0.0}

        # Normalise to display resolution
        if frame.shape[:2] != (FRAME_H, FRAME_W):
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        state = self.trackers.get(session_id)

        run_full = (
            click is not None
            or state is None
            or state.frame_count % RERUN_INTERVAL == 0
        )

        if run_full and click is not None:
            # ── HEAVY PATH: full FastSAM + prompt selection ───────────────
            px, py  = int(click[0]), int(click[1])
            mask_np = self.engine.segment_at_point(frame, px, py)
            self.trackers[session_id] = TrackerState(
                prev_gray   = gray,
                prev_mask   = mask_np,
                frame_count = 1,
            )

        elif state is not None and np.any(state.prev_mask):
            # ── CHEAP PATH: Lucas-Kanade optical flow ─────────────────────
            mask_np         = track_mask_optflow(state.prev_gray, gray, state.prev_mask)
            state.prev_gray  = gray
            state.prev_mask  = mask_np
            state.frame_count += 1

        else:
            # No click yet — return empty mask, initialise tracker
            mask_np = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
            if state is None:
                self.trackers[session_id] = TrackerState(
                    prev_gray   = gray,
                    prev_mask   = mask_np,
                    frame_count = 1,
                )

        # Encode mask as PNG → base64
        _, buf   = cv2.imencode(".png", mask_np)
        mask_b64 = base64.b64encode(buf.tobytes()).decode()

        return {
            "mask"       : mask_b64,
            "shape"      : [FRAME_H, FRAME_W],
            "latency_ms" : round((time.perf_counter() - t0) * 1000, 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  FastAPI application
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="VisionFlow", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine    : Optional[VisionEngine]       = None
_processor : Optional[FrameProcessor]    = None
_executor  : Optional[ThreadPoolExecutor] = None


@app.on_event("startup")
def _startup():
    global _engine, _processor, _executor
    _engine    = VisionEngine()
    _processor = FrameProcessor(_engine)
    # 2 workers: one handles inference, one handles frame decoding overlap
    _executor  = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vf")
    log.info("✓ VisionFlow server ready")


@app.on_event("shutdown")
def _shutdown():
    if _executor:
        _executor.shutdown(wait=False)


@app.get("/health")
async def health():
    return {
        "status" : "ok",
        "device" : DEVICE,
        "model"  : MODEL_PATH,
        "gpu"    : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
                   if torch.cuda.is_available() else 0,
    }


@app.websocket("/ws/video")
async def video_ws(ws: WebSocket):
    """
    WebSocket: receives frames + clicks, returns binary masks.

    Client → Server (JSON):
        { "frame": "<base64 JPEG>", "click": [x,y]|null, "session_id": "uuid" }

    Server → Client (JSON):
        { "mask": "<base64 PNG>", "shape": [480,640], "latency_ms": 22.4 }
    """
    await ws.accept()
    session_id = None

    try:
        init       = await ws.receive_json()
        session_id = init.get("session_id", "default")
        log.info("Session [%s] connected.", session_id)

        loop = asyncio.get_event_loop()

        async for msg in ws.iter_json():
            frame_bytes = base64.b64decode(msg["frame"])
            click       = msg.get("click")

            result = await loop.run_in_executor(
                _executor,
                _processor.process,
                session_id,
                frame_bytes,
                click,
            )

            await ws.send_json(result)

    except WebSocketDisconnect:
        log.info("Session [%s] disconnected.", session_id)
    except Exception as exc:
        log.error("Session [%s] error: %s", session_id, exc, exc_info=True)
    finally:
        if session_id and _processor and session_id in _processor.trackers:
            del _processor.trackers[session_id]
            log.info("Session [%s] tracker cleaned up.", session_id)