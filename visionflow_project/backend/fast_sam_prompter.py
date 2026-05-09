"""
VisionFlow — Task 2: Prompt-Guided Selection Algorithm
=======================================================
Academic Implementation of FastSAMPrompter

Module: fast_sam_prompter.py
Author: VisionFlow / CS Lead
Difficulty: Graduate-level (CSCI-6xx Real-Time CV)

ACADEMIC CONTEXT
----------------
FastSAM decomposes image segmentation into two stages:
  1. All-instance mask generation via YOLOv8-seg backbone
     → N binary masks  M ∈ {0,1}^(H×W)  and their bounding boxes.
  2. Prompt-guided selection: given a user click p=(x,y), determine WHICH
     of the N masks the user intended to interact with.

This module implements Stage 2 from mathematical first principles.
No Python for-loops are used. All operations are O(N) in the number of
masks and O(1) per mask via fully vectorised tensor algebra.

MATHEMATICAL FOUNDATIONS
-------------------------
Given:
  • N  — number of detected instances
  • B  ∈ ℝ^(N×4) — bounding boxes  [x1, y1, x2, y2]  (pixel coords)
  • M  ∈ {0,1}^(N×H×W) — binary mask stack (one binary image per instance)
  • p  = (px, py) — user click in pixel coords

Algorithm:
  ① Spatial filter  φ : B × p → {True,False}^N
       φ_i = (B_i[0] ≤ px ≤ B_i[2]) ∧ (B_i[1] ≤ py ≤ B_i[3])
       Implemented via broadcast subtraction and sign checks — O(N).

  ② Pixel-level containment  χ : M × p → {True,False}^N
       χ_i = M_i[py, px]   (direct index into the binary mask volume)
       Batch-indexed in one tensor operation — O(N).

  ③ Area computation  α : M → ℝ^N
       α_i = ||M_i||_1  (L1-norm ≡ sum of 1-bits = pixel count)
       torch.sum(M, dim=(1,2)) collapses H×W in one CUDA kernel — O(N·H·W)
       amortised over the batch, not per-mask.

  ④ Conflict resolution  argmin(α_i | χ_i ∧ φ_i)
       Among all masks that contain p, return the one with the smallest
       area. Smallest area ≡ frontmost/smallest foreground object, which
       matches perceptual user intent (Gestalt figure–ground).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

log = logging.getLogger("VisionFlow.Prompter")


# ─── Data containers ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SegmentationResult:
    """
    Return type for FastSAMPrompter.select_mask_at_point().

    Attributes
    ----------
    mask : torch.Tensor | None
        Shape (H, W), dtype bool.  The selected binary mask, or None if
        no mask contains the query point.
    mask_index : int | None
        Index into the original masks/boxes arrays; useful for debugging.
    area : int | None
        Area (px²) of the selected mask — the argmin of α.
    candidates : int
        Number of masks that survived both the bbox filter and pixel test.
    """
    mask       : Optional[torch.Tensor]
    mask_index : Optional[int]
    area       : Optional[int]
    candidates : int


# ─── Core class ───────────────────────────────────────────────────────────────

class FastSAMPrompter:
    """
    Prompt-Guided Mask Selection from scratch.

    This class receives the raw outputs of the FastSAM/YOLOv8-seg backbone
    and applies a click point (x, y) to select the most appropriate mask
    using purely vectorised linear algebra — no Python loops.

    Parameters
    ----------
    device : str
        Torch device string, e.g. "cuda:0" or "cpu".
    iou_threshold : float
        Reserved for future ensemble post-processing. Not used in the
        primary selection path.

    Usage
    -----
    >>> prompter = FastSAMPrompter(device="cuda")
    >>> result   = prompter.select_mask_at_point(boxes, masks, px=320, py=240)
    >>> if result.mask is not None:
    ...     overlay = result.mask.cpu().numpy().astype(np.uint8) * 255
    """

    def __init__(self, device: str = "cuda", iou_threshold: float = 0.7):
        self.device        = torch.device(device if torch.cuda.is_available() else "cpu")
        self.iou_threshold = iou_threshold
        log.info("FastSAMPrompter initialised on device: %s", self.device)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def select_mask_at_point(
        self,
        boxes  : torch.Tensor,   # (N, 4)  float  [x1, y1, x2, y2]
        masks  : torch.Tensor,   # (N, H, W) bool or uint8
        px     : int,
        py     : int,
    ) -> SegmentationResult:
        """
        Select the best-fitting mask for user click (px, py).

        Pipeline
        --------
        1.  _bbox_filter(B, p)   → boolean mask of length N  [vectorised]
        2.  _pixel_containment(M, p, filter) → boolean mask of length N [vectorised]
        3.  _resolve_conflict(M, combined_mask) → argmin area [vectorised]
        4.  Package result.

        Time Complexity
        ---------------
        Phase 1 (bbox filter)      : O(N)         — N comparisons, broadcast
        Phase 2 (pixel test)       : O(N)         — N direct tensor index ops
        Phase 3 (area + argmin)    : O(N × H × W) — one fused CUDA kernel
        Overall                    : O(N × H × W) — dominated by mask memory
        No Python for-loop executes at any stage.

        Parameters
        ----------
        boxes  : Tensor (N, 4) — float bounding boxes [x1, y1, x2, y2]
        masks  : Tensor (N, H, W) — binary masks (bool or uint8)
        px, py : int — pixel coordinates of the user's click

        Returns
        -------
        SegmentationResult
        """
        # ── Input validation & device placement ───────────────────────────────
        boxes = boxes.to(self.device).float()   # Ensure float for comparisons
        masks = masks.to(self.device).bool()    # Normalise to bool tensor

        N, H, W = masks.shape

        if not (0 <= px < W and 0 <= py < H):
            log.warning("Click (%d, %d) is outside frame bounds (%dx%d).", px, py, W, H)
            return SegmentationResult(None, None, None, 0)

        # ── Phase 1: Bounding-box pre-filter ──────────────────────────────────
        # φ — spatial boolean selector vector of shape (N,)
        #
        # Mathematical form:
        #   φ_i = [x1_i ≤ px] ∧ [px ≤ x2_i] ∧ [y1_i ≤ py] ∧ [py ≤ y2_i]
        #
        # Implementation:
        #   We broadcast the scalar (px, py) against the (N,4) box matrix.
        #   All N comparisons happen simultaneously in a single CUDA kernel.
        #   This eliminates the O(N) Python loop that naive implementations use.
        bbox_filter = self._bbox_filter(boxes, px, py)

        n_bbox_pass = bbox_filter.sum().item()
        if n_bbox_pass == 0:
            log.debug("Click (%d,%d) outside all bounding boxes.", px, py)
            return SegmentationResult(None, None, None, 0)

        log.debug("Bbox filter: %d/%d masks passed.", n_bbox_pass, N)

        # ── Phase 2: Pixel-level containment check ────────────────────────────
        # χ — pixel containment boolean vector of shape (N,)
        #
        # Mathematical form:
        #   χ_i = M_i[py, px]
        #
        # Implementation:
        #   masks[:, py, px] is a single vectorised index expression.
        #   PyTorch executes this as one gather() on the GPU — not N lookups.
        #   We combine χ with φ via bitwise AND (&) to get the candidate set.
        pixel_filter = self._pixel_containment(masks, px, py)
        combined     = bbox_filter & pixel_filter         # φ ∧ χ  shape (N,)

        n_candidates = combined.sum().item()
        if n_candidates == 0:
            log.debug("No mask pixel contains click (%d,%d).", px, py)
            return SegmentationResult(None, None, None, 0)

        log.debug("Pixel containment: %d candidates remain.", n_candidates)

        # ── Phase 3: Conflict resolution via area argmin ──────────────────────
        # α_i = Σ_{h,w} M_i[h, w]   (L1-norm of binary mask = pixel count)
        #
        # If n_candidates == 1, argmin is trivial.
        # If n_candidates  > 1, we select the SMALLEST mask.
        #
        # Rationale (perceptual / Gestalt):
        #   In a scene with overlapping objects (e.g., hand in front of body),
        #   the click lands on all overlapping masks simultaneously.
        #   The foreground object always has SMALLER area than background layers.
        #   ⟹  argmin(area) ≡ topmost foreground object.
        #
        # Implementation:
        #   torch.sum(masks, dim=(1,2)) collapses the H×W dimensions across
        #   ALL N masks in a single fused reduction kernel.
        #   We then mask non-candidates with +∞ before calling argmin().
        best_idx, best_area = self._resolve_conflict(masks, combined)

        selected_mask = masks[best_idx]   # (H, W) bool

        log.debug(
            "Selected mask idx=%d  area=%d px²  from %d candidates.",
            best_idx, best_area, n_candidates
        )

        return SegmentationResult(
            mask       = selected_mask,
            mask_index = best_idx,
            area       = best_area,
            candidates = n_candidates,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Private vectorised subroutines
    # ──────────────────────────────────────────────────────────────────────────

    def _bbox_filter(
        self,
        boxes : torch.Tensor,   # (N, 4) float  [x1, y1, x2, y2]
        px    : int,
        py    : int,
    ) -> torch.Tensor:           # (N,) bool
        """
        Phase 1 — Vectorised bounding-box spatial filter.

        Linear Algebra Detail
        ---------------------
        Let  p = [px, py]  (broadcast scalar pair).
        Let  B  be the (N,4) box matrix.

        We slice B into column vectors:
          x1 = B[:, 0]   shape (N,)
          y1 = B[:, 1]
          x2 = B[:, 2]
          y2 = B[:, 3]

        φ = (x1 ≤ px) & (px ≤ x2) & (y1 ≤ py) & (py ≤ y2)

        The four comparisons each produce an (N,) bool tensor.
        The & operator chains them via bitwise AND — still (N,) bool.
        PyTorch fuses these into ONE GPU kernel pass for cache efficiency.
        """
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # Four broadcasts against scalars — O(N) each, fused by autograd
        in_x = (x1 <= px) & (px <= x2)   # (N,) bool
        in_y = (y1 <= py) & (py <= y2)   # (N,) bool

        return in_x & in_y               # φ  ∈  {True,False}^N

    def _pixel_containment(
        self,
        masks  : torch.Tensor,   # (N, H, W) bool
        px     : int,
        py     : int,
    ) -> torch.Tensor:            # (N,) bool
        """
        Phase 2 — Vectorised pixel-level containment test.

        Linear Algebra Detail
        ---------------------
        M[:, py, px]  is a batch index into the 3-D mask tensor.
        This is equivalent to extracting column vector:
          χ_i = M_i[py, px]   ∀ i ∈ {0…N-1}

        PyTorch compiles this to a single gather() CUDA call — NOT a loop.
        The result is an (N,) bool tensor.

        Note: We intentionally check the pixel AFTER the bbox filter has
        already shortlisted candidates.  However, we compute χ over ALL N
        masks and combine with φ via bitwise AND — letting CUDA parallelise
        both phases rather than creating Python branches.
        """
        # masks[:, py, px]  →  advanced indexing  →  (N,) bool
        return masks[:, py, px]

    def _resolve_conflict(
        self,
        masks     : torch.Tensor,   # (N, H, W) bool
        candidate : torch.Tensor,   # (N,) bool
    ) -> Tuple[int, int]:
        """
        Phase 3 — Area-minimum conflict resolution.

        Linear Algebra Detail
        ---------------------
        α_i = ||M_i||_1 = Σ_{h=0}^{H-1} Σ_{w=0}^{W-1} M_i[h,w]

        torch.sum(masks, dim=(1,2)):
          Collapses dims 1 (height) and 2 (width) via parallel reduction.
          Result: α ∈ ℝ^N   (pixel count for each mask)

        Masking non-candidates:
          We set α_i = +∞  for all  i  where  candidate_i = False.
          This ensures non-candidates are never selected by argmin.
          Uses torch.where() — a vectorised conditional replace (no if/else).

        argmin(α):
          Returns the index of the smallest surviving area.
          Equivalent to  i* = argmin_{i: χ_i ∧ φ_i}  α_i

        Returns
        -------
        (best_index, best_area_px²)
        """
        # Compute L1-norm (area) for ALL N masks simultaneously
        # α shape: (N,)
        areas = masks.float().sum(dim=(1, 2))   # Σ pixels — CUDA parallel reduction

        # Set non-candidate areas to +∞ so argmin ignores them
        # torch.where is a fused ternary — one GPU pass, no Python branching
        inf_val  = torch.tensor(float("inf"), device=self.device)
        filtered = torch.where(candidate, areas, inf_val)   # (N,) float

        # argmin over the filtered area vector
        best_idx  = int(filtered.argmin().item())
        best_area = int(areas[best_idx].item())

        return best_idx, best_area

    # ──────────────────────────────────────────────────────────────────────────
    # Utility: numpy bridge (for OpenCV / frontend serialisation)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
        """
        Convert a (H, W) bool mask tensor to a uint8 numpy array (0/255).
        Used before passing to OpenCV or base64 encoding for WebSocket.
        """
        return mask.cpu().numpy().astype(np.uint8) * 255

    @staticmethod
    def mask_to_rgba_overlay(
        mask  : torch.Tensor,
        color : Tuple[int, int, int, int] = (0, 255, 0, 102),
    ) -> np.ndarray:
        """
        Convert binary mask to RGBA overlay array for canvas rendering.

        Parameters
        ----------
        mask  : (H, W) bool tensor
        color : (R, G, B, A) tuple; default green at α=0.4 (102/255)

        Returns
        -------
        np.ndarray of shape (H, W, 4) uint8
        """
        H, W      = mask.shape
        overlay   = np.zeros((H, W, 4), dtype=np.uint8)
        mask_np   = mask.cpu().numpy()
        overlay[mask_np, :] = color
        return overlay


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    torch.manual_seed(0)
    N, H, W = 12, 480, 640
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    # Synthesise N random bounding boxes + binary masks
    x1 = torch.randint(0, W // 2, (N,), device=device).float()
    y1 = torch.randint(0, H // 2, (N,), device=device).float()
    x2 = (x1 + torch.randint(50, W // 2, (N,), device=device)).clamp(max=W - 1)
    y2 = (y1 + torch.randint(50, H // 2, (N,), device=device)).clamp(max=H - 1)
    boxes = torch.stack([x1, y1, x2, y2], dim=1)   # (N, 4)

    masks = torch.zeros(N, H, W, dtype=torch.bool, device=device)
    for i in range(N):
        bx1, by1, bx2, by2 = int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])
        masks[i, by1:by2, bx1:bx2] = True   # Rectangular synthetic masks

    prompter = FastSAMPrompter(device=device)
    result   = prompter.select_mask_at_point(boxes, masks, px=320, py=240)

    print("\n=== FastSAMPrompter Self-Test ===")
    print(f"  Device      : {device}")
    print(f"  Masks tested: {N}")
    print(f"  Candidates  : {result.candidates}")
    print(f"  Selected idx: {result.mask_index}")
    print(f"  Mask area   : {result.area} px²")
    print(f"  Mask shape  : {result.mask.shape if result.mask is not None else 'None'}")
    print("=================================\n")
