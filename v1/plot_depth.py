#!/usr/bin/env python3
"""
plot_depth.py
=============
Reads depth_out.hex produced by the Verilog simulation and visualises it
alongside the input images and ground-truth disparity.

Layout (2 rows):
  Row 1: Left Image  |  Right Image
  Row 2: GT Disparity  |  Simulated Depth Map  (+ accuracy stats)

All hex files are resolved relative to the v1/ directory by default:
    data/left_img.hex
    data/right_img.hex
    data/gt_disparity.hex
    output/depth_out.hex   (written by the simulation)

Usage:
    uv run python v1/plot_depth.py [--out v1/output/depth_map.png]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# v1 layout
# ---------------------------------------------------------------------------

V1_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_hex(path: Path, width: int, height: int) -> np.ndarray:
    """Load a hex file (one value per line) into a 2-D uint8 array.

    If the file contains fewer values than width*height (because the SAD
    pipeline primes for the first few rows/columns), the array is zero-padded
    at the start so the valid output is bottom-right aligned — matching the
    spatial offset introduced by the pipeline latency.
    """
    values = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(int(line, 16))
    arr = np.array(values, dtype=np.uint8)
    expected = width * height
    if len(arr) < expected:
        pad = np.zeros(expected - len(arr), dtype=np.uint8)
        arr = np.concatenate([pad, arr])
    else:
        arr = arr[:expected]
    return arr.reshape(height, width)


def resolve(arg: str) -> Path:
    """Return an absolute path: use arg as-is if absolute, else relative to v1/."""
    p = Path(arg)
    return p if p.is_absolute() else V1_ROOT / p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise stereo depth map")
    parser.add_argument("--depth",    default="output/depth_out.hex",
                        help="Depth hex file (default: output/depth_out.hex)")
    parser.add_argument("--gt",       default="data/gt_disparity.hex",
                        help="Ground-truth hex file (default: data/gt_disparity.hex)")
    parser.add_argument("--left",     default="data/left_img.hex",
                        help="Left image hex file (default: data/left_img.hex)")
    parser.add_argument("--right",    default="data/right_img.hex",
                        help="Right image hex file (default: data/right_img.hex)")
    parser.add_argument("--out",      default="output/depth_map.png",
                        help="Output PNG (default: output/depth_map.png)")
    parser.add_argument("--width",    type=int, default=512)
    parser.add_argument("--height",   type=int, default=512)
    parser.add_argument("--max-disp", type=int, default=16,
                        help="Maximum disparity for colour scale (default 16)")
    args = parser.parse_args()

    depth_path = resolve(args.depth)
    gt_path    = resolve(args.gt)
    left_path  = resolve(args.left)
    right_path = resolve(args.right)

    W, H = args.width, args.height

    # ---- Load files --------------------------------------------------------
    if not depth_path.exists():
        sys.exit(f"ERROR: {depth_path} not found. Run the simulation first.")

    depth = load_hex(depth_path, W, H)
    print(f"Loaded depth:  {depth_path}  (min={depth.min()}, max={depth.max()})")

    has_gt    = gt_path.exists()
    has_left  = left_path.exists()
    has_right = right_path.exists()

    left_img  = load_hex(left_path,  W, H) if has_left  else np.zeros((H, W), dtype=np.uint8)
    right_img = load_hex(right_path, W, H) if has_right else np.zeros((H, W), dtype=np.uint8)
    if has_left:
        print(f"Loaded left:   {left_path}")
    if has_right:
        print(f"Loaded right:  {right_path}")
    if has_gt:
        gt = load_hex(gt_path, W, H)
        print(f"Loaded GT:     {gt_path}")
    else:
        gt = np.zeros((H, W), dtype=np.uint8)

    # ---- Accuracy ----------------------------------------------------------
    acc_str  = ""
    rmse_str = ""
    if has_gt:
        valid_mask = gt > 0
        if valid_mask.any():
            err      = np.abs(depth.astype(int) - gt.astype(int))
            acc1     = (err[valid_mask] <= 1).mean() * 100
            rmse     = np.sqrt((err[valid_mask] ** 2).mean())
            acc_str  = f"Accuracy (|err|≤1): {acc1:.1f}%"
            rmse_str = f"RMSE: {rmse:.2f} px"
            print(f"Accuracy (|err|≤1, foreground only): {acc1:.1f}%")
            print(f"RMSE (foreground only):               {rmse:.2f} px")

    # ---- Layout: 2 rows × 2 cols ------------------------------------------
    #   [0,0] Left image      [0,1] Right image
    #   [1,0] GT disparity    [1,1] Simulated depth
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.08)

    # Row 0, col 0 — Left image
    ax00 = fig.add_subplot(gs[0, 0])
    if has_left:
        ax00.imshow(left_img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax00.set_title("Left Image", fontsize=12)
    else:
        ax00.text(0.5, 0.5, "left_img.hex\nnot found",
                  ha="center", va="center", transform=ax00.transAxes)
        ax00.set_title("Left Image (missing)", fontsize=12)
    ax00.axis("off")

    # Row 0, col 1 — Right image
    ax01 = fig.add_subplot(gs[0, 1])
    if has_right:
        ax01.imshow(right_img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax01.set_title("Right Image", fontsize=12)
    else:
        ax01.text(0.5, 0.5, "right_img.hex\nnot found",
                  ha="center", va="center", transform=ax01.transAxes)
        ax01.set_title("Right Image (missing)", fontsize=12)
    ax01.axis("off")

    # Row 1, col 0 — Ground-truth disparity
    ax10 = fig.add_subplot(gs[1, 0])
    if has_gt:
        im_gt = ax10.imshow(gt, cmap="plasma", vmin=0, vmax=args.max_disp,
                            interpolation="nearest")
        ax10.set_title("Ground-Truth Disparity", fontsize=12)
        plt.colorbar(im_gt, ax=ax10, fraction=0.046, pad=0.04, label="Disparity (px)")
    else:
        ax10.text(0.5, 0.5, "gt_disparity.hex\nnot found",
                  ha="center", va="center", transform=ax10.transAxes)
        ax10.set_title("Ground-Truth Disparity (missing)", fontsize=12)
    ax10.axis("off")

    # Row 1, col 1 — Simulated depth
    ax11 = fig.add_subplot(gs[1, 1])
    im_d = ax11.imshow(depth, cmap="plasma", vmin=0, vmax=args.max_disp,
                       interpolation="nearest")
    title = "Simulated Depth Map"
    if acc_str:
        title += f"\n{acc_str}   {rmse_str}"
    ax11.set_title(title, fontsize=12)
    ax11.axis("off")
    plt.colorbar(im_d, ax=ax11, fraction=0.046, pad=0.04, label="Disparity (px)")

    # ---- Save & show -------------------------------------------------------
    plt.suptitle("Stereo Depth Estimation — SAD 5×5", fontsize=14)

    out_path = Path(args.out) if Path(args.out).is_absolute() else V1_ROOT / args.out
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
