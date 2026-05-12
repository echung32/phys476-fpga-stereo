"""Render a grid showing all 16 feature map channels for one (or more) frames.

Each cell in the 4×4 grid shows one channel, independently normalized.
A 17th panel (bottom-right of a 5-row layout) shows the L2-norm heatmap.
Left-most column shows the input grayscale.

Usage:
    uv run python -m v2.hls4ml.render_channel_grid \
        --f32-dir v2/hls4ml/parallel_sim \
        --npz-dir v2/data/perspective_stereo \
        --frame   frame00000_yaw+000_pitch+00 \
        --output  v2/hls4ml/channel_grid.png

    # render all frames to a video grid
    uv run python -m v2.hls4ml.render_channel_grid \
        --f32-dir v2/hls4ml/parallel_sim \
        --npz-dir v2/data/perspective_stereo \
        --output  v2/hls4ml/channel_grid.mp4 \
        --fps 10
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

H, W, N_CH = 160, 288, 16
N_PIX = H * W
CMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "Blues",
    "Greens",
    "Oranges",
    "Purples",
    "Reds",
    "YlOrBr",
    "PuBuGn",
    "RdPu",
    "GnBu",
    "OrRd",
    "PuRd",
]


def load_f32(path: pathlib.Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    if len(lines) != N_PIX:
        raise ValueError(f"{path}: expected {N_PIX} lines, got {len(lines)}")
    data = np.array([list(map(float, l.split())) for l in lines], dtype=np.float32)
    return data.reshape(H, W, N_CH)


def load_input(npz_path: pathlib.Path) -> np.ndarray | None:
    try:
        d = np.load(npz_path)
        img = d["left_image"]
        ih, iw = img.shape[:2]
        cy, cx = ih // 2, iw // 2
        crop = img[cy - H // 2 : cy + H // 2, cx - W // 2 : cx + W // 2, 0]
        lo, hi = crop.min(), crop.max()
        if hi - lo < 1e-6:
            return np.zeros((H, W), dtype=np.float32)
        return ((crop - lo) / (hi - lo)).astype(np.float32)
    except Exception as e:
        print(f"  WARNING: {npz_path}: {e}")
        return None


def channel_to_rgb(ch: np.ndarray, cmap_name: str) -> np.ndarray:
    """Normalize a (H,W) float array → (H,W,3) uint8 with given colormap."""
    lo, hi = ch.min(), ch.max()
    if hi - lo < 1e-8:
        norm = np.zeros_like(ch)
    else:
        norm = (ch - lo) / (hi - lo)
    mapper = cm.get_cmap(cmap_name)
    rgba = mapper(norm)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def render_grid_png(
    feat: np.ndarray,
    input_img: np.ndarray | None,
    stem: str,
    output: pathlib.Path,
) -> None:
    """Render a 5-row × 4-col grid PNG: input + 16 channels + L2 norm."""
    # Layout: row 0 = input+title, rows 1-4 = channels 0-15, last cell = L2 norm
    # Use a 4-col layout: input image spans top row; channels fill a 4×4 grid
    # Final row: channel 12, 13, 14, 15, then L2 norm in a 5th cell...
    # Actually: make a clean 5×4 grid (20 cells):
    #   cell 0 = input image (or blank), cells 1-16 = channels 0-15, cell 17 = L2 norm

    COLS = 4
    ROWS = 5  # row0=input+3blank, rows1-4=ch0-15 (4 per row)
    # Let's do: row0 = [input, ch0, ch1, ch2], rows 1-4 = ch3-15, last = L2 norm
    # Simpler: 4 cols × 5 rows = 20 panels
    # panels: 0=input, 1-16=ch0-15, 17=L2norm, 18-19=blank

    fig, axes = plt.subplots(
        ROWS, COLS, figsize=(COLS * 3.2, ROWS * 2.0), dpi=120, facecolor="#111"
    )
    fig.suptitle(f"Feature channels — {stem}", color="white", fontsize=11, y=0.995)

    panels = []
    # Panel 0: input grayscale
    if input_img is not None:
        panels.append(
            (
                "Input (grayscale)",
                np.stack([input_img, input_img, input_img], -1),
                "gray",
            )
        )
    else:
        panels.append(("Input (N/A)", np.zeros((H, W, 3), dtype=np.float32), "gray"))

    # Panels 1-16: individual channels
    for c in range(N_CH):
        ch = feat[:, :, c]
        lo, hi = ch.min(), ch.max()
        title = f"ch{c:02d}  [{lo:.2f}, {hi:.2f}]"
        rgb = channel_to_rgb(ch, CMAPS[c % len(CMAPS)])
        panels.append((title, rgb / 255.0, CMAPS[c % len(CMAPS)]))

    # Panel 17: L2 norm
    l2 = np.linalg.norm(feat, axis=-1)
    l2_rgb = channel_to_rgb(l2, "hot") / 255.0
    panels.append((f"L2 norm  [{l2.min():.2f}, {l2.max():.2f}]", l2_rgb, "hot"))

    # Panels 18-19: blank
    for _ in range(2):
        panels.append(("", np.zeros((H, W, 3), dtype=np.float32), None))

    for idx, (title, img, _cmap) in enumerate(panels):
        r, c = divmod(idx, COLS)
        ax = axes[r][c]
        ax.imshow(img, interpolation="nearest", aspect="auto")
        ax.set_title(title, color="white", fontsize=7, pad=2)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout(pad=0.3, rect=[0, 0, 1, 0.995])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="#111")
    plt.close(fig)
    print(f"  saved → {output}")


def collect_frames(f32_dir: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """Return sorted (stem, f32_path) pairs."""
    pat = re.compile(r"rtl_output_(frame\d+_yaw[^/]+)\.f32$")
    results = []
    for subdir in sorted(f32_dir.iterdir()):
        for f in subdir.glob("tb_data/rtl_output_*.f32"):
            m = pat.search(f.name)
            if m:
                results.append((m.group(1), f))
    results.sort(key=lambda x: x[0])
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--f32-dir",
        default="v2/hls4ml/parallel_sim",
        help="directory containing per-frame subdirs",
    )
    ap.add_argument(
        "--npz-dir",
        default="v2/data/perspective_stereo",
        help="directory with .npz input files",
    )
    ap.add_argument(
        "--frame",
        default=None,
        help="single frame stem, e.g. frame00000_yaw+000_pitch+00",
    )
    ap.add_argument(
        "--output",
        default="v2/hls4ml/channel_grid.png",
        help="output PNG (single frame) or MP4 (all frames)",
    )
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    f32_dir = pathlib.Path(args.f32_dir)
    npz_dir = pathlib.Path(args.npz_dir)
    output = pathlib.Path(args.output)

    if args.frame:
        # Single frame
        stem = args.frame
        # Search for the f32 file
        f32_path = f32_dir / stem / f"tb_data/rtl_output_{stem}.f32"
        if not f32_path.exists():
            # try old naming
            for subdir in f32_dir.iterdir():
                candidate = subdir / f"tb_data/rtl_output_{stem}.f32"
                if candidate.exists():
                    f32_path = candidate
                    break
        if not f32_path.exists():
            print(f"ERROR: could not find .f32 for {stem}", file=sys.stderr)
            sys.exit(1)
        feat = load_f32(f32_path)
        npz_path = npz_dir / f"{stem}.npz"
        inp = load_input(npz_path) if npz_path.exists() else None
        out = output if output.suffix.lower() == ".png" else output.with_suffix(".png")
        render_grid_png(feat, inp, stem, out)
        return

    # All frames → video
    frames = collect_frames(f32_dir)
    if not frames:
        print("ERROR: no .f32 files found", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(frames)} frames")

    if output.suffix.lower() == ".mp4":
        import cv2  # type: ignore

        tmp_dir = output.parent / "_grid_frames"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        png_paths = []
        for stem, f32_path in frames:
            feat = load_f32(f32_path)
            npz_path = npz_dir / f"{stem}.npz"
            inp = load_input(npz_path) if npz_path.exists() else None
            png_out = tmp_dir / f"{stem}.png"
            render_grid_png(feat, inp, stem, png_out)
            png_paths.append(png_out)

        # Build video
        ref = plt.imread(str(png_paths[0]))
        fh, fw = ref.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output), fourcc, args.fps, (fw, fh))
        for p in png_paths:
            frame_bgr = cv2.imread(str(p))
            writer.write(frame_bgr)
        writer.release()
        print(f"Video written → {output}  ({len(png_paths)} frames @ {args.fps} fps)")
    else:
        # Single output: render the first frame
        stem, f32_path = frames[0]
        feat = load_f32(f32_path)
        npz_path = npz_dir / f"{stem}.npz"
        inp = load_input(npz_path) if npz_path.exists() else None
        render_grid_png(feat, inp, stem, output)


if __name__ == "__main__":
    main()
