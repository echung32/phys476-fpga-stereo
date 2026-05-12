"""Render feature map outputs from parallel xsim runs into a visualization video.

Loads all rtl_output_*.f32 files (produced by myproject_tb.sv), computes a
per-pixel summary (L2 norm across 16 channels), and writes an MP4 showing:
  - Left: original grayscale input image (from matching .npz)
  - Right: RTL feature map magnitude (normalized heatmap)

Usage:
    uv run python -m v2.hls4ml.render_feature_video \\
        --f32-dir v2/hls4ml/parallel_sim \\
        --npz-dir v2/data/perspective_stereo \\
        --output  v2/hls4ml/feature_video.mp4 \\
        --fps     2

Without --npz-dir, only the feature heatmap is shown.
"""

import argparse
import pathlib
import sys
import re

import numpy as np

H, W, N_CH = 160, 288, 16
N_PIX = H * W


def load_f32(path: pathlib.Path) -> np.ndarray:
    """Load a .f32 text file → float32 (H, W, N_CH)."""
    lines = path.read_text().splitlines()
    if len(lines) != N_PIX:
        raise ValueError(f"{path}: expected {N_PIX} lines, got {len(lines)}")
    data = np.array([list(map(float, l.split())) for l in lines], dtype=np.float32)
    return data.reshape(H, W, N_CH)


def feature_heatmap(feat: np.ndarray) -> np.ndarray:
    """Compute L2 norm across channels → (H, W) normalized to [0, 255] uint8."""
    mag = np.linalg.norm(feat, axis=-1)  # (H, W)
    lo, hi = mag.min(), mag.max()
    if hi - lo < 1e-6:
        return np.zeros((H, W), dtype=np.uint8)
    norm = ((mag - lo) / (hi - lo) * 255).astype(np.uint8)
    return norm


def colorize(gray: np.ndarray, cmap: str = "viridis") -> np.ndarray:
    """Apply a matplotlib colormap → (H, W, 3) uint8 RGB."""
    import matplotlib.cm as cm  # type: ignore

    mapper = cm.get_cmap(cmap)
    rgba = mapper(gray.astype(np.float32) / 255.0)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def load_input_image(npz_path: pathlib.Path) -> np.ndarray | None:
    """Load left_image from a perspective_stereo .npz → (H, W, 3) uint8 RGB."""
    try:
        d = np.load(npz_path)
        img = d["left_image"]  # (512, 512, 1) float32
        # Center-crop to 160×288 matching hls4ml input preprocessing
        # (same crop used in gen_tb_vectors.py / stereo_data.py)
        ih, iw = img.shape[:2]
        cy, cx = ih // 2, iw // 2
        half_h, half_w = H // 2, W // 2
        crop = img[cy - half_h : cy + half_h, cx - half_w : cx + half_w, 0]
        # Normalize to [0,255]
        lo, hi = crop.min(), crop.max()
        if hi - lo < 1e-6:
            u8 = np.zeros((H, W), dtype=np.uint8)
        else:
            u8 = ((crop - lo) / (hi - lo) * 255).astype(np.uint8)
        # Convert grayscale → RGB for side-by-side display
        return np.stack([u8, u8, u8], axis=-1)
    except Exception as e:
        print(f"  WARNING: could not load input image from {npz_path}: {e}")
        return None


def make_frame_image(
    feat: np.ndarray,
    input_rgb: np.ndarray | None,
    title: str = "",
) -> np.ndarray:
    """Build a side-by-side (input | heatmap) RGB image, or just heatmap."""
    hmap = colorize(feature_heatmap(feat))

    if input_rgb is not None:
        # Resize input to match heatmap if needed
        if input_rgb.shape[:2] != (H, W):
            from PIL import Image  # type: ignore

            pil = Image.fromarray(input_rgb).resize((W, H), Image.BILINEAR)
            input_rgb = np.array(pil)
        # Add 4px divider
        divider = np.full((H, 4, 3), 128, dtype=np.uint8)
        frame_img = np.concatenate([input_rgb, divider, hmap], axis=1)
    else:
        frame_img = hmap

    return frame_img


def collect_f32_files(f32_dir: pathlib.Path) -> list[pathlib.Path]:
    """Recursively find all rtl_output_*.f32 files, sorted by frame stem."""
    files = sorted(f32_dir.rglob("rtl_output_*.f32"))
    return files


def stem_to_frame_num(stem: str) -> int:
    """Extract frame number from stem like 'rtl_output_frame00015_yaw+000_pitch+00'."""
    m = re.search(r"frame(\d+)", stem)
    return int(m.group(1)) if m else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Render RTL feature maps to video")
    ap.add_argument(
        "--f32-dir",
        required=True,
        metavar="DIR",
        help="Directory tree containing rtl_output_*.f32 files",
    )
    ap.add_argument(
        "--npz-dir",
        metavar="DIR",
        default=None,
        help="Directory containing matching .npz input files",
    )
    ap.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output video path (e.g. feature_video.mp4)",
    )
    ap.add_argument(
        "--fps", type=float, default=2.0, help="Output video FPS (default: 2)"
    )
    ap.add_argument(
        "--channel",
        type=int,
        default=None,
        help="If set, visualize a specific channel instead of L2 norm",
    )
    ap.add_argument("--stats", action="store_true", help="Print per-frame statistics")
    args = ap.parse_args()

    f32_files = collect_f32_files(pathlib.Path(args.f32_dir))
    if not f32_files:
        print(
            f"ERROR: no rtl_output_*.f32 files found under {args.f32_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Sort by frame number
    f32_files.sort(key=lambda p: stem_to_frame_num(p.stem))
    print(f"Found {len(f32_files)} output file(s):")
    for f in f32_files:
        print(f"  {f}")

    try:
        import cv2  # type: ignore

        has_cv2 = True
    except ImportError:
        has_cv2 = False
        print("WARNING: opencv-python not available — will save PNGs instead of video")

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames_list = []
    for f32_path in f32_files:
        stem = f32_path.stem  # e.g. rtl_output_frame00000_yaw+000_pitch+00
        print(f"\nProcessing: {stem}")

        feat = load_f32(f32_path)

        if args.stats:
            mag = np.linalg.norm(feat, axis=-1)
            print(
                f"  feat shape={feat.shape}, min={feat.min():.4f}, "
                f"max={feat.max():.4f}, L2 norm: min={mag.min():.4f} max={mag.max():.4f}"
            )
            ch_means = feat.reshape(-1, N_CH).mean(axis=0)
            print(f"  ch means: {' '.join(f'{v:.3f}' for v in ch_means)}")

        # Find matching input npz
        input_rgb = None
        if args.npz_dir:
            # stem looks like rtl_output_frame00015_yaw+000_pitch+00
            # npz is frame00015_yaw+000_pitch+00.npz
            frame_part = re.sub(r"^rtl_output_", "", stem)
            npz_path = pathlib.Path(args.npz_dir) / f"{frame_part}.npz"
            if npz_path.exists():
                input_rgb = load_input_image(npz_path)
            else:
                print(f"  No matching npz: {npz_path}")

        if args.channel is not None:
            ch_img = feat[:, :, args.channel]
            lo, hi = ch_img.min(), ch_img.max()
            if hi - lo > 1e-6:
                ch_u8 = ((ch_img - lo) / (hi - lo) * 255).astype(np.uint8)
            else:
                ch_u8 = np.zeros((H, W), dtype=np.uint8)
            hmap_rgb = colorize(ch_u8)
        else:
            hmap_rgb = colorize(feature_heatmap(feat))

        frame_img = make_frame_image(feat if args.channel is None else feat, input_rgb)
        frames_list.append(frame_img)

        # Save individual PNG
        png_path = out_path.parent / f"{stem}.png"
        if has_cv2:
            cv2.imwrite(str(png_path), cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))
        else:
            from PIL import Image  # type: ignore

            Image.fromarray(frame_img).save(str(png_path))
        print(f"  Saved PNG: {png_path}")

    # Write video
    if len(frames_list) >= 1 and has_cv2:
        fh, fw = frames_list[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_path), fourcc, args.fps, (fw, fh))
        for fr in frames_list:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release()
        print(
            f"\nVideo saved: {out_path}  ({len(frames_list)} frames @ {args.fps} fps)"
        )
    elif len(frames_list) >= 1:
        print(f"\nSaved {len(frames_list)} PNG frames to {out_path.parent}/")
        print("Install opencv-python (uv add opencv-python) for .mp4 output.")

    print("Done.")


if __name__ == "__main__":
    main()
