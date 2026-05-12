"""Decode and validate Vivado xsim output hex files.

The testbench writes one 64-hex-char (256-bit) line per pixel position.
Each line packs 16 × fixed<16,6> channels: bits [15:0]=ch0, [31:16]=ch1, …

Usage examples
--------------
# Single frame against Python reference:
  KERAS_BACKEND=torch uv run python -m v2.hls4ml.decode_vivado_output \\
      --hex-file v2/hls4ml/fixed16_6/tb_data/rtl_output_frame0.hex \\
      --npz      logs/monkaa_crop16x9_sf25_fc32_h160_w288_bs128/parity_batch.npz \\
      --model    v2/hls4ml/fixed16_6/feature_extractor.keras \\
      --frame-idx 0

# All .hex files in a directory (no reference comparison):
  KERAS_BACKEND=torch uv run python -m v2.hls4ml.decode_vivado_output \\
      --hex-dir v2/hls4ml/fixed16_6/tb_data/
"""

import argparse
import pathlib
import sys
import numpy as np


H, W, N_CH = 160, 288, 16
N_PIX = H * W  # 46 080


def decode_hex_file(hex_path: pathlib.Path) -> np.ndarray:
    """Return float32 array of shape (H, W, N_CH) decoded from xsim .hex output."""
    lines = hex_path.read_text().splitlines()
    if len(lines) != N_PIX:
        raise ValueError(f"{hex_path}: expected {N_PIX} lines, got {len(lines)}")
    # Parse 256-bit hex into 16 × int16 values
    data = np.zeros((N_PIX, N_CH), dtype=np.int16)
    for i, line in enumerate(lines):
        val = int(line.strip(), 16)
        for ch in range(N_CH):
            # Extract 16 bits for channel ch (little-endian packing)
            raw = (val >> (ch * 16)) & 0xFFFF
            # Reinterpret as signed int16
            data[i, ch] = np.int16(raw)
    # fixed<16,6> → float: divide by 2^6 = 64
    out = data.astype(np.float32) / 64.0
    return out.reshape(H, W, N_CH)


def compare_to_reference(
    rtl: np.ndarray,
    ref: np.ndarray,
    label: str = "",
) -> None:
    """Print numeric comparison statistics."""
    ae = np.abs(rtl - ref)
    print(f"\n{'=' * 60}")
    if label:
        print(f"Frame: {label}")
    print(f"  RTL output shape : {rtl.shape}")
    print(f"  Ref output shape : {ref.shape}")
    print(f"  MAE              : {ae.mean():.6f}")
    print(f"  Max abs error    : {ae.max():.6f}")
    print(f"  RMSE             : {np.sqrt((ae**2).mean()):.6f}")
    # Channel-wise mean absolute error
    cw = ae.reshape(-1, N_CH).mean(axis=0)
    print(f"  Per-channel MAE  : {' '.join(f'{v:.4f}' for v in cw)}")
    # What fraction of pixels have any channel mismatch > 1 quant step (1/64)?
    quant = 1.0 / 64.0
    mismatch = (ae > quant * 1.5).any(axis=-1)
    print(
        f"  Pixel mismatches (>1.5 quant): {mismatch.sum()} / {mismatch.size} "
        f"({100 * mismatch.mean():.2f}%)"
    )


def get_python_reference(
    npz_path: pathlib.Path,
    model_path: pathlib.Path,
    frame_idx: int,
) -> np.ndarray:
    """Run Python feature extractor on a frame from a parity .npz file."""
    import os

    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras  # noqa: F401 — ensure backend is initialized

    data = np.load(npz_path)
    # parity_batch.npz has 'left_images' key, shape (N, H, W) or (N, H, W, 1)
    imgs = data["left_images"]
    if imgs.ndim == 3:
        imgs = imgs[..., np.newaxis]
    img = imgs[frame_idx : frame_idx + 1]  # (1, H, W, 1)

    import keras as k

    full_model = k.models.load_model(model_path, compile=False)
    # Try to extract feature_extractor sub-model
    try:
        fe = full_model.get_layer("feature_extractor")
    except ValueError:
        fe = full_model  # model is already the extractor
    pred = fe.predict(img, verbose=0)  # (1, H, W, N_CH) or (1, N_CH, H, W)
    if pred.shape[1] == N_CH:
        pred = np.transpose(pred, (0, 2, 3, 1))  # NCHW → NHWC
    return pred[0].astype(np.float32)  # (H, W, N_CH)


def print_sample_values(feat: np.ndarray, n: int = 8) -> None:
    """Print first n pixel positions for all 16 channels."""
    print(f"\n  First {n} pixel positions (ch0–ch15):")
    flat = feat.reshape(-1, N_CH)
    for i in range(min(n, flat.shape[0])):
        vals = " ".join(f"{v:+.4f}" for v in flat[i])
        print(f"    [{i:5d}]  {vals}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Decode Vivado xsim output .hex files")
    ap.add_argument("--hex-file", metavar="PATH", help="Single .hex file to decode")
    ap.add_argument(
        "--hex-dir", metavar="DIR", help="Directory — decode all rtl_output_*.hex files"
    )
    ap.add_argument(
        "--npz", metavar="PATH", help="parity_batch.npz for Python reference comparison"
    )
    ap.add_argument(
        "--npz-input",
        metavar="PATH",
        help="Single-frame .npz (from perspective_stereo/) for reference",
    )
    ap.add_argument(
        "--model",
        metavar="PATH",
        help="feature_extractor.keras for Python reference inference",
    )
    ap.add_argument(
        "--frame-idx",
        type=int,
        default=0,
        metavar="N",
        help="Frame index into parity_batch.npz (default: 0)",
    )
    ap.add_argument(
        "--print-samples",
        action="store_true",
        help="Print first 8 pixel values for each frame",
    )
    args = ap.parse_args()

    hex_files: list[pathlib.Path] = []
    if args.hex_file:
        hex_files.append(pathlib.Path(args.hex_file))
    if args.hex_dir:
        d = pathlib.Path(args.hex_dir)
        hex_files.extend(sorted(d.glob("rtl_output_*.hex")))

    if not hex_files:
        print("ERROR: provide --hex-file or --hex-dir", file=sys.stderr)
        ap.print_help()
        sys.exit(1)

    # Optional reference model
    do_compare = bool(args.model and (args.npz or args.npz_input))

    for hf in hex_files:
        print(f"\nDecoding: {hf}")
        try:
            rtl = decode_hex_file(hf)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(
            f"  Shape: {rtl.shape}, min={rtl.min():.4f}, max={rtl.max():.4f}, "
            f"mean={rtl.mean():.4f}"
        )

        if args.print_samples:
            print_sample_values(rtl)

        if do_compare:
            try:
                model_path = pathlib.Path(args.model)
                if args.npz:
                    ref = get_python_reference(
                        pathlib.Path(args.npz), model_path, args.frame_idx
                    )
                else:
                    # Single .npz input — load left image directly
                    d = np.load(args.npz_input)
                    img = d["left_image"] if "left_image" in d else d[d.files[0]]
                    if img.ndim == 2:
                        img = img[np.newaxis, ..., np.newaxis]
                    import os

                    os.environ.setdefault("KERAS_BACKEND", "torch")
                    import keras as k

                    full_model = k.models.load_model(model_path, compile=False)
                    try:
                        fe = full_model.get_layer("feature_extractor")
                    except ValueError:
                        fe = full_model
                    ref = fe.predict(img, verbose=0)[0]
                    if ref.shape[0] == N_CH:
                        ref = np.transpose(ref, (1, 2, 0))
                compare_to_reference(rtl, ref, label=hf.stem)
            except Exception as e:
                print(f"  Reference comparison failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
