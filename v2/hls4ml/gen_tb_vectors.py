"""Generate tb_data .dat / .mem files for hls4ml co-simulation.

Two input modes:
  --parity-batch  PATH   Use parity_batch.npz (N frames, keys: images, labels)
  --npz-input     PATH   Use a single perspective_stereo .npz (keys: left_image)

Both modes require --hls-dir and --model.

Outputs written to <hls-dir>/tb_data/:
  tb_input_features.dat       space-separated float32, one inference per line
  tb_output_predictions.dat   space-separated float32, one inference per line
  tb_input_hex.mem            46080-line 4-digit hex (fixed<16,6>), for $readmemh

The .dat files are consumed by myproject_test.cpp (C-sim and RTL co-sim).
The .mem file is loaded by the SystemVerilog testbench via $readmemh.
"""

import argparse
import os
import pathlib

import numpy as np

# ---------------------------------------------------------------------------
# fixed<16,6> encoding
# ---------------------------------------------------------------------------


def float_to_fixed16_6(arr: np.ndarray) -> np.ndarray:
    """Convert float32 array to int16 fixed<16,6> representation.

    pixel_int16 = clip(round(pixel_float * 2^6), -32768, 32767)
    """
    scaled = np.round(arr.astype(np.float32) * 64.0)
    return np.clip(scaled, -32768, 32767).astype(np.int16)


def write_hex_mem(pixels_float: np.ndarray, out_path: pathlib.Path) -> None:
    """Write 46080 lines of 4-digit hex (uint16) from float pixel array."""
    flat = pixels_float.flatten()
    assert flat.shape[0] == 160 * 288, f"Expected 46080 pixels, got {flat.shape[0]}"
    int16_vals = float_to_fixed16_6(flat)
    # Reinterpret int16 as uint16 for hex output (two's complement)
    uint16_vals = int16_vals.view(np.uint16)
    with open(out_path, "w") as f:
        for v in uint16_vals:
            f.write(f"{v:04x}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate tb_data .dat and .mem files for hls4ml co-simulation."
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--parity-batch",
        metavar="PATH",
        help="Path to parity_batch.npz (keys: images (N,160,288,1), labels)",
    )
    group.add_argument(
        "--npz-input",
        metavar="PATH",
        help="Path to a single perspective_stereo .npz (key: left_image (160,288,1))",
    )
    ap.add_argument(
        "--hls-dir",
        required=True,
        metavar="PATH",
        help="Path to the fixed16_6/ hls4ml output directory",
    )
    ap.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="Path to feature_extractor.keras for reference output generation",
    )
    args = ap.parse_args()

    # --- Keras / model setup --------------------------------------------------
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras  # noqa: E402 — import after env var is set

    from v2.models.keras_student import (
        CorrelationCostVolume,
        _masked_smooth_l1,
        _mean_abs_error_valid,
    )

    model = keras.models.load_model(
        args.model,
        custom_objects={
            "CorrelationCostVolume": CorrelationCostVolume,
            "_masked_smooth_l1": _masked_smooth_l1,
            "_mean_abs_error_valid": _mean_abs_error_valid,
        },
    )

    # --- Load input images ----------------------------------------------------
    if args.parity_batch:
        data = np.load(args.parity_batch)
        images = data["images"]  # (N, 160, 288, 1) float32 in [0, 1]
    else:
        data = np.load(args.npz_input)
        # perspective_stereo .npz has key 'left_image' (160, 288, 1) float32
        img = data["left_image"]  # (160, 288, 1)
        images = img[np.newaxis]  # (1, 160, 288, 1)

    N = images.shape[0]
    print(f"Loaded {N} frame(s) — shape {images.shape}")

    # --- Run model to get reference outputs -----------------------------------
    # The feature extractor takes a single image and produces (H, W, 16) features.
    # Use the sub-model if available, otherwise use the full model.
    try:
        feat_model = model.get_layer("feature_extractor")
    except ValueError:
        # Fall back: model IS the feature extractor
        feat_model = model

    ref_outputs = feat_model.predict(images, batch_size=1)  # (N, 160, 288, 16)
    print(f"Reference output shape: {ref_outputs.shape}")

    # --- Write tb_data files --------------------------------------------------
    tb = pathlib.Path(args.hls_dir) / "tb_data"
    tb.mkdir(exist_ok=True)

    in_dat = tb / "tb_input_features.dat"
    out_dat = tb / "tb_output_predictions.dat"
    hex_mem = tb / "tb_input_hex.mem"

    with open(in_dat, "w") as f_in, open(out_dat, "w") as f_out:
        for i in range(N):
            inp = images[i].flatten()  # (46080,)
            out = ref_outputs[i].flatten()  # (737280,)
            f_in.write(" ".join(f"{v:.6f}" for v in inp) + "\n")
            f_out.write(" ".join(f"{v:.6f}" for v in out) + "\n")

    print(f"Wrote {N} inference pair(s) to {in_dat} and {out_dat}")

    # Write .mem for the first frame (SV testbench uses one frame at a time)
    write_hex_mem(images[0], hex_mem)
    print(f"Wrote hex .mem for frame 0 → {hex_mem}")


if __name__ == "__main__":
    main()
