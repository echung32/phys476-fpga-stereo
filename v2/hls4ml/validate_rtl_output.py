"""Numeric validation: compare RTL co-simulation output against Python reference.

Usage (from workspace root):
    KERAS_BACKEND=torch uv run python -m v2.hls4ml.validate_rtl_output \
        --hls-dir v2/hls4ml/fixed16_6

Expected result: MAE ~0.02, max error ~0.43  (matches parity_metrics.json).
Any larger discrepancy indicates a simulation or testbench bug.
"""

import argparse
import pathlib
import sys

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare RTL co-sim output to Python reference"
    )
    ap.add_argument(
        "--hls-dir",
        required=True,
        metavar="PATH",
        help="Path to the fixed16_6/ hls4ml output directory",
    )
    ap.add_argument(
        "--rtl-log",
        default=None,
        metavar="PATH",
        help="Path to RTL cosim results log (default: <hls-dir>/tb_data/rtl_cosim_results.log)",
    )
    ap.add_argument(
        "--ref-dat",
        default=None,
        metavar="PATH",
        help="Path to reference predictions .dat (default: <hls-dir>/tb_data/tb_output_predictions.dat)",
    )
    args = ap.parse_args()

    hls_dir = pathlib.Path(args.hls_dir)
    rtl_log = (
        pathlib.Path(args.rtl_log)
        if args.rtl_log
        else hls_dir / "tb_data" / "rtl_cosim_results.log"
    )
    ref_dat = (
        pathlib.Path(args.ref_dat)
        if args.ref_dat
        else hls_dir / "tb_data" / "tb_output_predictions.dat"
    )

    for path in (rtl_log, ref_dat):
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Load RTL cosim results (space-separated float lines, one inference per line)
    with open(rtl_log) as f:
        rtl_vals = np.array(
            [[float(x) for x in line.split()] for line in f if line.strip()],
            dtype=np.float32,
        )

    # Load Python reference (same format)
    with open(ref_dat) as f:
        ref_vals = np.array(
            [[float(x) for x in line.split()] for line in f if line.strip()],
            dtype=np.float32,
        )

    if rtl_vals.shape != ref_vals.shape:
        print(
            f"WARNING: shape mismatch — RTL {rtl_vals.shape} vs ref {ref_vals.shape}. "
            "Truncating to common length."
        )
        n = min(len(rtl_vals), len(ref_vals))
        rtl_vals = rtl_vals[:n]
        ref_vals = ref_vals[:n]

    diff = np.abs(rtl_vals - ref_vals)
    mae = float(np.mean(diff))
    maxe = float(np.max(diff))
    rmse = float(np.sqrt(np.mean(diff**2)))

    print(f"Frames compared : {len(rtl_vals)}")
    print(f"Values per frame: {rtl_vals.shape[1] if rtl_vals.ndim > 1 else 1}")
    print(f"RTL vs Python MAE  : {mae:.5f}")
    print(f"RTL vs Python RMSE : {rmse:.5f}")
    print(f"RTL vs Python max  : {maxe:.5f}")
    print()

    # Thresholds from parity_metrics.json: MAE ~0.021, max ~0.43
    if mae < 0.05 and maxe < 1.0:
        print(
            "RESULT: PASS — RTL output matches Python reference within expected tolerance."
        )
    else:
        print("RESULT: FAIL — errors exceed expected thresholds (MAE<0.05, max<1.0).")
        sys.exit(1)


if __name__ == "__main__":
    main()
