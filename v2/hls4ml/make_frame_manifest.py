#!/usr/bin/env python3
"""Generate a frame_manifest.txt for multi-frame Vivado xsim runs.

Copies selected .mem files into tb_data/ and writes a manifest listing them
so the testbench can iterate over multiple frames in one simulation run.

Usage (from workspace root):
    uv run python -m v2.hls4ml.make_frame_manifest \\
        --mem-dir v2/data/perspective_stereo \\
        --yaw +000 \\
        --n-frames 5 \\
        --tb-data v2/hls4ml/fixed16_6/tb_data

The manifest is written to <tb-data>/frame_manifest.txt.
Each .mem file is copied into <tb-data>/frames/ so the xsim staged copy works.
"""

import argparse
import pathlib
import shutil
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Build xsim frame manifest")
    ap.add_argument(
        "--mem-dir",
        required=True,
        metavar="DIR",
        help="Directory containing .mem files (e.g. v2/data/perspective_stereo)",
    )
    ap.add_argument(
        "--yaw",
        default="+000",
        metavar="YAW",
        help="Yaw angle suffix to select (default: +000, i.e. straight-ahead)",
    )
    ap.add_argument(
        "--pitch",
        default="+00",
        metavar="PITCH",
        help="Pitch angle suffix to select (default: +00)",
    )
    ap.add_argument(
        "--n-frames",
        type=int,
        default=5,
        metavar="N",
        help="Max number of frames to include (default: 5)",
    )
    ap.add_argument(
        "--tb-data",
        required=True,
        metavar="DIR",
        help="tb_data/ directory where manifest and copied .mem files land",
    )
    ap.add_argument(
        "--all-yaw",
        action="store_true",
        help="Include all yaw angles (ignores --yaw)",
    )
    ap.add_argument(
        "--frame-step",
        type=int,
        default=1,
        metavar="N",
        help="Take every Nth matching frame (default: 1). "
        "Use 5 to subsample 10fps → 2fps.",
    )
    args = ap.parse_args()

    mem_dir = pathlib.Path(args.mem_dir)
    tb_data = pathlib.Path(args.tb_data)
    frames_dir = tb_data / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Collect matching .mem files
    pattern = f"*yaw{args.yaw}_pitch{args.pitch}.mem" if not args.all_yaw else "*.mem"
    candidates = sorted(mem_dir.glob(pattern))
    if not candidates:
        print(f"ERROR: no .mem files matched {mem_dir}/{pattern}", file=sys.stderr)
        sys.exit(1)

    # Subsample at frame-step, then limit to n-frames
    subsampled = candidates[:: args.frame_step]
    selected = subsampled[: args.n_frames]

    manifest_lines = []
    for src in selected:
        dst = frames_dir / src.name
        shutil.copy2(src, dst)
        # Path relative to xsim run dir — tb_data/ is staged there
        manifest_lines.append(f"tb_data/frames/{src.name}")

    manifest_path = tb_data / "frame_manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")

    print(f"Manifest written: {manifest_path}")
    print(f"  {len(selected)} frame(s):")
    for line in manifest_lines:
        print(f"    {line}")


if __name__ == "__main__":
    main()
