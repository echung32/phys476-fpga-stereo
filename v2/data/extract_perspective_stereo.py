"""Extract perspective-projected stereo frame pairs from an equirectangular SBS stereo video.

Input format
------------
Side-by-side (SBS) equirectangular: total frame is W×H pixels where each half covers
a full hemisphere (VR180 format).

    [ LEFT eye (W/2 × H)  |  RIGHT eye (W/2 × H) ]

The pair is stereo-rectified: corresponding points lie on the same horizontal latitude
line.  This is the standard output of VR180 stereo rigs.

Why not use equirectangular directly
-------------------------------------
The correlation student model assumes rectilinear (perspective) geometry:
  - Horizontal epipolar lines (disparity is purely in X)
  - Uniform pixel density (no polar compression/stretching)
Equirectangular only satisfies these at elevation=0.  The model was trained on
SceneFlow/KITTI which are all rectilinear, so the representation must match.

Ground-truth disparity from 3D tracks (preferred)
--------------------------------------------------
When --metadata is supplied (a stereo-video.npz file in the format described below),
GT disparity is computed by projecting sparse 3D tracks into the perspective crop:

  disparity = f_x * B / Z_persp

where:
  - f_x  = perspective focal length derived from --fov and --out-size
  - B    = stereo baseline (default 63.5 mm for VR180 rigs; override with --baseline)
  - Z_persp = depth of the 3D point along the perspective camera's Z axis

The npz metadata file must contain:
  camera2world   : (N_frames, 3, 4)  LEFT camera pose [R|t] in world coords
  rectified2rig  : (2, 3, 3)         rotation from each rectified eye to rig frame
  track_lengths  : (N_tracks,)       observations per track
  track_indices  : (total_obs,)      frame index for each observation
  track_coordinates : (total_obs, 3) world-space 3D position per observation
  fov_bounds     : (4,)              [lon_min, lon_max, lat_min, lat_max] degrees

SGBM fallback
-------------
If --metadata is not supplied, or to supplement tracks in low-coverage regions,
--compute-disparity runs OpenCV SGBM.  Useful for textureless regions where SfM
tracks are absent.  SGBM results are masked by left-right consistency when --lr-check
is set.

Vivado simulation output
------------------------
Pass --export-hex to write a 16-bit fixed<16,6> .mem file alongside each .npz.
These can be loaded directly by a SystemVerilog testbench via $readmemh.
The fixed-point scale factor is 2^6 = 64 (6 fractional bits, matching the hls4ml
precision used in synthesis).

Usage
-----
# GT from 3D tracks (recommended when metadata is available)
uv run python -m v2.data.extract_perspective_stereo \\
    --video   v2/data/stereo-video.mp4 \\
    --metadata v2/data/stereo-video.npz \\
    --output-dir v2/data/perspective_stereo \\
    --fov 60 --out-size 512 \\
    --yaw-angles 0 45 90 135 180 -135 -90 -45 \\
    --frame-stride 3 \\
    --export-hex

# SGBM fallback (no metadata)
uv run python -m v2.data.extract_perspective_stereo \\
    --video v2/data/stereo-video.mp4 \\
    --output-dir v2/data/perspective_stereo \\
    --fov 60 --out-size 512 \\
    --compute-disparity --lr-check

Notes
-----
- Disparity units: pixels at the OUTPUT resolution (e.g., 512×512).
- valid_mask=0 pixels are ignored by the model's masked_smooth_l1 loss.
- .npz files contain: left_image, right_image, disparity, valid_mask, metadata.
- .mem files contain one 4-hex-digit uint16 per line (fixed<16,6> pixel value,
  left image only, row-major order) — ready for $readmemh in Vivado sim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Equirectangular → perspective remap
# ---------------------------------------------------------------------------


def build_equirect_to_persp_map(
    eq_h: int,
    eq_w: int,
    out_h: int,
    out_w: int,
    fov_h_deg: float,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (map_x, map_y) arrays for cv2.remap converting equirectangular → perspective.

    Parameters
    ----------
    eq_h, eq_w   : source equirectangular dimensions (height × width per eye)
    out_h, out_w : output perspective image dimensions
    fov_h_deg    : horizontal field-of-view of the output image, degrees
    yaw_deg      : azimuth of the view direction (0 = forward / +Z, positive = right)
    pitch_deg    : elevation of the view direction (0 = level, positive = up)
    """
    fov_h = np.radians(fov_h_deg)
    f_x = (out_w / 2.0) / np.tan(fov_h / 2.0)
    # Square pixels → same focal length vertically
    fov_v = 2.0 * np.arctan((out_h / out_w) * np.tan(fov_h / 2.0))
    f_y = (out_h / 2.0) / np.tan(fov_v / 2.0)
    cx, cy = out_w / 2.0, out_h / 2.0

    # Grid of output pixel coordinates
    u = np.arange(out_w, dtype=np.float64)
    v = np.arange(out_h, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)  # (out_h, out_w)

    # Ray direction in camera space (right-handed: X right, Y up, Z forward)
    dx = (uu - cx) / f_x
    dy = -(vv - cy) / f_y  # flip: image-y down → world-y up
    dz = np.ones_like(dx)

    # Normalise
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    dx, dy, dz = dx / norm, dy / norm, dz / norm

    # Apply pitch rotation (around world X axis, tilts up/down)
    p = np.radians(pitch_deg)
    dx1 = dx
    dy1 = np.cos(p) * dy - np.sin(p) * dz
    dz1 = np.sin(p) * dy + np.cos(p) * dz

    # Apply yaw rotation (around world Y axis, pans left/right)
    y = np.radians(yaw_deg)
    dx2 = np.cos(y) * dx1 + np.sin(y) * dz1
    dy2 = dy1
    dz2 = -np.sin(y) * dx1 + np.cos(y) * dz1

    # Spherical coordinates
    lon = np.arctan2(dx2, dz2)  # [-π, π]
    lat = np.arcsin(np.clip(dy2, -1.0, 1.0))  # [-π/2, π/2]

    # Equirectangular pixel lookup  — VR180 format (180°×180° per eye)
    # lon range [-π/2, π/2] → src_x in [0, eq_w]
    # lat range [-π/2, π/2] → src_y in [eq_h, 0]   (top = +lat)
    src_x = (lon / np.pi + 0.5) * eq_w  # VR180: 180° total horizontal
    src_y = (0.5 - lat / np.pi) * eq_h  # standard: 180° vertical

    return src_x.astype(np.float32), src_y.astype(np.float32)


def remap_equirect(
    img: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """Apply pre-computed VR180 equirect remap with bilinear interpolation.

    VR180 images cover only a hemisphere (no horizontal wrap-around).  Pixels
    that map outside the hemisphere boundary are filled with black.
    """
    out = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return out


# ---------------------------------------------------------------------------
# Pseudo-GT disparity via SGBM
# ---------------------------------------------------------------------------


def compute_sgbm_disparity(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    num_disparities: int = 64,
    block_size: int = 5,
    min_conf_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pseudo-GT disparity and valid mask using Semi-Global Block Matching.

    Parameters
    ----------
    left_gray, right_gray : uint8 grayscale images, same size
    num_disparities       : must be divisible by 16; search range [0, num_disparities)
    block_size            : must be odd, 3–11 is typical
    min_conf_ratio        : if > 0, computes WLS-filtered confidence map and masks low pixels

    Returns
    -------
    disparity   : float32 array, pixels; invalid pixels are 0.0
    valid_mask  : bool array, True where disparity is reliable
    """
    p1 = 8 * 3 * block_size**2
    p2 = 32 * 3 * block_size**2
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    raw = sgbm.compute(left_gray, right_gray)  # int16, ×16 fixed-point
    disp = raw.astype(np.float32) / 16.0

    valid = disp > 0.0

    if min_conf_ratio > 0.0:
        # Optional: Left-right consistency check for extra confidence
        sgbm_r = cv2.StereoSGBM_create(
            minDisparity=-num_disparities,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        raw_r = sgbm_r.compute(right_gray, left_gray)
        disp_r = raw_r.astype(np.float32) / 16.0

        h, w = disp.shape
        xs = np.arange(w)
        ys = np.arange(h)
        xx, yy = np.meshgrid(xs, ys)
        x_right = np.clip(xx - disp.astype(int), 0, w - 1)
        lr_diff = np.abs(disp + disp_r[yy, x_right])
        valid &= (lr_diff < 1.0) & (disp_r[yy, x_right] < 0)

    disp[~valid] = 0.0
    return disp, valid


# ---------------------------------------------------------------------------
# GT disparity from stereo-video.npz 3D tracks
# ---------------------------------------------------------------------------


def load_track_metadata(metadata_npz: str | Path) -> dict:
    """Load and preprocess track data from a stereo-video.npz metadata file.

    Returns a dict with:
        camera2world   : (N_frames, 3, 4) float64 — LEFT camera pose [R|t]
        track_coords   : (total_obs, 3) float32   — world-space 3D positions
        track_indices  : (total_obs,) int32        — frame index per observation
        frame_starts   : (N_frames+1,) int64       — CSR row pointers indexed by frame
        n_frames       : int
    """
    d = np.load(str(metadata_npz), allow_pickle=True)

    camera2world = d["camera2world"].astype(np.float64)  # (N, 3, 4)
    track_lengths = d["track_lengths"].astype(np.int64)  # (N_tracks,)
    track_indices = d["track_indices"].astype(np.int32)  # (total_obs,)
    track_coords = d["track_coordinates"].astype(np.float32)  # (total_obs, 3)

    n_frames = camera2world.shape[0]

    # Build a CSR-style lookup: for each frame, which observation indices belong to it?
    # Sort observations by frame so we can binary-search or use an index array.
    sort_order = np.argsort(track_indices, kind="stable")
    sorted_idx = track_indices[sort_order]  # frame ids, ascending
    sorted_coords = track_coords[sort_order]  # 3D coords in same order

    # For each frame f, observation slice = [frame_starts[f], frame_starts[f+1])
    frame_starts = np.searchsorted(sorted_idx, np.arange(n_frames + 1), side="left")
    frame_starts = frame_starts.astype(np.int64)

    return {
        "camera2world": camera2world,
        "track_coords": sorted_coords,
        "frame_starts": frame_starts,
        "n_frames": n_frames,
    }


def project_tracks_to_disparity(
    track_coords_cam: np.ndarray,
    out_h: int,
    out_w: int,
    f_x: float,
    f_y: float,
    yaw_deg: float,
    pitch_deg: float,
    baseline_m: float = 0.0635,
    min_depth: float = 0.1,
    max_depth: float = 500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points (in LEFT camera frame) to a perspective-crop disparity map.

    The perspective crop is defined by the same yaw/pitch/FoV as the image remap.
    The forward model mirrors `build_equirect_to_persp_map`:
      - Remap (backward): P_eq = Ry(yaw) @ Rx(pitch) @ P_persp_ray
      - Forward (this fn): P_persp = Rx(pitch)^T @ Ry(yaw)^T @ P_cam

    Parameters
    ----------
    track_coords_cam : (N, 3) float32 — 3D points in LEFT camera frame (metres)
    out_h, out_w     : output perspective image size (pixels)
    f_x, f_y         : perspective focal lengths (pixels)
    yaw_deg          : azimuth of perspective crop centre (degrees)
    pitch_deg        : elevation of perspective crop centre (degrees)
    baseline_m       : stereo baseline (metres; default 63.5 mm for VR180)
    min_depth        : minimum valid Z depth (metres) — reject points behind camera
    max_depth        : maximum valid Z depth (metres) — reject far clutter

    Returns
    -------
    disparity_map : (out_h, out_w) float32, pixels; 0.0 where no GT
    valid_mask    : (out_h, out_w) bool
    """
    if len(track_coords_cam) == 0:
        return (
            np.zeros((out_h, out_w), dtype=np.float32),
            np.zeros((out_h, out_w), dtype=bool),
        )

    cx = out_w / 2.0
    cy = out_h / 2.0

    # Build the rotation from left-camera-equirect space to perspective-crop space
    # (inverse of the remap rotation: Ry(yaw) @ Rx(pitch))
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)

    # Rx(pitch):  rotates Y-up camera frame by pitch around X
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]],
        dtype=np.float64,
    )
    # Ry(yaw):  rotates by yaw around Y
    Ry = np.array(
        [[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]],
        dtype=np.float64,
    )

    # R_persp transforms from camera-equirect to perspective-crop frame
    # P_persp = Rx.T @ Ry.T @ P_cam
    R_persp = Rx.T @ Ry.T  # (3, 3)

    # Rotate all points — vectorised
    P = track_coords_cam.astype(np.float64)  # (N, 3)
    P_persp = P @ R_persp.T  # (N, 3)

    X = P_persp[:, 0]
    Y = P_persp[:, 1]
    Z = P_persp[:, 2]

    # Keep only points in front of camera and within depth range
    valid_depth = (Z > min_depth) & (Z < max_depth)

    # Perspective projection: image-Y is flipped (down +)
    u = cx + f_x * X / np.where(valid_depth, Z, 1.0)
    v = cy - f_y * Y / np.where(valid_depth, Z, 1.0)

    # Integer pixel coordinates
    u_int = np.round(u).astype(np.int32)
    v_int = np.round(v).astype(np.int32)

    in_bounds = (u_int >= 0) & (u_int < out_w) & (v_int >= 0) & (v_int < out_h)
    valid = valid_depth & in_bounds

    # Disparity = f_x * B / Z
    disp = np.where(valid, f_x * baseline_m / Z, 0.0)

    # Scatter to image: for collisions keep the closest point (smallest Z overwrites)
    disparity_map = np.zeros((out_h, out_w), dtype=np.float32)
    valid_mask = np.zeros((out_h, out_w), dtype=bool)

    idx_valid = np.where(valid)[0]
    if len(idx_valid) > 0:
        # Sort by descending Z so closer points write last (overwrite farther ones)
        order = np.argsort(-Z[idx_valid])
        for i in order:
            ti = idx_valid[i]
            pu, pv = u_int[ti], v_int[ti]
            disparity_map[pv, pu] = float(disp[ti])
            valid_mask[pv, pu] = True

    return disparity_map, valid_mask


# ---------------------------------------------------------------------------
# Vivado simulation hex export
# ---------------------------------------------------------------------------


def export_fixed16_hex(
    image_float: np.ndarray,
    out_path: str | Path,
    frac_bits: int = 6,
) -> None:
    """Write a 16-bit fixed-point .mem file for Vivado $readmemh.

    The image is converted to fixed<16, frac_bits> format:
      pixel_fixed = clip(round(pixel_float * 2^frac_bits), -32768, 32767)

    Output format: one 4-digit lowercase hex value per line, row-major order.
    This matches the AXI-Stream input format of the hls4ml feature extractor.

    Parameters
    ----------
    image_float : (H, W) or (H, W, 1) float32 in [0, 1]
    out_path    : output .mem file path
    frac_bits   : fractional bits of the fixed-point format (default 6 for fixed<16,6>)
    """
    img = image_float.squeeze()  # → (H, W)
    scale = float(2**frac_bits)
    raw = np.round(img.astype(np.float64) * scale)
    raw = np.clip(raw, -32768, 32767).astype(np.int16)
    # Reinterpret as uint16 for hex formatting
    raw_u = raw.view(np.uint16).ravel()

    with open(out_path, "w") as f:
        for val in raw_u:
            f.write(f"{val:04x}\n")


# ---------------------------------------------------------------------------
# Pixel density diagnostics
# ---------------------------------------------------------------------------


def _effective_resolution(eq_w: int, fov_deg: float, out_w: int) -> float:
    """Source pixels per output pixel along the equatorial horizontal axis."""
    eq_pixels_in_fov = eq_w * fov_deg / 360.0
    return eq_pixels_in_fov / out_w


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def extract(
    video_path: str | Path,
    output_dir: str | Path,
    fov_h_deg: float = 60.0,
    out_height: int = 160,
    out_width: int = 288,
    yaw_angles: list[float] | None = None,
    pitch_angles: list[float] | None = None,
    frame_stride: int = 1,
    metadata_npz: str | Path | None = None,
    baseline_m: float = 0.0635,
    compute_disparity: bool = False,
    sgbm_num_disparities: int = 64,
    sgbm_block_size: int = 5,
    lr_check: bool = False,
    export_hex: bool = False,
) -> list[Path]:
    """Run the full extraction pipeline.

    Returns list of .npz paths written.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if yaw_angles is None:
        yaw_angles = [0.0]
    if pitch_angles is None:
        pitch_angles = [0.0]

    # --- Load track metadata (optional) ---
    meta = None
    if metadata_npz is not None:
        print(f"Loading track metadata from: {metadata_npz}")
        meta = load_track_metadata(metadata_npz)
        print(
            f"  {meta['n_frames']} camera poses, "
            f"{meta['frame_starts'][-1]} total track observations"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {frame_w}×{frame_h}, {total_frames} frames @ {fps} fps")

    eq_w = frame_w // 2  # per-eye equirectangular width
    eq_h = frame_h  # per-eye equirectangular height
    print(f"Per-eye equirectangular: {eq_w}×{eq_h} (VR180, 180°×180°)")

    out_h = out_height
    out_w = out_width

    # Perspective focal lengths (same for all crops at this FoV / size)
    fov_h = np.radians(fov_h_deg)
    f_x = float((out_w / 2.0) / np.tan(fov_h / 2.0))
    fov_v = 2.0 * np.arctan((out_h / out_w) * np.tan(fov_h / 2.0))
    f_y = float((out_h / 2.0) / np.tan(fov_v / 2.0))

    eff_res = _effective_resolution(eq_w, fov_h_deg, out_w)
    if eff_res < 0.8:
        print(
            f"WARNING: upsampling factor {1 / eff_res:.2f}× "
            f"(source has {eff_res:.2f} eq-pixels per output pixel). "
            f"Consider reducing --out-size or increasing --fov."
        )
    else:
        print(f"Source resolution: {eff_res:.2f} eq-pixels per output pixel")

    # Pre-build all remap tables (one per yaw×pitch combination)
    remaps: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
    for yaw in yaw_angles:
        for pitch in pitch_angles:
            remaps[(yaw, pitch)] = build_equirect_to_persp_map(
                eq_h, eq_w, out_h, out_w, fov_h_deg, yaw_deg=yaw, pitch_deg=pitch
            )

    written: list[Path] = []
    manifest: list[dict] = []
    frame_idx = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        # Split SBS into left/right equirectangular
        left_eq = bgr[:, :eq_w, :]
        right_eq = bgr[:, eq_w:, :]

        # --- GT depth from 3D tracks (if metadata available) ---
        # Pre-compute track positions in the LEFT camera frame for this video frame.
        # We reuse this across all yaw/pitch crops for the same frame.
        tracks_in_cam: np.ndarray | None = None
        if meta is not None and frame_idx < meta["n_frames"]:
            pose = meta["camera2world"][frame_idx]  # (3, 4)
            R_c2w = pose[:, :3]  # (3, 3) world columns
            t_w = pose[:, 3]  # (3,)   world position of left cam

            fs = int(meta["frame_starts"][frame_idx])
            fe = int(meta["frame_starts"][frame_idx + 1])
            if fe > fs:
                pts_world = meta["track_coords"][fs:fe].astype(np.float64)  # (N, 3)
                # World → left camera: P_cam = R_c2w^T @ (P_world - t_w)
                tracks_in_cam = ((pts_world - t_w) @ R_c2w).astype(np.float32)  # (N, 3)

        for yaw in yaw_angles:
            for pitch in pitch_angles:
                map_x, map_y = remaps[(yaw, pitch)]

                left_persp = remap_equirect(left_eq, map_x, map_y)
                right_persp = remap_equirect(right_eq, map_x, map_y)

                # Grayscale conversion (matching training pipeline: single-channel float in [0,1])
                left_gray = (
                    cv2.cvtColor(left_persp, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    / 255.0
                )
                right_gray = (
                    cv2.cvtColor(right_persp, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    / 255.0
                )

                # Add channel dim: (H, W, 1)
                left_gray = left_gray[..., np.newaxis]
                right_gray = right_gray[..., np.newaxis]

                # --- GT disparity ---
                # Priority: 3D-track projection > SGBM > zeros

                disparity = np.zeros((out_h, out_w), dtype=np.float32)
                valid_mask = np.zeros((out_h, out_w), dtype=bool)

                if tracks_in_cam is not None:
                    disparity, valid_mask = project_tracks_to_disparity(
                        tracks_in_cam,
                        out_h,
                        out_w,
                        f_x,
                        f_y,
                        yaw_deg=yaw,
                        pitch_deg=pitch,
                        baseline_m=baseline_m,
                    )

                if compute_disparity:
                    # SGBM needs uint8 input
                    l8 = (left_gray[..., 0] * 255).astype(np.uint8)
                    r8 = (right_gray[..., 0] * 255).astype(np.uint8)
                    sgbm_disp, sgbm_valid = compute_sgbm_disparity(
                        l8,
                        r8,
                        num_disparities=sgbm_num_disparities,
                        block_size=sgbm_block_size,
                        min_conf_ratio=1.0 if lr_check else 0.0,
                    )
                    # Merge: SGBM fills pixels without track GT
                    no_gt = ~valid_mask
                    disparity[no_gt] = sgbm_disp[no_gt]
                    valid_mask[no_gt] |= sgbm_valid[no_gt]

                has_any_disp = bool(valid_mask.any())

                # Build filename
                yaw_tag = f"yaw{int(yaw):+04d}"
                pitch_tag = f"pitch{int(pitch):+03d}"
                name = f"frame{frame_idx:05d}_{yaw_tag}_{pitch_tag}.npz"
                out_path = output_dir / name

                np.savez_compressed(
                    out_path,
                    left_image=left_gray,
                    right_image=right_gray,
                    disparity=disparity,
                    valid_mask=valid_mask.astype(np.uint8),  # bool→uint8 for npz compat
                    metadata=np.array(
                        [
                            str(video_path),
                            str(frame_idx),
                            str(yaw),
                            str(pitch),
                            str(fov_h_deg),
                        ]
                    ),
                )
                written.append(out_path)

                if export_hex:
                    hex_path = out_path.with_suffix(".mem")
                    export_fixed16_hex(left_gray, hex_path)

                manifest.append(
                    {
                        "source": "stereo_video_local",
                        "path": str(out_path),
                        "video": str(video_path),
                        "frame_idx": frame_idx,
                        "yaw_deg": yaw,
                        "pitch_deg": pitch,
                        "fov_h_deg": fov_h_deg,
                        "out_height": out_h,
                        "out_width": out_w,
                        "has_disparity": has_any_disp,
                        "n_track_pts": int(valid_mask.sum())
                        if tracks_in_cam is not None
                        else 0,
                        "hex_file": str(hex_path) if export_hex else None,
                    }
                )

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  frame {frame_idx}/{total_frames} — {len(written)} pairs written")

    cap.release()

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(written)} pairs → {output_dir}")
    print(f"Manifest: {manifest_path}")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract perspective stereo pairs from equirectangular SBS stereo video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--video", required=True, help="Input SBS equirectangular video (.mp4)"
    )
    ap.add_argument(
        "--output-dir", required=True, help="Directory for output .npz pairs + manifest"
    )
    ap.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Horizontal field of view of the perspective crop, degrees",
    )
    ap.add_argument(
        "--out-height",
        type=int,
        default=160,
        help="Output image height in pixels (default matches model training: 160)",
    )
    ap.add_argument(
        "--out-width",
        type=int,
        default=288,
        help="Output image width in pixels (default matches model training: 288)",
    )
    ap.add_argument(
        "--yaw-angles",
        type=float,
        nargs="+",
        default=[0.0],
        help="Azimuth angles to sample, degrees (0=forward, +90=right)",
    )
    ap.add_argument(
        "--pitch-angles",
        type=float,
        nargs="+",
        default=[0.0],
        help="Elevation angles to sample, degrees (0=level, +15=slightly up)",
    )
    ap.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame (1=all frames, 3=every 3rd, etc.)",
    )
    # GT from 3D tracks
    ap.add_argument(
        "--metadata",
        default=None,
        help="stereo-video.npz metadata file for GT depth from 3D tracks",
    )
    ap.add_argument(
        "--baseline",
        type=float,
        default=0.0635,
        help="Stereo baseline in metres (default 63.5 mm for VR180 rigs)",
    )
    # SGBM fallback
    ap.add_argument(
        "--compute-disparity",
        action="store_true",
        help="Supplement GT with SGBM disparity in track-sparse regions",
    )
    ap.add_argument(
        "--sgbm-num-disp",
        type=int,
        default=64,
        help="SGBM numDisparities (must be multiple of 16)",
    )
    ap.add_argument(
        "--sgbm-block", type=int, default=5, help="SGBM blockSize (odd, 3–11)"
    )
    ap.add_argument(
        "--lr-check",
        action="store_true",
        help="Left-right consistency check for tighter SGBM validity mask",
    )
    # Vivado simulation export
    ap.add_argument(
        "--export-hex",
        action="store_true",
        help="Write fixed<16,6> .mem hex files alongside .npz for Vivado $readmemh",
    )
    args = ap.parse_args()

    extract(
        video_path=args.video,
        output_dir=args.output_dir,
        fov_h_deg=args.fov,
        out_height=args.out_height,
        out_width=args.out_width,
        yaw_angles=args.yaw_angles,
        pitch_angles=args.pitch_angles,
        frame_stride=args.frame_stride,
        metadata_npz=args.metadata,
        baseline_m=args.baseline,
        compute_disparity=args.compute_disparity,
        sgbm_num_disparities=args.sgbm_num_disp,
        sgbm_block_size=args.sgbm_block,
        lr_check=args.lr_check,
        export_hex=args.export_hex,
    )


if __name__ == "__main__":
    main()
