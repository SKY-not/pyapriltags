#!/usr/bin/env python3
"""
Calibrate Intel RealSense D405 camera (capture, calibrate, undistort)

Usage examples:
  # Capture 20 images to ./calib_images
  python calibrate_d405.py capture --out calib_images --count 20

  # Calibrate from images
  python calibrate_d405.py calibrate --images "calib_images/*.png" --rows 6 --cols 9 --square 25

  # Undistort an image using saved params
  python calibrate_d405.py undistort --input img.png --params camera_params.yaml

Requires: pyrealsense2, opencv-python, numpy, pyyaml
"""
import argparse
import glob
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml


def capture_images(output_dir: str, count: int = 20, width: int = 1280, height: int = 720, fps: int = 30):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)

    print(f"Started RealSense stream ({width}x{height} @{fps}fps). Press SPACE to save a frame, ESC to quit.")
    saved = 0
    try:
        while saved < count:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())

            cv2.putText(img, f"Saved: {saved}/{count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Realsense Capture", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == 32 or key == ord("c"):  # SPACE or 'c'
                filename = os.path.join(output_dir, f"img_{saved:03d}.png")
                cv2.imwrite(filename, img)
                print(f"Saved {filename}")
                saved += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def find_corners(images, pattern_size):
    objpoints = []
    imgpoints = []

    rows, cols = pattern_size
    # prepare object points, like (0,0,0), (1,0,0), ... multiplied by square size later
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    good_images = []
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
            imgpoints.append(corners2)
            objpoints.append(objp)
            good_images.append(fname)
        else:
            print(f"Chessboard not found in {fname}")

    return objpoints, imgpoints, good_images


def calibrate_camera(images_glob: str, rows: int, cols: int, square_size_mm: float, save_path: str):
    images = sorted(glob.glob(images_glob))
    if not images:
        raise RuntimeError("No images found for calibration")

    objpoints, imgpoints, good_images = find_corners(images, (rows, cols))
    if len(objpoints) < 5:
        raise RuntimeError(f"Not enough valid calibration images ({len(objpoints)}). Need at least 5.")

    # scale object points by square size (convert mm -> meters optional; keep mm consistent)
    square = float(square_size_mm)
    objpoints = [op * square for op in objpoints]

    img = cv2.imread(good_images[0])
    h, w = img.shape[:2]

    ret, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

    params = {
        "image_width": int(w),
        "image_height": int(h),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coefs": dist_coefs.tolist(),
        "new_camera_matrix": newcameramtx.tolist(),
        "reprojection_error": float(ret),
        "pattern_rows": int(rows),
        "pattern_cols": int(cols),
        "square_size_mm": float(square_size_mm),
    }

    with open(save_path, "w") as f:
        yaml.safe_dump(params, f)

    print(f"Calibration saved to {save_path}")
    print(f"Reprojection error: {ret:.4f}")

    return params


def undistort_image(input_path: str, params_path: str, output_path: str = None):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    cam = np.array(params["camera_matrix"])
    dist = np.array(params["dist_coefs"])
    newcam = np.array(params.get("new_camera_matrix", params["camera_matrix"]))

    img = cv2.imread(input_path)
    h, w = img.shape[:2]
    und = cv2.undistort(img, cam, dist, None, newcam)

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_undistorted{ext}"

    cv2.imwrite(output_path, und)
    print(f"Saved undistorted image to {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="RealSense D405 calibration helper")
    sub = p.add_subparsers(dest="cmd")

    cap = sub.add_parser("capture", help="Capture images from RealSense D405")
    cap.add_argument("--out", required=True, help="Output directory for images")
    cap.add_argument("--count", type=int, default=20)
    cap.add_argument("--width", type=int, default=1280)
    cap.add_argument("--height", type=int, default=720)
    cap.add_argument("--fps", type=int, default=30)

    cal = sub.add_parser("calibrate", help="Calibrate from captured images")
    cal.add_argument("--images", required=True, help="Glob pattern for calibration images, e.g. 'calib/*.png'")
    cal.add_argument("--rows", type=int, required=True, help="Number of inner corners per chessboard row (vertical)")
    cal.add_argument("--cols", type=int, required=True, help="Number of inner corners per chessboard column (horizontal)")
    cal.add_argument("--square", type=float, default=25.0, help="Square size in mm")
    cal.add_argument("--out", default="camera_params.yaml", help="Path to save camera params YAML")

    und = sub.add_parser("undistort", help="Undistort an image using saved params")
    und.add_argument("--input", required=True)
    und.add_argument("--params", required=True)
    und.add_argument("--out", default=None)

    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == "capture":
        capture_images(args.out, count=args.count, width=args.width, height=args.height, fps=args.fps)
    elif args.cmd == "calibrate":
        calibrate_camera(args.images, args.rows, args.cols, args.square, args.out)
    elif args.cmd == "undistort":
        undistort_image(args.input, args.params, args.out)
    else:
        print("No command specified. Use --help for usage.")


if __name__ == "__main__":
    main()
