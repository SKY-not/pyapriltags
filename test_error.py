#!/usr/bin/env python3
"""
main.py

实时读取 RealSense D405 彩色流，使用 module/apriltag_pose.py 中的
AprilTagPoseEstimator 进行检测、计算重投影误差并显示结果。
"""
import argparse
import time
import os
import sys

import cv2
import numpy as np

from module.apriltag_pose import AprilTagPoseEstimator, RealSenseCamera


def run(args):
    est = AprilTagPoseEstimator(args.intrinsics, args.tag_size, tag_family=args.family)

    print("Loaded camera intrinsics:")
    print(est.cam_mtx)

    cam = RealSenseCamera(est.w, est.h, fps=args.fps)

    window = "AprilTag - D405"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    last_time = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue

            und, detections = est.detect(frame)
            vis = est.draw(und, detections)

            # average reprojection error (px)
            reproj_vals = [d["reproj"] for d in detections if d.get("reproj") is not None]
            if len(reproj_vals) > 0:
                avg_reproj = float(np.mean(reproj_vals))
                cv2.putText(vis, f"avg reproj: {avg_reproj:.2f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # fps
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                fps = frame_count / (now - last_time)
                last_time = now
                frame_count = 0

            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"apriltag_{int(time.time())}.png"
                cv2.imwrite(fname, vis)
                print("Saved:", fname)

    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="RealSense D405 AprilTag realtime detector")
    p.add_argument("--intrinsics", "-i", default="d405_rgb_intrinsics.yaml",
                   help="path to intrinsics yaml (relative to module folder)")
    p.add_argument("--tag-size", "-t", type=float, default=0.05,
                   help="AprilTag size in meters (default 0.05)")
    p.add_argument("--family", "-f", default="tag36h11",
                   help="AprilTag family (default tag36h11)")
    p.add_argument("--fps", type=int, default=30, help="camera fps")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
