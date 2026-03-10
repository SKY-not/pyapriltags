#!/usr/bin/env python3
"""
apriltag_pose.py

AprilTag pose estimation module.

Features
--------
- Load camera intrinsics from YAML
- Detect AprilTags using pyapriltags
- Compute pose (rvec, tvec)
- Compute reprojection error
- Optional RealSense frame capture
- Optional hand-eye transform

Author: your_name
"""

import yaml
import numpy as np
import cv2
import os

try:
    import pyapriltags
except Exception:
    raise RuntimeError("Please install pyapriltags")

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


class AprilTagPoseEstimator:

    def __init__(self,
                 intrinsics_yaml,
                 tag_size,
                 tag_family="tag36h11",
                 handeye=None):
        """
        Parameters
        ----------
        intrinsics_yaml : str
            camera intrinsics yaml
        tag_size : float
            tag size (meters)
        handeye : 4x4 matrix
            optional camera->robot transform
        """

        self.cam_mtx, self.dist, (self.w, self.h) = self.load_intrinsics(intrinsics_yaml)
        self.tag_size = tag_size

        self.detector = pyapriltags.Detector(families=tag_family)

        self.handeye = handeye

        # precompute undistortion map
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.cam_mtx,
            self.dist,
            None,
            self.cam_mtx,
            (self.w, self.h),
            cv2.CV_32FC1
        )

        # tag object coordinates
        s = tag_size
        self.obj_pts = np.array([
            [-s/2,  s/2, 0],
            [ s/2,  s/2, 0],
            [ s/2, -s/2, 0],
            [-s/2, -s/2, 0]
        ], dtype=np.float32)

    def load_intrinsics(self, path):
        if not os.path.isabs(path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base_dir, path)
            
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cam_mtx = np.array(
            data.get("camera_matrix", data.get("cameraMatrix")),
            dtype=np.float64
        )

        dist = np.array(
            data.get("dist_coeffs", data.get("distCoeffs")),
            dtype=np.float64
        ).ravel()

        w = int(data.get("image_width", data.get("width")))
        h = int(data.get("image_height", data.get("height")))

        return cam_mtx, dist, (w, h)

    def undistort(self, img):
        return cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

    def detect(self, img):
        und = self.undistort(img)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

        fx = self.cam_mtx[0, 0]
        fy = self.cam_mtx[1, 1]
        cx = self.cam_mtx[0, 2]
        cy = self.cam_mtx[1, 2]

        results = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=self.tag_size
        )

        detections = []

        for r in results:
            corners = np.array(r.corners, dtype=np.float32)
            rvec = None
            tvec = None
            if hasattr(r, "pose_R"):
                rvec, _ = cv2.Rodrigues(r.pose_R)
                tvec = np.array(r.pose_t).reshape(3)
            reproj = None

            if rvec is not None:
                proj, _ = cv2.projectPoints(
                    self.obj_pts,
                    rvec,
                    tvec,
                    self.cam_mtx,
                    np.zeros((5,))
                )
                proj = proj.reshape(-1, 2)
                reproj = float(
                    np.sqrt(
                        np.mean(np.sum((proj - corners) ** 2, axis=1))
                    )
                )
            T_cam_tag = None
            if rvec is not None:
                R, _ = cv2.Rodrigues(rvec)
                T_cam_tag = np.eye(4)
                T_cam_tag[:3, :3] = R
                T_cam_tag[:3, 3] = tvec
                if self.handeye is not None:
                    T_cam_tag = self.handeye @ T_cam_tag
            detections.append({
                "id": int(r.tag_id),
                "corners": corners,
                "rvec": rvec,
                "tvec": tvec,
                "T": T_cam_tag,
                "reproj": reproj
            })
        return und, detections

    def draw(self, img, detections):

        out = img.copy()

        for d in detections:
            corners = d["corners"]
            pts = corners.astype(int).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, (0, 255, 0), 2)
            cxy = tuple(corners.mean(axis=0).astype(int))
            text = f"id={d['id']} reproj={d['reproj']:.2f}px"
            cv2.putText(
                out,
                text,
                (cxy[0] - 50, cxy[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            
        return out

class RealSenseCamera:

    def __init__(self, width, height, fps=30):

        if rs is None:
            raise RuntimeError("pyrealsense2 not installed")

        self.pipe = rs.pipeline()
        self.cfg = rs.config()

        self.cfg.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.bgr8,
            fps
        )

        self.pipe.start(self.cfg)

        for _ in range(30):
            self.pipe.wait_for_frames()

    def get_frame(self):

        frames = self.pipe.wait_for_frames()
        color = frames.get_color_frame()

        if not color:
            return None

        return np.asanyarray(color.get_data())

    def stop(self):
        self.pipe.stop()