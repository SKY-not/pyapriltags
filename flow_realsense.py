#!/usr/bin/env python3
"""
flow_realsense.py
- 使用 Intel RealSense 获取图像
- 使用相机内参去畸变
- 使用 Apriltag 检测并基于四角点用 solvePnP 估计位姿
- 计算并输出重投影误差

用法示例：
python flow_realsense.py --intrinsics d405_rgb_intrinsics.yaml --tag-size 0.01
"""

import argparse
import sys
import time
import yaml

import numpy as np
import cv2
import glob
import os
import json

try:
    import pyrealsense2 as rs
except Exception as e:
    rs = None

# Try apriltag detectors (support multiple wrappers)
_detector_impl = None
_pyapriltag = None
try:
    import pyapriltags as pyapriltag
    _pyapriltag = pyapriltag
    _detector_impl = 'pyapriltag'
except Exception:
    raise RuntimeError('No apriltag detector installed. Install `pyapriltags`.')


def load_intrinsics(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    cm = np.array(data.get('camera_matrix', data.get('cameraMatrix', [])), dtype=np.float64)
    dist = np.array(data.get('dist_coeffs', data.get('distCoeffs', [])), dtype=np.float64)
    h = int(data.get('image_height', data.get('height', 0)))
    w = int(data.get('image_width', data.get('width', 0)))
    return cm, dist.ravel() if dist.size else np.zeros((5,)), (w, h)


def build_detector(family='tag36h11'):
    if _detector_impl == 'pyapriltag' and _pyapriltag is not None:
        det = _pyapriltag.Detector()
        return ('pyapriltag', det)
    raise RuntimeError('No apriltag detector installed. Install `pyapriltag`.')


def detect_tags(detector_tuple, gray):
    impl, det = detector_tuple
    results = det.detect(gray)
    detections = []
    for r in results:
        # corners may be attribute or key depending on implementation
        corners = None
        tag_id = None
        if hasattr(r, 'corners'):
            corners = np.array(r.corners, dtype=np.float32)
        elif isinstance(r, dict):
            if 'corners' in r:
                corners = np.array(r['corners'], dtype=np.float32)
            elif 'pts' in r:
                corners = np.array(r['pts'], dtype=np.float32)
        else:
            try:
                corners = np.array(getattr(r, 'pts', None), dtype=np.float32)
            except Exception:
                corners = None

        if hasattr(r, 'tag_id'):
            tag_id = int(getattr(r, 'tag_id'))
        elif hasattr(r, 'id'):
            tag_id = int(getattr(r, 'id'))
        elif isinstance(r, dict):
            tag_id = int(r.get('id', r.get('tag_id', -1)))

        if corners is None:
            continue
        corners = corners.reshape((4,2))
        detections.append({'id': int(tag_id), 'corners': corners})
    return detections


def estimate_pose_and_reproj(tag_corners, cam_mtx, tag_size_m, dist=np.zeros((5,))):
    # tag_corners: 4x2 array in image coordinates. We assume order corresponds to
    # tag's corners: (top-left, top-right, bottom-right, bottom-left) but solvePnP
    # works regardless if the order matches the object points order.
    s = float(tag_size_m)
    obj_pts = np.array([[-s/2, -s/2, 0.0],
                        [ s/2, -s/2, 0.0],
                        [ s/2,  s/2, 0.0],
                        [-s/2,  s/2, 0.0]], dtype=np.float32)
    img_pts = tag_corners.reshape((4,2)).astype(np.float32)
    # Attempt solvePnP
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cam_mtx, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam_mtx, dist)
    proj_pts = proj_pts.reshape((-1,2))
    err = np.sqrt(np.mean(np.sum((proj_pts - img_pts)**2, axis=1)))
    return rvec.reshape(3,), tvec.reshape(3,), float(err)

def get_transformation_matrix(rvec, tvec):
    # 1. 将旋转向量 (3x1) 转换为旋转矩阵 (3x3)
    rmat, _ = cv2.Rodrigues(rvec)
    
    # 2. 创建 4x4 的单位矩阵
    T = np.eye(4)
    
    # 3. 将旋转矩阵填入左上角 3x3 区域
    T[:3, :3] = rmat
    
    # 4. 将平移向量填入右上角 3x1 区域
    # 确保 tvec 是 (3,) 或 (3,1) 形状
    T[:3, 3] = tvec.reshape(3)
    
    return T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics', default='d405_rgb_intrinsics.yaml', help='YAML intrinsics path')
    parser.add_argument('--tag-size', type=float, default=0.01, help='Tag size in meters')
    parser.add_argument('--tag-family', default='tag36h11', help='AprilTag family')
    parser.add_argument('--once', action='store_true', help='Capture a single frame and exit')
    parser.add_argument('--images', help='If set, process images from this folder instead of RealSense')
    parser.add_argument('--out', default='out_vis', help='Output folder when processing images')
    args = parser.parse_args()

    cam_mtx, dist, (w,h) = load_intrinsics(args.intrinsics)
    print('Loaded intrinsics:', cam_mtx.shape, dist.shape, (w,h))

    detector = build_detector(args.tag_family)
    print('Using detector:', detector[0])
    # If images folder provided, run offline processing
    if args.images:
        images_dir = args.images
        out_dir = args.out
        os.makedirs(out_dir, exist_ok=True)
        # collect images
        images = []
        for ext in ('*.png','*.jpg','*.jpeg','*.bmp','*.pnm'):
            images.extend(sorted(glob.glob(os.path.join(images_dir, ext))))
        if not images:
            print('No images found in', images_dir)
            return

        map1, map2 = cv2.initUndistortRectifyMap(cam_mtx, dist, None, cam_mtx, (w,h), cv2.CV_32FC1)
        summary = []
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                print('failed read', img_path)
                continue
            und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
            detections = detect_tags(detector, gray)
            out = und.copy()
            per_image = {'file': os.path.basename(img_path), 'detections': []}
            for det in detections:
                corners = det['corners']
                pts = corners.astype(np.int32).reshape((-1,1,2))
                cv2.polylines(out, [pts], True, (0,255,0), 2)
                try:
                    rvec, tvec, reproj = estimate_pose_and_reproj(corners, cam_mtx, args.tag_size)
                except Exception as e:
                    print('pose failed', det.get('id'), 'err', e)
                    continue
                cxy = tuple(corners.mean(axis=0).astype(int).tolist())
                text = f"id={det['id']} reproj={reproj:.2f}px"
                cv2.putText(out, text, (cxy[0]-50, cxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                per_image['detections'].append({'id': int(det['id']), 'rvec': rvec.tolist(), 'tvec': tvec.tolist(), 'reproj_px': reproj})
                print(f"{os.path.basename(img_path)}: Tag {det['id']} rvec={rvec.tolist()} tvec={tvec.tolist()} reproj_px={reproj:.3f}")
            out_path = os.path.join(out_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, out)
            summary.append(per_image)
        with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print('Processed', len(images), 'images. Visualizations saved to', out_dir)
        return

    # Otherwise run RealSense live capture
    if rs is None:
        print('pyrealsense2 not available. Exiting.')
        sys.exit(1)

    # Configure RealSense
    pipe = rs.pipeline()
    cfg = rs.config()
    try:
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
    except Exception:
        # fallback to default
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipe.start(cfg)

    try:
        # prepare undistort maps
        map1, map2 = cv2.initUndistortRectifyMap(cam_mtx, dist, None, cam_mtx, (w,h), cv2.CV_32FC1)

        while True:
            frames = pipe.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)

            detections = detect_tags(detector, gray)
            out = und.copy()
            for det in detections:
                corners = det['corners']  # shape (4,2)
                # draw
                pts = corners.astype(np.int32).reshape((-1,1,2))
                cv2.polylines(out, [pts], True, (0,255,0), 2)
                # estimate pose
                try:
                    rvec, tvec, reproj = estimate_pose_and_reproj(corners, cam_mtx, args.tag_size)
                except Exception as e:
                    print('pose failed for id', det.get('id'), 'err', e)
                    continue
                # Draw id and error
                cxy = tuple(corners.mean(axis=0).astype(int).tolist())
                text = f"id={det['id']} reproj={reproj:.2f}m"
                cv2.putText(out, text, (cxy[0]-50, cxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                # print pose
                print(f"Tag {det['id']}: rvec={rvec.tolist()} tvec={tvec.tolist()} reproj_px={reproj:.3f}")

            cv2.imshow('undistorted', out)
            key = cv2.waitKey(1) & 0xFF
            if args.once:
                # save image and break
                ts = int(time.time())
                cv2.imwrite(f'apriltag_capture_{ts}.png', out)
                break
            if key == ord('q'):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
