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


def load_intrinsics(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    cm = np.array(data.get('camera_matrix', data.get('cameraMatrix', [])), dtype=np.float64)
    dist = np.array(data.get('dist_coeffs', data.get('distCoeffs', [])), dtype=np.float64)
    h = int(data.get('image_height', data.get('height', 0)))
    w = int(data.get('image_width', data.get('width', 0)))
    return cm, dist.ravel() if dist.size else np.zeros((5,)), (w, h)


def build_detector(family:str='tag36h11'):
    if _detector_impl == 'pyapriltag' and _pyapriltag is not None:
        det = _pyapriltag.Detector()
        return ('pyapriltag', det)
    raise RuntimeError('No apriltag detector installed. Install `pyapriltag`.')


def detect_tags(detector_tuple: tuple, gray: np.ndarray, cam_mtx: np.ndarray=None, tag_size: float=None):
    """
    Detect AprilTags and return unified detections including pose if available.
    """
    impl, det = detector_tuple
    fx = cam_mtx[0,0]
    fy = cam_mtx[1,1]
    cx = cam_mtx[0,2]
    cy = cam_mtx[1,2]
    results = det.detect(gray, estimate_tag_pose=True, camera_params=(fx,fy,cx,cy), tag_size=tag_size)
    detections = []

    for r in results:
        corners = None
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

        if corners is None:
            continue
        corners = corners.reshape((4,2))

        tag_id = None
        if hasattr(r, 'tag_id'):
            tag_id = int(getattr(r, 'tag_id'))
        elif hasattr(r, 'id'):
            tag_id = int(getattr(r, 'id'))
        elif isinstance(r, dict):
            tag_id = int(r.get('id', r.get('tag_id', -1)))

        rvec = None
        tvec = None
        if hasattr(r, 'rvec') and hasattr(r, 'tvec'):
            rvec = np.array(r.rvec, dtype=np.float32).reshape(3,)
            tvec = np.array(r.tvec, dtype=np.float32).reshape(3,)
        elif hasattr(r, 'pose_R') and hasattr(r, 'pose_t'):
            rvec, _ = cv2.Rodrigues(np.array(r.pose_R, dtype=np.float32))
            tvec = np.array(r.pose_t, dtype=np.float32).reshape(3,)

        detections.append({
            'id': int(tag_id),
            'corners': corners,
            'rvec': rvec,
            'tvec': tvec
        })

    return detections

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics', default='d405_rgb_intrinsics.yaml', help='YAML intrinsics path')
    parser.add_argument('--tag-size', type=float, default=0.01, help='Tag size in meters')
    parser.add_argument('--tag-family', default='tag36h11', help='AprilTag family')
    parser.add_argument('--once', action='store_true', help='Capture a single frame and exit')
    # default='imgs'
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
            detections = detect_tags(detector, gray, cam_mtx=cam_mtx, tag_size=args.tag_size)
            out = und.copy()
            per_image = {'file': os.path.basename(img_path), 'detections': []}

            for det in detections:
                corners = det['corners']
                pts = corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts], True, (0, 255, 0), 2)

                # 使用Detection自带的位姿
                rvec = det['rvec']
                tvec = det['tvec']

                # 如果Detection里没有rvec/tvec，跳过
                if rvec is None or tvec is None:
                    print('pose missing for tag', det['id'])
                    continue

                # 重投影误差计算
                proj_pts, _ = cv2.projectPoints(
                    np.array([[-args.tag_size/2, args.tag_size/2, 0],
                            [ args.tag_size/2, args.tag_size/2, 0],
                            [ args.tag_size/2,  -args.tag_size/2, 0],
                            [-args.tag_size/2,  -args.tag_size/2, 0]], dtype=np.float32),
                    rvec,
                    tvec,
                    cam_mtx,
                    np.zeros((5,))
                )
                proj_pts = proj_pts.reshape((-1, 2))
                reproj = float(np.sqrt(np.mean(np.sum((proj_pts - corners)**2, axis=1))))

                # 在图像上标注
                cxy = tuple(corners.mean(axis=0).astype(int).tolist())
                text = f"id={det['id']} reproj={reproj:.2f}px"
                cv2.putText(out, text, (cxy[0]-50, cxy[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 保存信息
                per_image['detections'].append({
                    'id': int(det['id']),
                    'rvec': rvec.tolist(),
                    'tvec': tvec.tolist(),
                    'reproj_px': reproj
                })
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
    device = profile.get_device()
    color_sensor = None
    for s in device.sensors:
        if s.supports(rs.option.enable_auto_exposure):
            color_sensor = s
            break
    if color_sensor is None:
        raise RuntimeError("No color sensor found")
    if color_sensor.supports(rs.option.enable_auto_exposure):
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    pipe.stop()
    time.sleep(0.5)
    pipe.start(cfg)

    print("[INFO] Restored auto exposure (default)")
    # 等待自动曝光稳定
    for _ in range(30):
        pipe.wait_for_frames()

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

            detections = detect_tags(detector, gray, cam_mtx=cam_mtx, tag_size=args.tag_size)
            out = und.copy()
            for det in detections:
                corners = det['corners']  # shape (4,2)
                pts = corners.astype(np.int32).reshape((-1,1,2))
                cv2.polylines(out, [pts], True, (0,255,0), 2)
                
                rvec = det['rvec']
                tvec = det['tvec']
                if rvec is None or tvec is None:
                    print('pose missing for tag', det['id'])
                    continue

                obj_pts = np.array([[-args.tag_size/2, args.tag_size/2, 0],
                                    [ args.tag_size/2, args.tag_size/2, 0],
                                    [ args.tag_size/2, -args.tag_size/2, 0],
                                    [-args.tag_size/2, -args.tag_size/2, 0]], dtype=np.float32)
                proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, cam_mtx, np.zeros((5,)))
                proj_pts = proj_pts.reshape((-1,2))
                reproj = float(np.sqrt(np.mean(np.sum((proj_pts - corners)**2, axis=1))))

                cxy = tuple(corners.mean(axis=0).astype(int).tolist())
                text = f"id={det['id']} reproj={reproj:.2f}px"
                cv2.putText(out, text, (cxy[0]-50, cxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                print(f"Tag {det['id']}: rvec={rvec.tolist()} tvec={tvec.tolist()} reproj_px={reproj:.3f}")

            cv2.imshow('undistorted', out)
            key = cv2.waitKey(1) & 0xFF
            if args.once:
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
