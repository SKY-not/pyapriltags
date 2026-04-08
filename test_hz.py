import pyrealsense2 as rs
import numpy as np
import cv2
import time
import csv
from pyapriltags import Detector

# Camera configuration
WIDTH = 1280
HEIGHT = 720
FPS = 30
TAG_SIZE = 0.010  # 10 mm (meters)
TAG_FAMILY = "tag36h11"
OUTPUT_CSV = "apriltag_detections.csv"

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

# Start pipeline
profile = pipeline.start(config)

# Get camera intrinsics
color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()
fx, fy = intr.fx, intr.fy
cx, cy = intr.ppx, intr.ppy

# Initialize AprilTag detector
detector = Detector(
    families=TAG_FAMILY,
    nthreads=2,
    quad_decimate=2.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

# Open CSV file for writing
with open(OUTPUT_CSV, mode='w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["timestamp", "Tag ID", "x", "y", "z", "qx", "qy", "qz", "qw"])

    # Variables to track frame rates
    total_frames = 0
    processed_frames = 0
    fps_start_time = time.time()

    print("[INFO] Starting detection for 5 seconds...")
    start_time = time.time()

    try:
        while time.time() - start_time < 10:
            frames = pipeline.wait_for_frames()
            total_frames += 1  # Count every frame received

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert frame to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            color_image_undistorted = cv2.undistort(color_image, 
                                                   np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), 
                                                   np.array(intr.coeffs))
            gray_image_undistorted = cv2.cvtColor(color_image_undistorted, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags on the undistorted image
            detections = detector.detect(
                gray_image_undistorted,
                estimate_tag_pose=True,
                camera_params=(fx, fy, cx, cy),
                tag_size=TAG_SIZE
            )

            if detections:
                processed_frames += 1  # Count frames where AprilTags are detected

            for det in detections:
                t = det.pose_t.ravel()  # Translation vector (x, y, z)
                R = det.pose_R  # Rotation matrix

                # Calculate timestamp relative to start time
                timestamp = time.time() - start_time

                # Write detection to CSV
                csvwriter.writerow([timestamp, det.tag_id, t[0], t[1], t[2], R[0][0], R[1][1], R[2][2], R[2][1]])

                # Visualize detection
                corners = det.corners.astype(int)
                for i in range(4):
                    cv2.line(color_image_undistorted, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
                cv2.putText(
                    color_image_undistorted,
                    f"ID: {det.tag_id}",
                    (corners[0][0], corners[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

            # Display the undistorted image
            cv2.imshow("AprilTag Detection", color_image_undistorted)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

        # Calculate and print frame rates
        elapsed_time = time.time() - fps_start_time
        actual_fps = total_frames / elapsed_time
        processed_fps = processed_frames / elapsed_time
        print(f"[INFO] Actual FPS: {actual_fps:.2f}")
        print(f"[INFO] AprilTag Processed FPS: {processed_fps:.2f}")
        print(f"[INFO] Detection finished. Results saved to {OUTPUT_CSV}")