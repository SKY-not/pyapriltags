import cv2
from apriltag_pose import AprilTagPoseEstimator, RealSenseCamera

estimator = AprilTagPoseEstimator(
    intrinsics_yaml="d405_rgb_intrinsics.yaml",
    tag_size=0.01
)

cam = RealSenseCamera(estimator.w, estimator.h)

while True:

    img = cam.get_frame()

    if img is None:
        continue

    und, detections = estimator.detect(img)

    vis = estimator.draw(und, detections)

    cv2.imshow("tag", vis)

    for d in detections:
        print("tag", d["id"], "reproj", d["reproj"])

    if cv2.waitKey(1) == ord("q"):
        break

cam.stop()