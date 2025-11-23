"""
pose_detector.py

Stage 1: Pose (Skeleton) Extraction for Behavior Anomaly Detection.
- Wraps Ultralytics YOLO Pose model (YOLO11 or YOLOv8 pose).
- Outputs per-person skeletons (keypoints) + bounding boxes for the next GNN stage.

Requirements:
    pip install ultralytics opencv-python torch numpy
"""

from typing import List, Dict, Optional
import numpy as np
import torch
import cv2
from ultralytics import YOLO


class PoseDetector:
    """
    Thin wrapper around Ultralytics YOLO Pose model.

    You can use:
        - "yolo11n-pose.pt" / "yolo11s-pose.pt"   (YOLO11 pose)
        - "yolov8n-pose.pt" / "yolov8s-pose.pt"   (YOLOv8 pose)

    For your RTX 3050 4GB:
        Recommended starting point: "yolo11s-pose.pt" or "yolov8s-pose.pt"
    """

    def __init__(
        self,
        model_path: str = "yolov8s-pose.pt",  # change to "yolov8s-pose.pt" if needed
        device: Optional[str] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.5,
    ):
        """
        Args:
            model_path: Path or name of the YOLO pose model.
            device: "cuda", "cpu", or None to auto-select.
            conf_thres: Confidence threshold for detections.
            iou_thres: IoU threshold for NMS.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Load YOLO pose model
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Basic preprocessing. Here it's just BGR->RGB conversion if needed by you later.
        Currently Ultralytics handles preprocessing internally, so we just return frame.
        """
        # If you want to enforce RGB internally, uncomment:
        # return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def detect_poses(
        self,
        frame: np.ndarray,
    ) -> List[Dict]:
        """
        Run pose detection on a single frame.

        Args:
            frame: BGR image (H, W, 3) from OpenCV.

        Returns:
            List of dicts, one per detected person:
                {
                    "bbox": [x1, y1, x2, y2],
                    "score": float,  # detection confidence
                    "keypoints": [[x, y], ...],          # list of (K, 2)
                    "keypoint_scores": [c1, c2, ...],    # list of (K,)
                }

            This format is ready to be converted into graph nodes/edges for the GNN.
        """
        img = self._preprocess(frame)

        # Run model
        results = self.model(
            img,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
        )

        if len(results) == 0:
            return []

        res = results[0]

        if res.keypoints is None or res.boxes is None:
            return []

        keypoints_xy = res.keypoints.xy  # (num_persons, num_kpts, 2)
        keypoints_conf = res.keypoints.conf  # (num_persons, num_kpts) or None
        boxes_xyxy = res.boxes.xyxy  # (num_persons, 4)
        boxes_conf = res.boxes.conf  # (num_persons,)

        detections: List[Dict] = []

        num_persons = keypoints_xy.shape[0]

        for i in range(num_persons):
            # BBox
            bbox = boxes_xyxy[i].detach().cpu().numpy().tolist()
            score = float(boxes_conf[i].detach().cpu().item())

            # Keypoints
            kpts_xy = keypoints_xy[i].detach().cpu().numpy()  # (K, 2)
            if keypoints_conf is not None:
                kpts_conf = keypoints_conf[i].detach().cpu().numpy()  # (K,)
            else:
                kpts_conf = np.ones((kpts_xy.shape[0],), dtype=np.float32)

            detections.append(
                {
                    "bbox": bbox,
                    "score": score,
                    "keypoints": kpts_xy.tolist(),
                    "keypoint_scores": kpts_conf.tolist(),
                }
            )

        return detections

    @staticmethod
    def draw_poses(
        frame: np.ndarray,
        detections: List[Dict],
        draw_bbox: bool = True,
        draw_skeleton: bool = True,
    ) -> np.ndarray:
        """
        Utility for debugging/visualization.
        Draws bounding boxes and skeletons on the frame.

        NOTE:
            - Skeleton connections here assume COCO-style 17 keypoints.
            - If your model uses different layout, adjust the pairs accordingly.
        """

        # COCO-style skeleton pairs (indexing starts from 0)
        skeleton_pairs = [
            (5, 7), (7, 9),     # left arm
            (6, 8), (8, 10),    # right arm
            (11, 13), (13, 15), # left leg
            (12, 14), (14, 16), # right leg
            (5, 6),             # shoulders
            (11, 12),           # hips
            (5, 11), (6, 12),   # body
        ]

        out = frame.copy()

        for det in detections:
            bbox = det["bbox"]
            keypoints = det["keypoints"]

            # Draw bbox
            if draw_bbox:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if draw_skeleton:
                # Draw joints
                for (x, y) in keypoints:
                    cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)

                # Draw limbs
                for j1, j2 in skeleton_pairs:
                    if j1 < len(keypoints) and j2 < len(keypoints):
                        x1, y1 = keypoints[j1]
                        x2, y2 = keypoints[j2]
                        cv2.line(
                            out,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (255, 0, 0),
                            2,
                        )

        return out


if __name__ == "__main__":
    """
    Quick test using webcam:

    python pose_detector.py
    """
    cap = cv2.VideoCapture(1)

    detector = PoseDetector(
        model_path="yolov8n-pose.pt",  # or "yolov8s-pose.pt"
        conf_thres=0.5,
        iou_thres=0.5,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_poses(frame)
        vis = detector.draw_poses(frame, detections)

        cv2.imshow("Pose Detector", vis)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
