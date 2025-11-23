from ultralytics import YOLO
import cv2

class WeaponDetector:
    def __init__(self, model_path="best_weapon.pt", conf=0.45):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        boxes = []
        labels = []

        for r in results:
            if r.boxes is None:
                continue

            for b in r.boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                x1, y1, x2, y2 = b.xyxy[0].tolist()

                boxes.append((int(x1), int(y1), int(x2), int(y2)))
                labels.append(self.model.names[cls])

        present = len(boxes) > 0
        return present, boxes, labels
