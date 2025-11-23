# Violence/ViolenceDetector.py

from ultralytics import YOLO

class ViolenceDetector:
    def __init__(self, model_path="Violence/best.pt", conf=0.45, stable_frames=5):
        self.model = YOLO(model_path)
        self.conf = conf
        
        # “violence” is always class ID 1 in your model
        self.violence_class_ids = [1]

        # Stability buffer
        self.stable_frames = stable_frames
        self.current_streak = 0      # frames with violence
        self.no_streak = 0           # frames with no violence

        self.last_boxes = []
        self.last_labels = []

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)

        raw_boxes = []
        raw_labels = []

        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue

            for b in r.boxes:
                cls = int(b.cls[0])

                # Keep only violence (class 1)
                if cls not in self.violence_class_ids:
                    continue

                x1, y1, x2, y2 = b.xyxy[0].tolist()
                raw_boxes.append((int(x1), int(y1), int(x2), int(y2)))
                raw_labels.append(self.model.names[cls])  # "violence"

        # STABILITY CHECK
        if len(raw_boxes) > 0:
            self.current_streak += 1
            self.no_streak = 0

            if self.current_streak >= self.stable_frames:
                self.last_boxes = raw_boxes
                self.last_labels = raw_labels
                return True, raw_boxes, raw_labels

            return False, [], []  # unstable streak not enough

        else:
            # No violence this frame
            self.no_streak += 1
            self.current_streak = 0

            # reset after a small no-detect run
            if self.no_streak >= 3:
                self.last_boxes = []
                self.last_labels = []

            return False, [], []
