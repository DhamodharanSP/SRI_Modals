import cv2

def blur_region(frame, box, pixelation=20):
    x1, y1, x2, y2 = [int(c) for c in box]
    roi = frame[y1:y2, x1:x2]

    if roi.size != 0:
        h, w = roi.shape[:2]
        # Prevent zero dimensions
        new_w = max(1, w // pixelation)
        new_h = max(1, h // pixelation)

        roi_small = cv2.resize(roi, (new_w, new_h))
        roi_pixelated = cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = roi_pixelated

    return frame


def draw_box(frame, box, name, score=None, color=(255,0,0)):
    x1, y1, x2, y2 = [int(c) for c in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{name}" + (f" ({score:.2f})" if score is not None else "")
    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame

