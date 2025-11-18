import time
import cv2
import os
from utils.cloudinary_utils import upload_media

RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480
RECORD_DURATION = 3
OUTPUT_DIR = "recordings"

recording = False
record_buffer = []
record_start_time = 0
recording_suspects = set()
snapshot_frame = None
snapshot_taken = False


def start_recording(suspects):
    global recording, record_start_time, record_buffer
    global recording_suspects, snapshot_taken, snapshot_frame

    recording = True
    record_start_time = time.time()
    record_buffer = []
    recording_suspects = suspects.copy()
    snapshot_taken = False
    snapshot_frame = None

    print("[INFO] Recording started for:", suspects)


def stop_recording():
    global recording, record_buffer, recording_suspects, snapshot_frame

    recording = False
    duration = time.time() - record_start_time
    total_frames = len(record_buffer)
    if total_frames == 0:
        print("[WARN] No frames in buffer. Nothing to save.")
        return None

    actual_fps = total_frames / duration
    actual_fps = max(5, min(actual_fps, 30))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name_part = "_".join(sorted(list(recording_suspects))) or "unknown"
    filename = f"{OUTPUT_DIR}/{timestamp}_{name_part}.mp4"

    # first try avc1, fall back to mp4v if it fails
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(filename, fourcc, actual_fps,(RESIZE_WIDTH, RESIZE_HEIGHT))

    if not writer.isOpened():
        print("[WARN] avc1 not available. Falling back to mp4v.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filename, fourcc, actual_fps, RESIZE_WIDTH, RESIZE_HEIGHT)

    if not writer.isOpened():
        print("[ERROR] Could not open VideoWriter. Recording not saved.")
        return None

    for frame in record_buffer:
        writer.write(frame)
    writer.release()

    # choose snapshot if not already set
    if snapshot_frame is None:
        snapshot_frame = record_buffer[0]

    cloud_result = upload_media(
        video_path=filename,
        image_frame=snapshot_frame,
        suspects=recording_suspects
    )

    print(f"[INFO] Saved recording: {filename}")
    print(f"[INFO] Duration: {duration:.2f}s  Frames: {total_frames}  FPS: {actual_fps:.1f}")
    return cloud_result
