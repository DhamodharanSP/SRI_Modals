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

def start_recording(suspects):
    global recording, record_start_time, record_buffer, recording_suspects

    recording = True
    record_start_time = time.time()
    record_buffer = []
    recording_suspects = suspects.copy()

    print("[INFO] Recording started for:", suspects)

def stop_recording():
    global recording, record_buffer, recording_suspects

    recording = False
    duration = time.time() - record_start_time
    total_frames = len(record_buffer)
    if total_frames == 0:
        return None

    actual_fps = total_frames / duration
    actual_fps = max(5, min(actual_fps, 30))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name_part = "_".join(sorted(list(recording_suspects)))
    filename = f"{OUTPUT_DIR}/{timestamp}_{name_part}.mp4"

    # writer
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(filename, fourcc, actual_fps,(RESIZE_WIDTH, RESIZE_HEIGHT))
    for frame in record_buffer:
        writer.write(frame)
    writer.release()

    # pick snapshot
    snapshot = record_buffer[len(record_buffer)//2]

    # upload to cloudinary
    cloud_result = upload_media(filename, snapshot, recording_suspects)

    print(f"[INFO] Saved recording: {filename}")
    return cloud_result
