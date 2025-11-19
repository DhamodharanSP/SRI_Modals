import time
import cv2
import os
from utils.cloudinary_utils import upload_media

RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480
RECORD_DURATION = 3
OUTPUT_DIR = "recordings"


class CameraRecorder:
    def __init__(self, cam_id):
        self.cam_id = cam_id

        self.recording = False
        self.record_buffer = []
        self.record_start_time = 0
        self.recording_suspects = set()
        self.snapshot_frame = None
        self.snapshot_taken = False

    def start(self, suspects):
        self.recording = True
        self.record_start_time = time.time()
        self.record_buffer = []
        self.recording_suspects = suspects.copy()
        self.snapshot_frame = None
        self.snapshot_taken = False

        print(f"[{self.cam_id}] Recording started for:", suspects)

    def stop(self):
        self.recording = False

        duration = time.time() - self.record_start_time
        total_frames = len(self.record_buffer)

        if total_frames == 0:
            print(f"[{self.cam_id}] No frames captured. Skipped.")
            return None

        actual_fps = total_frames / duration
        actual_fps = max(5, min(actual_fps, 30))

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name_part = "_".join(sorted(list(self.recording_suspects))) or "unknown"
        filename = f"{OUTPUT_DIR}/{self.cam_id}_{timestamp}_{name_part}.mp4"

        # safe fourcc
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(filename, fourcc, actual_fps, (RESIZE_WIDTH, RESIZE_HEIGHT))

        if not writer.isOpened():
            print(f"[{self.cam_id}] avc1 failed, using mp4v")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(filename, fourcc, actual_fps, (RESIZE_WIDTH, RESIZE_HEIGHT))

        for frame in self.record_buffer:
            writer.write(frame)
        writer.release()

        # ensure snapshot exists
        if self.snapshot_frame is None:
            self.snapshot_frame = self.record_buffer[0]
            #len(self.record_buffer)//2 -> Use this in case of retrieving middle frame

        cloud_result = upload_media(
            video_path=filename,
            image_frame=self.snapshot_frame,
            suspects=self.recording_suspects,
            cam_id=self.cam_id,
        )

        print(f"[{self.cam_id}] Saved recording: {filename}")

        return cloud_result
