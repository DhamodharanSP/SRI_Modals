import cv2
import time
import threading
from playsound import playsound
from FRT.FaceRecognition import load_known_faces, recognize_faces
from utils.video_utils import blur_region, draw_box
from utils.recording_utils import CameraRecorder, RESIZE_WIDTH, RESIZE_HEIGHT, RECORD_DURATION

COOLDOWN = 10
ALERT_SOUND = "sounds/alert.mp3"


def play_audio_once():
    playsound(ALERT_SOUND)

def process_camera(cam_id, source, embeddings, names):
    recorder = CameraRecorder(cam_id)

    # per-camera tracking
    active_suspects = {}             # suspect -> last_seen_time
    recently_recorded = {}           # suspect -> last_recorded_time

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[{cam_id}] Cannot open source:", source)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        resized_frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

        # run face recognition
        recognized = recognize_faces(frame, embeddings, names)
        detected_now = set()  # suspects currently in this frame

        # draw and snapshot logic
        for name, score, box in recognized:
            if box is not None:
                if name == "Unknown":
                    resized_frame = blur_region(resized_frame, box)
                else:
                    resized_frame = draw_box(resized_frame, box, name, score)
                    detected_now.add(name)
                    active_suspects[name] = now  # update last_seen_time

                    # capture crisp snapshot only once per recording cycle
                    if score > 50 and not recorder.snapshot_taken:
                        recorder.snapshot_frame = resized_frame.copy()
                        recorder.snapshot_taken = True
                        print(f"[{cam_id}] SNAPSHOT captured for {name}")

        # 1. remove LONG ABSENT suspects from active_suspects
        for s, last_seen in list(active_suspects.items()):
            if now - last_seen > COOLDOWN:
                active_suspects.pop(s)  # fully disappeared for > 10 sec

        # 2. identify suspects who are eligible to trigger recording
        new_suspects = set()

        for s in detected_now:
            # A. must NOT currently be active if we want a new event
            # (but since detected_now contains them, we check cooldown rules)
            
            # B. If suspect never recorded before → eligible
            if s not in recently_recorded:
                new_suspects.add(s)
                continue

            # C. They must have disappeared fully before → meaning they are NOT in active_suspects
            if s in active_suspects and s in detected_now:
                # still visible. not eligible
                continue

            # No active condition here since detected_now shows they are visible NOW.
            # So disappearance detection happened in step 1.

            # D. Check if they were absent long enough since last recording
            if now - recently_recorded[s] >= COOLDOWN:
                new_suspects.add(s)

        # 3. Start recording ONLY if:
        # - We have at least one newly eligible suspect
        # - No current recording in progress
        if len(new_suspects) > 0 and not recorder.recording:
            recorder.start(detected_now)
            threading.Thread(target=play_audio_once, daemon=True).start()

        # 4. During recording, push frames
        if recorder.recording:
            recorder.record_buffer.append(resized_frame.copy())

            if time.time() - recorder.record_start_time >= RECORD_DURATION:
                mongo_record = recorder.stop()
                print(f"[{cam_id}] Mongo ready:", mongo_record)

                # update cooldown for recorded suspects
                if mongo_record:
                    for s in mongo_record["suspects"]:
                        recently_recorded[s] = time.time()

        # display window
        cv2.imshow(f"Camera {cam_id}", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    embeddings, names = load_known_faces("known_faces")

    camera_sources = {
        "CAM1": 0,
        "CAM2": 1,
        # "CAM3": "http://192.168.29.155:8080/video",
        # "CAM3": "rtsp://192.168.1.10/stream",
    }

    threads = []
    for cam_id, source in camera_sources.items():
        t = threading.Thread(
            target=process_camera,
            args=(cam_id, source, embeddings, names),
            daemon=True
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
