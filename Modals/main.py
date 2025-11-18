# import cv2
# import time
# import threading
# from playsound import playsound
# from FRT.FaceRecognition import load_known_faces, recognize_faces
# from utils.video_utils import blur_region, draw_box
# from utils.recording_utils import (
#     start_recording, stop_recording,
#     recording, record_buffer, record_start_time,
#     RESIZE_WIDTH, RESIZE_HEIGHT, RECORD_DURATION,
# )

# COOLDOWN = 10
# ALERT_SOUND = "sounds/alert.mp3"
# OUTPUT_DIR = "recordings"

# audio_playing = False

# active_suspects = {}
# recently_recorded = {}

# def play_audio_once():
#     global audio_playing
#     if audio_playing:
#         return
#     audio_playing = True
#     try:
#         playsound(ALERT_SOUND)
#     finally:
#         audio_playing = False

# def process_webcam(embeddings, names):
#     global active_suspects, recently_recorded

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Cannot open webcam")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         now = time.time()
#         resized_frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

#         recognized = recognize_faces(frame, embeddings, names)
#         detected_now = set()

#         # draw and collect suspects
#         for name, score, box in recognized:
#             if box is not None:
#                 if name == "Unknown":
#                     resized_frame = blur_region(resized_frame, box)
#                 else:
#                     resized_frame = draw_box(resized_frame, box, name, score)
#                     detected_now.add(name)
#                     active_suspects[name] = now

#                     # take crisp snapshot once when score is reliable
#                     if score > 60 and not snapshot_taken:
#                         snapshot_frame = resized_frame.copy()
#                         snapshot_taken = True
#                         print("[SNAPSHOT] Crisp suspect image captured.")

#         # remove inactive suspects
#         for s, last_seen in list(active_suspects.items()):
#             if now - last_seen > COOLDOWN:
#                 active_suspects.pop(s)

#         # find new suspects
#         new_suspects = set()
#         for s in detected_now:
#             if s not in recently_recorded:
#                 new_suspects.add(s)
#             else:
#                 if s not in active_suspects:
#                     if now - recently_recorded[s] >= COOLDOWN:
#                         new_suspects.add(s)

#         # start recording
#         if len(new_suspects) > 0 and not recording:
#             start_recording(detected_now)
#             threading.Thread(target=play_audio_once, daemon=True).start()

#         # during recording
#         if recording:
#             record_buffer.append(resized_frame.copy())
#             if time.time() - record_start_time >= RECORD_DURATION:
#                 mongo_record = stop_recording()
#                 print("Mongo ready:", mongo_record)

#                 # mark cooldown
#                 for s in new_suspects:
#                     recently_recorded[s] = time.time()

#         cv2.imshow("Webcam", resized_frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def main():
#     import os
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     embeddings, names = load_known_faces("known_faces")
#     process_webcam(embeddings, names)

# if __name__ == "__main__":
#     main()

import cv2
import time
import threading
from playsound import playsound
from FRT.FaceRecognition import load_known_faces, recognize_faces
from utils.video_utils import blur_region, draw_box
import utils.recording_utils as rec

COOLDOWN = 10
ALERT_SOUND = "sounds/alert.mp3"
OUTPUT_DIR = "recordings"

audio_playing = False

active_suspects = {}
recently_recorded = {}          # suspect -> last_record_time


def play_audio_once():
    global audio_playing
    if audio_playing:
        return
    audio_playing = True
    try:
        playsound(ALERT_SOUND)
    finally:
        audio_playing = False


def process_webcam(embeddings, names):
    global active_suspects, recently_recorded

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        resized_frame = cv2.resize(frame, (rec.RESIZE_WIDTH, rec.RESIZE_HEIGHT))

        recognized = recognize_faces(frame, embeddings, names)
        detected_now = set()

        # draw boxes and collect suspects
        for name, score, box in recognized:
            if box is not None:
                if name == "Unknown":
                    resized_frame = blur_region(resized_frame, box)
                else:
                    resized_frame = draw_box(resized_frame, box, name, score)
                    detected_now.add(name)
                    active_suspects[name] = now

                    # single crisp snapshot when confidence is high
                    if score > 50 and not rec.snapshot_taken:
                        rec.snapshot_frame = resized_frame.copy()
                        rec.snapshot_taken = True
                        print("[SNAPSHOT] Crisp suspect image captured.")

        # remove inactive suspects from tracking
        for s, last_seen in list(active_suspects.items()):
            if now - last_seen > COOLDOWN:
                active_suspects.pop(s)

        # decide which suspects are new with respect to cooldown
        new_suspects = set()
        for s in detected_now:
            if s not in recently_recorded:
                new_suspects.add(s)
            else:
                # time based cooldown from last recording
                if now - recently_recorded[s] >= COOLDOWN:
                    new_suspects.add(s)

        # start recording once for the current group
        if len(new_suspects) > 0 and not rec.recording:
            suspects_to_record = detected_now.copy()
            rec.start_recording(suspects_to_record)
            threading.Thread(target=play_audio_once, daemon=True).start()

        # during recording, push frames into buffer
        if rec.recording:
            rec.record_buffer.append(resized_frame.copy())
            if time.time() - rec.record_start_time >= rec.RECORD_DURATION:
                mongo_record = rec.stop_recording()
                print("Mongo ready:", mongo_record)

                # update cooldown for all suspects in that clip
                if mongo_record is not None:
                    for s in mongo_record["suspects"]:
                        recently_recorded[s] = time.time()

        cv2.imshow("Webcam", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    embeddings, names = load_known_faces("known_faces")
    process_webcam(embeddings, names)


if __name__ == "__main__":
    main()
