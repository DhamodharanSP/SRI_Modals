# main_skeleton_pipeline_example.py

import cv2
import numpy as np
from pose.pose_detector import PoseDetector   # your Stage 1 file
from sequence.skeleton_sequence_builder import SkeletonSequenceBuilder, SkeletonDetection

def process_camera_streams(camera_sources):
    pose_detector = PoseDetector(model_path="yolov8n-pose.pt")
    seq_builder = SkeletonSequenceBuilder(
        max_seq_len=30,
        iou_threshold=0.3,
        max_missing=10,
        min_confidence=0.25,
    )

    caps = {
        cam_id: cv2.VideoCapture(src)
        for cam_id, src in camera_sources.items()
    }

    frame_idx = {cam_id: 0 for cam_id in camera_sources.keys()}

    while True:
        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                continue

            # ---- Stage 1: pose detection ----
            pose_results = pose_detector.detect_poses(frame)  
            # Make sure detect() returns something we can convert to SkeletonDetection
            print(f"[{cam_id}] POSE COUNT:", len(pose_results))
            # Visual Debug: draw skeletons
            vis_frame = PoseDetector.draw_poses(frame, pose_results)
            cv2.imshow(f"{cam_id} Pose", vis_frame)

            detections = []
            for pose in pose_results:

                # Convert keypoints (17,2) + keypoint_scores (17,) → (17,3)
                kps_xy = np.array(pose["keypoints"])             # shape (17,2)
                kps_conf = np.array(pose["keypoint_scores"])     # shape (17,)
                kps_full = np.hstack([kps_xy, kps_conf[:, None]])  # shape (17,3)

                detections.append(
                    SkeletonDetection(
                        bbox=pose["bbox"],
                        keypoints=kps_full,    # (17,3) REQUIRED FOR STAGE 2
                        score=pose["score"]
                    )
                )


            # ---- Stage 2: sequence builder ----
            ready_sequences, active_tracks = seq_builder.update(
                camera_id=cam_id,
                detections=detections,
                frame_idx=frame_idx[cam_id],
            )

            print(f"[{cam_id}] Active Tracks:", active_tracks)
            
            # Debug: print ready sequences
            # for seq_info in ready_sequences:
            #     print(f"[{cam_id}] Sequence READY for Track {seq_info['track_id']}")
                
            # Use ready_sequences as input to Stage 3 (GNN)
            for seq_info in ready_sequences:
                sequence = seq_info["sequence"]  # (T, num_joints, 2)
                track_id = seq_info["track_id"]
                # TODO: pass to GNN model: gnn_model.predict(sequence)

            frame_idx[cam_id] += 1

        # Add break key if needed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_sources = {
        "CAM1": 1,      # your webcam
        # "CAM2": "rtsp://...",  # example
    }
    process_camera_streams(camera_sources)
