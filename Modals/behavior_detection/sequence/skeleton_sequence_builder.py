# skeleton_sequence_builder.py

from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import itertools


@dataclass
class SkeletonDetection:
    """
    Generic container for a single pose detection in one frame.

    Adapt this to match your PoseDetector output.

    Expected:
        bbox: (x1, y1, x2, y2)
        keypoints: np.ndarray of shape (num_joints, 3) -> (x, y, conf)
        score: overall detection confidence
    """
    bbox: Tuple[float, float, float, float]
    keypoints: np.ndarray  # shape: (num_joints, 3)
    score: float


@dataclass
class Tracklet:
    """
    Holds temporal skeleton data for one person (one track) in one camera.
    """
    track_id: int
    camera_id: str
    max_len: int
    seq: deque = field(default_factory=deque)  # deque of (num_joints, 2) normalized
    last_frame_idx: int = -1
    missing_frames: int = 0

    def add_skeleton(self, skel_2d: np.ndarray, frame_idx: int):
        """
        Append new skeleton to this tracklet.
        skel_2d: (num_joints, 2) normalized coords
        """
        self.seq.append(skel_2d)
        if len(self.seq) > self.max_len:
            self.seq.popleft()
        self.last_frame_idx = frame_idx
        self.missing_frames = 0

    def mark_missing(self):
        self.missing_frames += 1

    def is_ready(self) -> bool:
        """
        Whether this track has enough frames for a full sequence.
        """
        return len(self.seq) == self.max_len

    def get_sequence_array(self) -> np.ndarray:
        """
        Returns sequence as np.ndarray of shape (T, num_joints, 2)
        """
        return np.stack(list(self.seq), axis=0)


class SkeletonSequenceBuilder:
    """
    Stage 2:
      - Assigns temporal track IDs to detections (per camera)
      - Normalizes skeletons
      - Builds fixed-length sequences for GNN

    Call update(camera_id, detections, frame_idx) every frame.

    Args:
        max_seq_len: frames per sequence window (e.g., 30)
        iou_threshold: minimum IOU to associate detection to existing track
        max_missing: how many frames a track can be missing before it is dropped
        min_confidence: minimum detection score to consider
    """

    def __init__(
        self,
        max_seq_len: int = 30,
        iou_threshold: float = 0.1,
        max_missing: int = 25,
        min_confidence: float = 0.2,
    ):
        self.max_seq_len = max_seq_len
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.min_confidence = min_confidence

        # Tracks indexed by (camera_id, track_id)
        self.tracks: Dict[Tuple[str, int], Tracklet] = {}

        # For generating new track IDs per camera
        self._track_id_counters: Dict[str, itertools.count] = {}

    # ------------ Public API ------------ #

    def update(
        self,
        camera_id: str,
        detections: List[SkeletonDetection],
        frame_idx: int,
    ):
        """
        Main function for Stage 2.

        Args:
            camera_id: e.g., "CAM1", "CAM2"
            detections: list of SkeletonDetection for this frame
            frame_idx: increasing index of frames for this camera

        Returns:
            ready_sequences: list of dicts:
                {
                    "camera_id": str,
                    "track_id": int,
                    "sequence": np.ndarray (T, num_joints, 2)
                }
            active_tracks: list of simple debug info (optional)
        """
        # Filter low-confidence detections
        detections = [
            det for det in detections if det.score >= self.min_confidence
        ]

        # Split tracks by camera
        cam_tracks = {
            (cam, tid): v
            for (cam, tid), v in self.tracks.items()
            if cam == camera_id
        }


        # Step 1: associate detections to tracks
        matches, unmatched_tracks, unmatched_dets = self._associate(
            camera_id, cam_tracks, detections
        )

        # Step 2: update matched tracks with new skeletons
        for track_key, det_idx in matches.items():
            track = self.tracks[track_key]
            det = detections[det_idx]
            skel_norm = self._normalize_keypoints(det.keypoints, det.bbox)
            track.add_skeleton(skel_norm, frame_idx)

        # Step 3: mark unmatched tracks as missing
        for track_key in unmatched_tracks:
            track = self.tracks[track_key]
            track.mark_missing()

        # Step 4: create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            skel_norm = self._normalize_keypoints(det.keypoints, det.bbox)
            new_track = self._create_track(camera_id, skel_norm, frame_idx)
            new_track.last_bbox = det.bbox   
            self.tracks[(camera_id, new_track.track_id)] = new_track

        # Step 5: collect ready sequences + clean dead tracks
        ready_sequences = []
        dead_keys = []

        for key, track in self.tracks.items():
            if track.camera_id != camera_id:
                continue

            if track.is_ready():
                ready_sequences.append(
                    {
                        "camera_id": track.camera_id,
                        "track_id": track.track_id,
                        "sequence": track.get_sequence_array(),
                    }
                )

            if track.missing_frames > self.max_missing:
                dead_keys.append(key)

        for key in dead_keys:
            del self.tracks[key]

        # Optional: simple view of active tracks
        active_tracks_info = [
            {
                "camera_id": t.camera_id,
                "track_id": t.track_id,
                "len": len(t.seq),
                "missing": t.missing_frames,
                "last_frame_idx": t.last_frame_idx,
            }
            for t in self.tracks.values()
            if t.camera_id == camera_id
        ]

        return ready_sequences, active_tracks_info

    # ------------ Internal helpers ------------ #

    def _get_track_id_counter(self, camera_id: str):
        if camera_id not in self._track_id_counters:
            self._track_id_counters[camera_id] = itertools.count(start=1)
        return self._track_id_counters[camera_id]

    def _create_track(self, camera_id: str, skel_2d: np.ndarray, frame_idx: int) -> Tracklet:
        track_id = next(self._get_track_id_counter(camera_id))
        track = Tracklet(
            track_id=track_id,
            camera_id=camera_id,
            max_len=self.max_seq_len,
        )
        track.add_skeleton(skel_2d, frame_idx)
        track.last_bbox = None   # <-- REQUIRED for clean association
        return track


    def _associate(
        self,
        camera_id: str,
        tracks: Dict[Tuple[str, int], Tracklet],
        detections: List[SkeletonDetection],
    ):
        """
        Very simple greedy association based on IoU between bboxes.

        Args:
            camera_id: current camera
            tracks: dict of (cam, id) -> Tracklet for this camera
            detections: list of SkeletonDetection

        Returns:
            matches: dict[(camera_id, track_id)] = det_idx
            unmatched_tracks: list of track_keys
            unmatched_dets: list of det indices
        """
        track_keys = list(tracks.keys())
        num_tracks = len(track_keys)
        num_dets = len(detections)
        if num_tracks == 0 or num_dets == 0:
            return {}, track_keys, list(range(num_dets))

        # Build IOU matrix: [num_tracks, num_dets]
        iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        det_bboxes = [det.bbox for det in detections]

        for i, track_key in enumerate(track_keys):
            # last bbox not stored, so we approximate from latest skeleton
            # -> Here we'll approximate bbox from the normalized skeleton later if needed.
            # Simpler approach: re-use detection bbox stored in external code.
            # For now, we approximate using the last sequence frame (unnormalized is not kept),
            # so better: compute IoU directly from detections each frame.
            # Instead, we will not use previous bbox; we will just not compute IoU for old tracks
            # that have no recent update. To keep it simple, use a constant small IOU when matching.
            # However, that is too hacky.
            # Better: store last bbox explicitly. Let's do that.
            pass

        # ---- Better design: keep last bbox in Tracklet ----
        # To avoid confusion, we re-implement association in a simpler way:

        return self._associate_with_last_bbox(camera_id, tracks, detections)

    def _associate_with_last_bbox(
        self,
        camera_id: str,
        tracks: Dict[Tuple[str, int], Tracklet],
        detections: List[SkeletonDetection],
    ):
        """
        Association with last bbox stored per track.

        We'll store last_bbox inside a side dict because Tracklet currently
        doesn't hold it. If you want, you can extend Tracklet to store bbox.
        """
        # We'll build a local map of last bboxes per track by reading from an
        # attribute we attach dynamically. If it doesn't exist yet, we skip IOU.
        track_keys = list(tracks.keys())
        num_tracks = len(track_keys)
        num_dets = len(detections)

        if num_tracks == 0 or num_dets == 0:
            return {}, track_keys, list(range(num_dets))

        iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
        det_bboxes = [det.bbox for det in detections]

        for i, track_key in enumerate(track_keys):
            track = tracks[track_key]
            last_bbox: Optional[Tuple[float, float, float, float]] = getattr(
                track, "last_bbox", None
            )

            if last_bbox is None:
                continue

            for j, det_bbox in enumerate(det_bboxes):
                iou_matrix[i, j] = self._iou(last_bbox, det_bbox)

        # Greedy matching
        matches = {}
        unmatched_tracks = set(track_keys)
        unmatched_dets = set(range(num_dets))

        # process pairs sorted by IOU desc
        idxs = np.dstack(np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape))[0]

        for i, j in idxs:
            if iou_matrix[i, j] < self.iou_threshold:
                break
            track_key = track_keys[i]
            if track_key in unmatched_tracks and j in unmatched_dets:
                matches[track_key] = j
                unmatched_tracks.remove(track_key)
                unmatched_dets.remove(j)

        # Update last_bbox for matched tracks
        for track_key, det_idx in matches.items():
            track = tracks[track_key]
            track.last_bbox = det_bboxes[det_idx]

        # For new tracks (unmatched detections), we'll set last_bbox when we create them
        # in _create_track_with_bbox. To keep code consistent, we modify _create_track.

        return matches, list(unmatched_tracks), list(unmatched_dets)

    @staticmethod
    def _iou(b1, b2) -> float:
        x1, y1, x2, y2 = b1
        x1b, y1b, x2b, y2b = b2

        inter_x1 = max(x1, x1b)
        inter_y1 = max(y1, y1b)
        inter_x2 = min(x2, x2b)
        inter_y2 = min(y2, y2b)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area2 = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)

        union = area1 + area2 - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _normalize_keypoints(
        self,
        keypoints: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """
        Convert absolute image coords to normalized person-centric coords.

        Strategy:
            - If keypoints shape is (num_joints, 3): (x, y, conf)
            - Use bbox center as origin
            - Scale by bbox height (so size is roughly between -1 and 1)

        This makes GNN focus on relative motion rather than camera zoom.

        Returns:
            skel_norm: (num_joints, 2)
        """
        assert keypoints.ndim == 2 and keypoints.shape[1] >= 2, \
            f"Expected keypoints (num_joints, >=2), got {keypoints.shape}"

        xs = keypoints[:, 0]
        ys = keypoints[:, 1]

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        h = max(1e-3, y2 - y1)

        # Translate to center, then divide by height
        xs_norm = (xs - cx) / h
        ys_norm = (ys - cy) / h

        skel_norm = np.stack([xs_norm, ys_norm], axis=-1)
        return skel_norm
