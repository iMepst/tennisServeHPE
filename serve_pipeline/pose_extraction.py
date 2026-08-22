from dataclasses import dataclass
from typing import Any, Dict, List

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from .ingestion import BgrImage
from .landmarks import NUM_LANDMARKS


@dataclass
class LandmarkObservation:
    landmark_id: int
    x: float           # normalized [~0..1] by image width
    y: float           # normalized [~0..1] by image height
    z: float           # normalized relative depth (image landmark z)
    visibility: float  # [0..1], low = likely occluded
    world_x: float     # meters, origin at hip center
    world_y: float
    world_z: float


@dataclass
class FramePose:
    frame_index: int
    time_s: float
    detected: bool
    landmarks: List[LandmarkObservation]  # empty when detected is False


class PoseExtractor:
    """Runs PoseLandmarker over successive frames of one video."""

    def __init__(self, model_path: str,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5) -> None:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms: int = -1
        self.config: Dict[str, Any] = {
            "model_path": model_path,
            "running_mode": "VIDEO",
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
            "min_presence_confidence": min_presence_confidence,
        }

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def process(self, frame_index: int, time_s: float,
                image_bgr: BgrImage) -> FramePose:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # PoseLandmarker requires strictly increasing integer milliseconds.
        timestamp_ms = int(round(time_s * 1000.0))
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return FramePose(frame_index=frame_index, time_s=time_s,
                             detected=False, landmarks=[])

        image_lms = result.pose_landmarks[0]
        world_lms = result.pose_world_landmarks[0]
        if len(image_lms) != NUM_LANDMARKS:
            raise RuntimeError(
                f"Expected {NUM_LANDMARKS} landmarks, got {len(image_lms)}"
            )

        observations = [
            LandmarkObservation(
                landmark_id=i,
                x=lm.x, y=lm.y, z=lm.z,
                visibility=lm.visibility,
                world_x=wlm.x, world_y=wlm.y, world_z=wlm.z,
            )
            for i, (lm, wlm) in enumerate(zip(image_lms, world_lms))
        ]
        return FramePose(frame_index=frame_index, time_s=time_s,
                         detected=True, landmarks=observations)

    def close(self) -> None:
        self._landmarker.close()
