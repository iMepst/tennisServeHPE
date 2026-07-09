"""Stage 1b: pose extraction with BlazePose via the MediaPipe Tasks API.

The legacy ``mp.solutions.pose`` API was removed from mediapipe >= 0.10.x,
so this wraps ``PoseLandmarker`` in VIDEO running mode instead. VIDEO mode
uses temporal tracking between frames, which matters for the fast phases of
the serve.

Two coordinate systems are extracted per landmark and both are persisted:

  image landmarks  x, y normalized to image width/height (z: relative depth,
                   roughly hip-scaled) -- used for the overlay and any
                   pixel-space checks.
  world landmarks  metric 3D coordinates in meters, origin at the hip
                   center -- the right input for joint-angle computation in
                   later stages because they are camera-projection free.

Each landmark carries a ``visibility`` score (is it likely visible, i.e. not
occluded) and a ``presence`` score (is it likely inside the frame). Low
visibility on the racket arm during the trophy position / contact phase is
exactly what the overlay video is meant to reveal.
"""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from .landmarks import NUM_LANDMARKS


@dataclass
class LandmarkObservation:
    landmark_id: int
    x: float           # normalized [~0..1] by image width
    y: float           # normalized [~0..1] by image height
    z: float           # normalized relative depth (image landmark z)
    visibility: float  # [0..1], low = likely occluded
    presence: float    # [0..1], low = likely outside the frame
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
    """Runs PoseLandmarker over successive frames of one video.

    One instance per video: VIDEO running mode requires strictly increasing
    timestamps, so instances must not be reused across videos.
    """

    def __init__(self, model_path, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, min_presence_confidence=0.5):
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
        self._last_timestamp_ms = -1
        self.config = {
            "model_path": model_path,
            "running_mode": "VIDEO",
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
            "min_presence_confidence": min_presence_confidence,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def process(self, frame_index: int, time_s: float, image_bgr) -> FramePose:
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
                visibility=lm.visibility, presence=lm.presence,
                world_x=wlm.x, world_y=wlm.y, world_z=wlm.z,
            )
            for i, (lm, wlm) in enumerate(zip(image_lms, world_lms))
        ]
        return FramePose(frame_index=frame_index, time_s=time_s,
                         detected=True, landmarks=observations)

    def close(self):
        self._landmarker.close()
