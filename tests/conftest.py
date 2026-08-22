import random

import numpy as np
import pytest

from serve_pipeline.landmarks import NUM_LANDMARKS
from serve_pipeline.pose_extraction import FramePose, LandmarkObservation


def make_frame_pose(frame_index=0, time_s=0.0, detected=True, seed=None):
    """Synthetic FramePose with plausible value ranges for tests."""
    if not detected:
        return FramePose(frame_index=frame_index, time_s=time_s,
                         detected=False, landmarks=[])
    rng = random.Random(seed if seed is not None else frame_index)
    landmarks = [
        LandmarkObservation(
            landmark_id=i,
            x=rng.uniform(0.2, 0.8), y=rng.uniform(0.1, 0.95),
            visibility=rng.uniform(0.0, 1.0),
        )
        for i in range(NUM_LANDMARKS)
    ]
    return FramePose(frame_index=frame_index, time_s=time_s,
                     detected=True, landmarks=landmarks)


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)
