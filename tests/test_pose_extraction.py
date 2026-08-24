import dataclasses
import os

import numpy as np
import pytest

from serve_pipeline.pose_extraction import LandmarkObservation, PoseExtractor
from serve_pipeline.stage1_extract import DEFAULT_MODEL

pytestmark = pytest.mark.skipif(
    not os.path.isfile(DEFAULT_MODEL),
    reason="pose model file not downloaded",
)


def test_observation_carries_only_image_plane_fields():
    """The 2D operating point: no depth, world, or presence channels."""
    field_names = [f.name for f in dataclasses.fields(LandmarkObservation)]
    assert field_names == ["landmark_id", "x", "y", "visibility"]


def test_no_person_returns_undetected():
    noise = np.random.default_rng(0).integers(
        0, 255, size=(480, 640, 3), dtype=np.uint8)
    with PoseExtractor(DEFAULT_MODEL) as ex:
        fp = ex.process(0, 0.0, noise)
    assert fp.detected is False
    assert fp.landmarks == []


def test_timestamps_strictly_increasing_at_high_fps():
    """Even at unrealistic fps the internal ms timestamps must not collide."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    with PoseExtractor(DEFAULT_MODEL) as ex:
        for i in range(5):
            ex.process(i, i / 10000.0, frame)  # 10000 fps -> sub-ms steps
    # detect_for_video raises on non-monotonic timestamps, so reaching this
    # point proves the guard works.
