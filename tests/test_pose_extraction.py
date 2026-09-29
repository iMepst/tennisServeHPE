import dataclasses
import os

import numpy as np
import pytest

from serve_pipeline.pose_extraction import LandmarkObservation, PoseExtractor
from serve_pipeline.extract import DEFAULT_MODEL

pytestmark = pytest.mark.skipif(
    not os.path.isfile(DEFAULT_MODEL),
    reason="pose model file not downloaded",
)


def test_observation_carries_only_image_plane_fields():
    field_names = [f.name for f in dataclasses.fields(LandmarkObservation)]
    assert field_names == ["landmark_id", "x", "y", "visibility"]


def test_no_person_returns_undetected():
    noise = np.random.default_rng(0).integers(
        0, 255, size=(480, 640, 3), dtype=np.uint8)
    with PoseExtractor(DEFAULT_MODEL) as extractor:
        fp = extractor.process(0, 0.0, noise)
    assert fp.detected is False
    assert fp.landmarks == []


def test_timestamps_strictly_increasing_at_high_fps():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    with PoseExtractor(DEFAULT_MODEL) as extractor:
        for i in range(5):
            extractor.process(i, i / 10000.0, frame)
