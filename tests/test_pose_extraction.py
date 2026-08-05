import os

import cv2
import numpy as np
import pytest

from serve_pipeline.landmarks import NUM_LANDMARKS
from serve_pipeline.pose_extraction import PoseExtractor
from serve_pipeline.stage1_extract import DEFAULT_MODEL

pytestmark = pytest.mark.skipif(
    not os.path.isfile(DEFAULT_MODEL),
    reason="pose model file not downloaded",
)


@pytest.fixture(scope="module")
def person_image():
    """A frame with a real person if available, else None."""
    path = os.path.join(os.path.dirname(DEFAULT_MODEL), "..",
                        "data", "sample_person.jpg")
    if os.path.isfile(path):
        return cv2.imread(path)
    return None


def test_no_person_returns_undetected():
    noise = np.random.default_rng(0).integers(
        0, 255, size=(480, 640, 3), dtype=np.uint8)
    with PoseExtractor(DEFAULT_MODEL) as ex:
        fp = ex.process(0, 0.0, noise)
    assert fp.detected is False
    assert fp.landmarks == []


def test_person_detected_with_valid_ranges(person_image):
    if person_image is None:
        pytest.skip("no sample person image in data/")
    with PoseExtractor(DEFAULT_MODEL) as ex:
        fp = ex.process(0, 0.0, person_image)
    assert fp.detected is True
    assert len(fp.landmarks) == NUM_LANDMARKS
    for obs in fp.landmarks:
        assert 0.0 <= obs.visibility <= 1.0
        assert 0.0 <= obs.presence <= 1.0
        assert -0.5 <= obs.x <= 1.5   # normalized, may exceed frame slightly
        assert -0.5 <= obs.y <= 1.5
        assert abs(obs.world_x) < 3.0  # meters from hip center
        assert abs(obs.world_y) < 3.0
        assert abs(obs.world_z) < 3.0


def test_timestamps_strictly_increasing_at_high_fps():
    """Even at unrealistic fps the internal ms timestamps must not collide."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    with PoseExtractor(DEFAULT_MODEL) as ex:
        for i in range(5):
            ex.process(i, i / 10000.0, frame)  # 10000 fps -> sub-ms steps
    # detect_for_video raises on non-monotonic timestamps, so reaching this
    # point proves the guard works.
