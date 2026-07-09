import numpy as np

from serve_pipeline.landmarks import NUM_LANDMARKS, POSE_CONNECTIONS
from serve_pipeline.visualization import (
    draw_pose,
    save_contact_sheet,
    visibility_to_bgr,
)

from conftest import make_frame_pose


def test_visibility_color_endpoints():
    assert visibility_to_bgr(1.0) == (0, 255, 0)   # fully visible: green
    assert visibility_to_bgr(0.0) == (0, 0, 255)   # occluded: red
    b, g, r = visibility_to_bgr(0.5)
    assert b == 0 and g == r  # midpoint is balanced


def test_visibility_color_clips_out_of_range():
    assert visibility_to_bgr(1.7) == (0, 255, 0)
    assert visibility_to_bgr(-0.3) == (0, 0, 255)


def test_connections_reference_valid_landmarks():
    for start, end in POSE_CONNECTIONS:
        assert 0 <= start < NUM_LANDMARKS
        assert 0 <= end < NUM_LANDMARKS


def test_draw_pose_modifies_copy_not_input(blank_frame):
    fp = make_frame_pose(0, 0.0, detected=True)
    out = draw_pose(blank_frame, fp)
    assert blank_frame.sum() == 0          # input untouched
    assert out.sum() > 0                   # something was drawn
    assert out.shape == blank_frame.shape


def test_draw_pose_without_detection_shows_warning(blank_frame):
    fp = make_frame_pose(3, 0.1, detected=False)
    out = draw_pose(blank_frame, fp)
    # HUD and NO POSE text are drawn, red channel must be present.
    assert out.sum() > 0
    assert out[..., 2].sum() > 0


def test_draw_pose_handles_out_of_frame_coordinates(blank_frame):
    fp = make_frame_pose(0, 0.0, detected=True)
    fp.landmarks[0].x = -0.5
    fp.landmarks[0].y = 2.0
    fp.landmarks[1].x = 1.5
    out = draw_pose(blank_frame, fp)  # must not raise
    assert out.shape == blank_frame.shape


def test_contact_sheet(tmp_path, blank_frame):
    images = [blank_frame.copy() for _ in range(6)]
    path = str(tmp_path / "sheet.png")
    save_contact_sheet(path, images, columns=4, thumb_width=100)
    import cv2
    sheet = cv2.imread(path)
    assert sheet is not None
    assert sheet.shape[1] == 4 * 100          # 4 columns
    assert sheet.shape[0] == 2 * 75           # 2 rows, 4:3 aspect kept
