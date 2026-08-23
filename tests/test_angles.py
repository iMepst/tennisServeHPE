from serve_pipeline.angles import pixel_point
from serve_pipeline.config import ClipParams


def _params(width: int = 1920, height: int = 1080) -> ClipParams:
    return ClipParams(serving_arm="right", front_leg="left",
                      camera_plane="frontal", view_direction="front",
                      fps=25.0, frame_width=width, frame_height=height)


def test_pixel_point_rescales_by_frame_size() -> None:
    assert pixel_point(0.5, 0.5, _params()) == (960.0, 540.0)
    assert pixel_point(0.0, 1.0, _params()) == (0.0, 1080.0)
