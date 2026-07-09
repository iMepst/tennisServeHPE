import cv2
import numpy as np
import pytest

from serve_pipeline.ingestion import VideoReader

FPS = 30.0
WIDTH, HEIGHT, N_FRAMES = 320, 240, 12


@pytest.fixture
def synthetic_video(tmp_path):
    """Video whose frame index is encoded in the pixel intensity."""
    path = str(tmp_path / "synthetic.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (WIDTH, HEIGHT))
    for i in range(N_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_metadata(synthetic_video):
    with VideoReader(synthetic_video) as reader:
        assert reader.metadata.fps == pytest.approx(FPS)
        assert reader.metadata.width == WIDTH
        assert reader.metadata.height == HEIGHT
        assert reader.metadata.frame_count_reported == N_FRAMES


def test_iterates_all_frames_in_order(synthetic_video):
    with VideoReader(synthetic_video) as reader:
        frames = list(reader)
    assert [f.index for f in frames] == list(range(N_FRAMES))
    for f in frames:
        assert f.time_s == pytest.approx(f.index / FPS)
        assert f.image_bgr.shape == (HEIGHT, WIDTH, 3)
        # Codec is lossy; intensity must still be close to the encoded value.
        assert abs(float(f.image_bgr.mean()) - f.index * 20) < 5


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        VideoReader("does_not_exist.mp4")
