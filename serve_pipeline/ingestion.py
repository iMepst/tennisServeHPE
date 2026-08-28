from dataclasses import dataclass, asdict
import os
from typing import Any, Dict, Iterator

import cv2
import numpy as np

# OpenCV frame: H×W×3, BGR order, uint8.
BgrImage = np.ndarray


@dataclass
class VideoMetadata:
    path: str
    fps: float
    width: int
    height: int
    frame_count_reported: int  # from container header; may be 0 or wrong

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Frame:
    index: int
    time_s: float
    image_bgr: BgrImage


class VideoReader:
    """Iterates a video file frame by frame."""

    def __init__(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"OpenCV could not open video: {path}")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            raise ValueError(
                f"Video reports invalid FPS ({fps}); timestamps would be "
                f"meaningless. Re-encode the file with a valid frame rate."
            )
        self.metadata = VideoMetadata(
            path=os.path.abspath(path),
            fps=float(fps),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count_reported=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                return
            yield Frame(index=index,
                        time_s=index / self.metadata.fps,
                        image_bgr=image)
            index += 1

    def close(self) -> None:
        self._cap.release()
