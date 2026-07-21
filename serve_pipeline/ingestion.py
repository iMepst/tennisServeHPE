"""Stage 1a: video ingestion.

Wraps OpenCV video decoding behind a small iterator interface so the rest of
the pipeline never touches cv2.VideoCapture directly. Yields BGR frames
together with their index and timestamp; timestamps are derived from the
container FPS, which is also what later stages must use to convert frame
indices back to time.
"""

from dataclasses import dataclass, asdict
import os
from typing import Any, Dict, Iterator

import cv2
import numpy as np

# H x W x 3, BGR channel order, dtype uint8 -- the array OpenCV decodes into.
# Defined here (the pipeline's image entry point) and reused by the modules
# that pass frames around, so signatures stay readable.
BgrImage = np.ndarray


@dataclass
class VideoMetadata:
    path: str
    fps: float
    width: int
    height: int
    frame_count_reported: int  # container header value, may be 0 or wrong

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Frame:
    index: int
    time_s: float
    image_bgr: BgrImage


class VideoReader:
    """Iterates a video file frame by frame.

    Usage:
        with VideoReader(path) as reader:
            print(reader.metadata)
            for frame in reader:
                ...
    """

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
