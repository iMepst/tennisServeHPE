"""Stage 1d: diagnostic overlay rendering.

Draws the extracted skeleton back onto the source frames so tracking quality
can be judged visually. Landmark dots are colour-coded by visibility score
(green = visible, red = likely occluded); a HUD line shows frame index,
time and detection state. Frames where the racket arm turns red, or where
"NO POSE" appears, are the frames to distrust in later stages.

Pure image-in / image-out functions plus a thin video-writer wrapper, so the
drawing logic is unit-testable without any video file.
"""

from typing import List

import cv2
import numpy as np

from .landmarks import POSE_CONNECTIONS
from .pose_extraction import FramePose

_BONE_COLOR = (200, 200, 200)
_HUD_COLOR = (255, 255, 255)
_NO_POSE_COLOR = (0, 0, 255)


def visibility_to_bgr(visibility: float):
    """Map visibility in [0, 1] to BGR: 1 -> green, 0 -> red."""
    v = float(np.clip(visibility, 0.0, 1.0))
    return (0, int(round(255 * v)), int(round(255 * (1.0 - v))))


def draw_pose(image_bgr, frame_pose: FramePose):
    """Draws skeleton + HUD onto a copy of the frame and returns it."""
    out = image_bgr.copy()
    h, w = out.shape[:2]

    if frame_pose.detected:
        pts = [(int(round(o.x * w)), int(round(o.y * h)))
               for o in frame_pose.landmarks]
        for start, end in POSE_CONNECTIONS:
            cv2.line(out, pts[start], pts[end], _BONE_COLOR, 2, cv2.LINE_AA)
        for obs, pt in zip(frame_pose.landmarks, pts):
            cv2.circle(out, pt, 4, visibility_to_bgr(obs.visibility), -1,
                       cv2.LINE_AA)

    hud = f"frame {frame_pose.frame_index}  t={frame_pose.time_s:.3f}s"
    cv2.putText(out, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                _HUD_COLOR, 2, cv2.LINE_AA)
    if not frame_pose.detected:
        cv2.putText(out, "NO POSE", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    _NO_POSE_COLOR, 2, cv2.LINE_AA)
    return out


class OverlayVideoWriter:
    """Writes overlay frames to an mp4 with the source video's fps/size."""

    def __init__(self, path, fps, width, height):
        self.path = path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise IOError(f"Could not open video writer for {path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def write(self, image_bgr):
        self._writer.write(image_bgr)

    def close(self):
        self._writer.release()


def save_contact_sheet(path, images: List[np.ndarray], columns=4,
                       thumb_width=320):
    """Tiles overlay frames into one PNG for a quick visual sanity check."""
    if not images:
        raise ValueError("no images given")
    thumbs = []
    for img in images:
        scale = thumb_width / img.shape[1]
        thumbs.append(cv2.resize(img, (thumb_width,
                                       int(round(img.shape[0] * scale)))))
    th, tw = thumbs[0].shape[:2]
    rows = (len(thumbs) + columns - 1) // columns
    sheet = np.zeros((rows * th, columns * tw, 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, columns)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = thumb
    cv2.imwrite(path, sheet)
    return path
