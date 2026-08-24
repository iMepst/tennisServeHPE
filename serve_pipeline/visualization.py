from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .ingestion import BgrImage, VideoReader
from .landmarks import POSE_CONNECTIONS
from .pose_extraction import FramePose

_BgrColor = Tuple[int, int, int]

_BONE_COLOR: _BgrColor = (200, 200, 200)
_HUD_COLOR: _BgrColor = (255, 255, 255)
_NO_POSE_COLOR: _BgrColor = (0, 0, 255)


def visibility_to_bgr(visibility: float) -> _BgrColor:
    """Map visibility in [0, 1] to BGR: 1 -> green, 0 -> red."""
    v = float(np.clip(visibility, 0.0, 1.0))
    return (0, int(round(255 * v)), int(round(255 * (1.0 - v))))


def draw_pose(image_bgr: BgrImage, frame_pose: FramePose) -> BgrImage:
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

    def __init__(self, path: str, fps: float,
                 width: int, height: int) -> None:
        self.path = path
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise IOError(f"Could not open video writer for {path}")

    def __enter__(self) -> "OverlayVideoWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def write(self, image_bgr: BgrImage) -> None:
        self._writer.write(image_bgr)

    def close(self) -> None:
        self._writer.release()


def label_frame(image_bgr: BgrImage, frame_pose: Optional[FramePose],
                lines: Sequence[str],
                draw_skeleton: bool = True) -> BgrImage:
    """One key-frame panel: pose overlay plus a text label block.

    Returns a copy of the frame with the detected skeleton drawn (when a
    FramePose is given and draw_skeleton is set) and the label lines in a
    dark box below the frame HUD. Used to render the trophy and impact
    stills; the input frame is not modified.
    """
    if draw_skeleton and frame_pose is not None:
        out = draw_pose(image_bgr, frame_pose)
    else:
        out = image_bgr.copy()

    scale, thickness, line_h, pad, x0, y0 = 0.9, 2, 34, 10, 12, 72
    widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX,
                              scale, thickness)[0][0]
              for t in lines]
    box_w = (max(widths) if widths else 0) + 2 * pad
    box_h = line_h * len(lines) + 2 * pad
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        baseline = y0 + pad + line_h * i + int(round(line_h * 0.7))
        cv2.putText(out, text, (x0 + pad, baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, _HUD_COLOR, thickness,
                    cv2.LINE_AA)
    return out


def _hstack_common_height(images: List[BgrImage]) -> BgrImage:
    """Tile images left to right, scaled to their common (minimum) height."""
    h = min(im.shape[0] for im in images)
    resized = []
    for im in images:
        if im.shape[0] != h:
            w = int(round(im.shape[1] * h / im.shape[0]))
            im = cv2.resize(im, (w, h))
        resized.append(im)
    return np.hstack(resized)


def save_key_frame_stills(video_path: str, frame_poses: List[FramePose],
                          specs: Sequence[Tuple[int, Sequence[str]]],
                          out_path: str,
                          draw_skeleton: bool = True) -> Optional[str]:
    """Write one side-by-side PNG of the located key frames.

    ``specs`` is a list of ``(frame_index, label lines)``; the named
    frames are read once from the source video, each gets its pose overlay
    and label (label_frame), and the panels are tiled left to right in
    ``specs`` order. Returns out_path, or None when no spec resolves to a
    frame (e.g. no key event was locatable).
    """
    if not specs:
        return None
    poses = {fp.frame_index: fp for fp in frame_poses}
    wanted = {idx for idx, _ in specs}
    images: dict = {}
    with VideoReader(video_path) as reader:
        for frame in reader:
            if frame.index in wanted:
                images[frame.index] = frame.image_bgr
                if len(images) == len(wanted):
                    break
    panels = [label_frame(images[idx], poses.get(idx), lines, draw_skeleton)
              for idx, lines in specs if idx in images]
    if not panels:
        return None
    cv2.imwrite(out_path, _hstack_common_height(panels))
    return out_path


def save_contact_sheet(path: str, images: List[BgrImage], columns: int = 4,
                       thumb_width: int = 960) -> str:
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
