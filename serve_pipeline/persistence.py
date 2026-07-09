"""Stage 1c: persistence of the raw landmark time series.

Writes the raw extraction output row by row while the video is processed, so
a crash mid-video still leaves a valid, truncated CSV. Values are stored
exactly as the model produced them: no smoothing, no interpolation, no
coordinate conversion. Later stages read this file and make their own
processing decisions, which keeps error attribution per stage possible.

CSV schema (long format, one row per frame x landmark):

    frame, time_s, landmark_id, landmark_name,
    x, y, z, visibility, presence,
    world_x, world_y, world_z

Frames without a detection still get their 33 rows with empty value fields,
so frame indexing stays dense and gaps are explicit in the data.
"""

import csv
import json
import math

from .landmarks import LANDMARK_NAMES, NUM_LANDMARKS
from .pose_extraction import FramePose, LandmarkObservation

CSV_HEADER = [
    "frame", "time_s", "landmark_id", "landmark_name",
    "x", "y", "z", "visibility", "presence",
    "world_x", "world_y", "world_z",
]

_VALUE_FIELDS = ["x", "y", "z", "visibility", "presence",
                 "world_x", "world_y", "world_z"]


class LandmarkCsvWriter:
    """Streaming writer: call write_frame() once per video frame."""

    def __init__(self, path):
        self.path = path
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_HEADER)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def write_frame(self, frame_pose: FramePose):
        base = [frame_pose.frame_index, f"{frame_pose.time_s:.6f}"]
        if frame_pose.detected:
            for obs in frame_pose.landmarks:
                self._writer.writerow(
                    base
                    + [obs.landmark_id, LANDMARK_NAMES[obs.landmark_id]]
                    + [f"{getattr(obs, f):.6f}" for f in _VALUE_FIELDS]
                )
        else:
            for lm_id in range(NUM_LANDMARKS):
                self._writer.writerow(
                    base + [lm_id, LANDMARK_NAMES[lm_id]] + [""] * len(_VALUE_FIELDS)
                )
        self._file.flush()

    def close(self):
        self._file.close()


def read_landmarks_csv(path):
    """Reads a landmarks CSV back into a list of FramePose objects.

    Empty value fields (undetected frames) come back as detected=False with
    an empty landmark list, mirroring exactly what the extractor produced.
    """
    frames = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_HEADER:
            raise ValueError(
                f"Unexpected CSV schema in {path}: {reader.fieldnames}"
            )
        for row in reader:
            idx = int(row["frame"])
            if idx not in frames:
                frames[idx] = FramePose(
                    frame_index=idx,
                    time_s=float(row["time_s"]),
                    detected=row["x"] != "",
                    landmarks=[],
                )
            if row["x"] != "":
                frames[idx].landmarks.append(LandmarkObservation(
                    landmark_id=int(row["landmark_id"]),
                    **{f: float(row[f]) for f in _VALUE_FIELDS},
                ))

    result = [frames[i] for i in sorted(frames)]
    for fp in result:
        if fp.detected and len(fp.landmarks) != NUM_LANDMARKS:
            raise ValueError(
                f"Frame {fp.frame_index} has {len(fp.landmarks)} landmarks, "
                f"expected {NUM_LANDMARKS}"
            )
    return result


def write_metadata(path, meta: dict):
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def read_metadata(path):
    with open(path) as f:
        return json.load(f)


def summarize_extraction(frame_poses):
    """Aggregate sanity-check numbers for the run report / metadata JSON."""
    n = len(frame_poses)
    detected = [fp for fp in frame_poses if fp.detected]
    per_landmark_visibility = {}
    if detected:
        for lm_id in range(NUM_LANDMARKS):
            vals = [fp.landmarks[lm_id].visibility for fp in detected]
            per_landmark_visibility[LANDMARK_NAMES[lm_id]] = (
                sum(vals) / len(vals)
            )
    mean_vis = (
        sum(per_landmark_visibility.values()) / NUM_LANDMARKS
        if detected else math.nan
    )
    return {
        "frames_processed": n,
        "frames_with_pose": len(detected),
        "detection_rate": len(detected) / n if n else 0.0,
        "mean_visibility": mean_vis,
        "mean_visibility_per_landmark": per_landmark_visibility,
    }
