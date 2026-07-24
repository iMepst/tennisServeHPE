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
import os
import subprocess
from typing import Any, Dict, List, Optional

from .gating import GatedFrame, GatedSample
from .landmarks import LANDMARK_NAMES, NUM_LANDMARKS
from .pose_extraction import FramePose, LandmarkObservation

CSV_HEADER = [
    "frame", "time_s", "landmark_id", "landmark_name",
    "x", "y", "z", "visibility", "presence",
    "world_x", "world_y", "world_z",
]

# Gated series (Stage 2a): raw schema plus the gating decision.
GATED_CSV_HEADER = CSV_HEADER + ["valid", "mask_reason"]

_VALUE_FIELDS = ["x", "y", "z", "visibility", "presence",
                 "world_x", "world_y", "world_z"]


class LandmarkCsvWriter:
    """Streaming writer: call write_frame() once per video frame."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_HEADER)

    def __enter__(self) -> "LandmarkCsvWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def write_frame(self, frame_pose: FramePose) -> None:
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
                    base + [lm_id, LANDMARK_NAMES[lm_id]]
                    + [""] * len(_VALUE_FIELDS)
                )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def read_landmarks_csv(path: str) -> List[FramePose]:
    """Reads a landmarks CSV back into a list of FramePose objects.

    Empty value fields (undetected frames) come back as detected=False with
    an empty landmark list, mirroring exactly what the extractor produced.
    """
    frames: Dict[int, FramePose] = {}
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


def _fmt(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.6f}"


def write_gated_csv(path: str, gated: List[GatedFrame]) -> None:
    """Persist the Stage 2a gated series (raw schema + valid/mask_reason)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(GATED_CSV_HEADER)
        for gframe in gated:
            base = [gframe.frame_index, f"{gframe.time_s:.6f}"]
            for s in gframe.samples:
                writer.writerow(
                    base + [s.landmark_id, LANDMARK_NAMES[s.landmark_id]]
                    + [_fmt(getattr(s, fld)) for fld in _VALUE_FIELDS]
                    + [1 if s.valid else 0, s.mask_reason]
                )


def read_gated_csv(path: str) -> List[GatedFrame]:
    """Read a gated CSV back into GatedFrame objects (round-trips write)."""
    frames: Dict[int, GatedFrame] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != GATED_CSV_HEADER:
            raise ValueError(
                f"Unexpected gated CSV schema in {path}: {reader.fieldnames}"
            )
        for row in reader:
            idx = int(row["frame"])
            if idx not in frames:
                frames[idx] = GatedFrame(
                    frame_index=idx, time_s=float(row["time_s"]), samples=[])
            frames[idx].samples.append(GatedSample(
                landmark_id=int(row["landmark_id"]),
                valid=row["valid"] == "1",
                mask_reason=row["mask_reason"],
                **{fld: (None if row[fld] == "" else float(row[fld]))
                   for fld in _VALUE_FIELDS},
            ))
    result = [frames[i] for i in sorted(frames)]
    for gf in result:
        if len(gf.samples) != NUM_LANDMARKS:
            raise ValueError(
                f"Frame {gf.frame_index} has {len(gf.samples)} samples, "
                f"expected {NUM_LANDMARKS}"
            )
    return result


def write_metadata(path: str, meta: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def read_metadata(path: str) -> Dict[str, Any]:
    with open(path) as f:
        data: Dict[str, Any] = json.load(f)
    return data


def git_commit_hash() -> Optional[str]:
    """Return the producing commit hash for provenance in metadata.

    Appends ``-dirty`` when the working tree has uncommitted changes, so a
    run made from an unclean checkout is not silently attributed to a clean
    commit. Returns ``None`` outside a git repository or when git is
    unavailable, rather than raising: provenance is best-effort metadata and
    must never abort a run.
    """
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _git(args: List[str]) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=repo_dir,
            capture_output=True, text=True, check=True,
        )
        return completed.stdout.strip()

    try:
        commit = _git(["rev-parse", "HEAD"])
        dirty = _git(["status", "--porcelain"])
    except (OSError, subprocess.SubprocessError):
        return None
    if not commit:
        return None
    return f"{commit}-dirty" if dirty else commit


def summarize_extraction(
        frame_poses: List[FramePose]) -> Dict[str, Any]:
    """Aggregate sanity-check numbers for the run report / metadata JSON."""
    n = len(frame_poses)
    detected = [fp for fp in frame_poses if fp.detected]
    per_landmark_visibility: Dict[str, float] = {}
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
