import json
import os
from typing import List

from serve_pipeline.interpolation import ProcessedFrame, ProcessedSample
from serve_pipeline.landmarks import NAME_TO_ID, NUM_LANDMARKS
from serve_pipeline.persistence import write_filtered_csv, write_metadata
from serve_pipeline.run import process_clip

FPS = 25.0
N = 25


def _synthetic_frames() -> List[ProcessedFrame]:
    """A tiny reliable clip with a clear wrist minimum and pelvis maximum.

    Every landmark is reliable and distinctly placed (so the angles are
    non-degenerate); the racket wrist dips to a unique minimum at frame 18
    (ball impact) and the two hips rise to a unique maximum at frame 8
    (trophy, before impact).
    """
    wrist_id = NAME_TO_ID["right_wrist"]
    hip_ids = (NAME_TO_ID["left_hip"], NAME_TO_ID["right_hip"])

    wrist_y = [0.6] * N
    wrist_y[17], wrist_y[18], wrist_y[19] = 0.2, 0.1, 0.2  # min at 18
    hip_y = [0.5] * N
    hip_y[7], hip_y[8], hip_y[9] = 0.65, 0.70, 0.65        # max at 8

    frames: List[ProcessedFrame] = []
    for i in range(N):
        samples: List[ProcessedSample] = []
        for lm in range(NUM_LANDMARKS):
            if lm == wrist_id:
                y = wrist_y[i]
            elif lm in hip_ids:
                y = hip_y[i]
            else:
                y = 0.5
            samples.append(ProcessedSample(
                landmark_id=lm, valid=True, mask_reason="ok",
                interpolated=False, reliable=True, filtered=True,
                x=0.4 + 0.01 * lm, y=y, visibility=0.9))
        frames.append(ProcessedFrame(frame_index=i, time_s=i / FPS,
                                     samples=samples))
    return frames


def _setup_clip(results_root: str, clip: str) -> None:
    """Lay down the Stage 1/2 outputs so process_clip reuses them."""
    stage1 = os.path.join(results_root, clip, "stage1")
    stage2 = os.path.join(results_root, clip, "stage2")
    os.makedirs(stage1)
    os.makedirs(stage2)
    # Markers whose presence makes ensure_filtered skip Stages 1/2a.
    open(os.path.join(stage1, "landmarks.csv"), "w").close()
    open(os.path.join(stage2, "gated.csv"), "w").close()
    write_metadata(os.path.join(stage1, "meta.json"),
                   {"video": {"fps": FPS, "width": 1920, "height": 1080}})
    write_filtered_csv(os.path.join(stage2, "filtered.csv"),
                       _synthetic_frames())


def test_process_clip_writes_result_json(tmp_path) -> None:
    results_root = str(tmp_path)
    clip = "serve_test"
    _setup_clip(results_root, clip)

    out_path = process_clip(
        video_path=os.path.join(results_root, f"{clip}.mp4"),
        serving_arm="right", front_leg="left",
        camera_plane="frontal", view_direction="front",
        outdir=results_root, reuse=True)

    assert out_path == os.path.join(results_root, clip, "result.json")
    with open(out_path) as f:
        result = json.load(f)

    # Top-level record.
    assert set(result) >= {
        "clip", "pipeline_version", "created_utc", "clip_params",
        "key_events", "slow_motion", "angles", "indicators"}
    assert result["clip"] == clip

    # Key events located on the planted extrema.
    assert result["key_events"]["trophy_frame"] == 8
    assert result["key_events"]["impact_frame"] == 18

    # One indicator per rule; the frontal plane reads trunk, not the knee.
    by_criterion = {i["criterion"]: i for i in result["indicators"]}
    assert set(by_criterion) == {
        "trunk_inclination", "front_knee_flexion",
        "elbow_flexion", "shoulder_elevation"}
    assert by_criterion["front_knee_flexion"]["status"] == "unavailable"
    assert by_criterion["trunk_inclination"]["status"] in ("inside",
                                                           "outside")
