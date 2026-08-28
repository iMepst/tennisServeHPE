import os

DEFAULT_RESULTS_ROOT = "results"

STAGE1 = "stage1"
STAGE2 = "stage2"

META_JSON = "meta.json"


def clip_from_video(video_path: str) -> str:
    """Clip id from a video path (serve_01.mp4 -> serve_01)."""
    return os.path.splitext(os.path.basename(video_path))[0]


def clip_from_stage_file(path: str) -> str:
    """Clip id from any file inside <root>/<clip>/<stage>/file."""
    stage_dir = os.path.dirname(os.path.abspath(path))
    clip_dir = os.path.dirname(stage_dir)
    return os.path.basename(clip_dir)


def stage_dir(results_root: str, clip: str, stage: str) -> str:
    """The directory a given stage writes into for a given clip."""
    return os.path.join(results_root, clip, stage)


def sibling_stage_dir(stage_file: str, stage: str) -> str:
    """A sibling stage folder next to a file, e.g. stage1 CSV -> stage2 dir."""
    clip_dir = os.path.dirname(os.path.dirname(os.path.abspath(stage_file)))
    return os.path.join(clip_dir, stage)
