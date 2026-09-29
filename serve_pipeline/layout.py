import os

DEFAULT_RESULTS_ROOT = "results"

STAGE1 = "stage1"
STAGE2 = "stage2"

META_JSON = "meta.json"


def clip_from_video(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def clip_from_stage_file(path: str) -> str:
    """Clip id from any file inside <root>/<clip>/<stage>/file."""
    stage_dir = os.path.dirname(os.path.abspath(path))
    clip_dir = os.path.dirname(stage_dir)
    return os.path.basename(clip_dir)


def stage_dir(results_root: str, clip: str, stage: str) -> str:
    return os.path.join(results_root, clip, stage)


def sibling_stage_dir(stage_file: str, stage: str) -> str:
    clip_dir = os.path.dirname(os.path.dirname(os.path.abspath(stage_file)))
    return os.path.join(clip_dir, stage)
