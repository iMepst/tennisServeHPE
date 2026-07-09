"""Modular tennis-serve analysis pipeline.

Stage layout (strictly separated, each stage persists its output):

    Stage 1  ingestion + pose extraction   -> landmarks CSV, meta JSON, overlay MP4
    Stage 2  processing (planned)          -> reads Stage 1 CSV only
    Stage 3  evaluation (planned)          -> reads Stage 2 output only

Modules:
    ingestion        video decoding, frame iteration, video metadata
    pose_extraction  BlazePose (MediaPipe Tasks API) wrapper
    persistence      CSV / JSON writers and readers for the raw time series
    visualization    diagnostic overlay rendering
    landmarks        BlazePose topology constants (names, connections)
"""

__version__ = "0.1.0"
