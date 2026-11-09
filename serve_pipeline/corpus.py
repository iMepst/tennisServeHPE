import argparse
import csv
import logging
import os

from .run import process_clip

logger = logging.getLogger(__name__)


def run_corpus(metadata_csv: str, outdir: str = "results",
               reuse: bool = True) -> int:
    data_dir = os.path.dirname(metadata_csv)
    with open(metadata_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["classification"] == "accepted"]
    for i, row in enumerate(rows, 1):
        clip = row["video_id"]
        logger.info("[%d/%d] %s", i, len(rows), clip)
        process_clip(
            video_path=os.path.join(data_dir, f"{clip}.mp4"),
            serving_arm=row["serving_arm"],
            front_leg=row["front_leg"],
            camera_plane=("frontal"
                          if row["camera_plane"] in ("frontal", "rear")
                          else "sagittal"),
            view_direction=row["view_direction"],
            fps=float(row["fps"]),
            frame_width=int(row["width"]),
            frame_height=int(row["height"]),
            outdir=outdir,
            reuse=reuse,
        )
    return len(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Run all accepted clips of the corpus end to end.")
    parser.add_argument("--metadata", default=os.path.join("data",
                                                           "videos.csv"),
                        help="corpus metadata (default: data/videos.csv)")
    parser.add_argument("--outdir", default="results",
                        help="results root (default: results)")
    parser.add_argument("--no-reuse", dest="reuse", action="store_false",
                        help="recompute extraction/gating/filtering even if "
                             "outputs exist")
    args = parser.parse_args()
    n = run_corpus(args.metadata, outdir=args.outdir, reuse=args.reuse)
    logger.info("Processed %d clip(s) into %s", n, args.outdir)


if __name__ == "__main__":
    main()
