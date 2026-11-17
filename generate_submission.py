#!/usr/bin/env python3
"""Regenerate The Unpundits' complete possession submission from the provided bundle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parent


def run(command, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root of the organizers' extracted data bundle (contains tactical/)",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "deliverables" / "The_Unpundits.csv"),
        help="Destination submission CSV",
    )
    parser.add_argument(
        "--video",
        help="Optional path to the tactical MP4 when it is stored outside tactical/",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep the extracted WAV and audio-candidate CSVs under work/audio/",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if args.video:
        video = Path(args.video).expanduser().resolve()
    else:
        exact_video = data_dir / "tactical" / "BAR-ATH_tactical.mp4"
        candidates = sorted((data_dir / "tactical").glob("BAR-ATH_tactical*.mp4"))
        video = exact_video if exact_video.is_file() else (candidates[0] if candidates else exact_video)
    if not video.is_file():
        parser.error(
            "tactical video not found; place BAR-ATH_tactical*.mp4 under tactical/ "
            "or provide --video /path/to/video.mp4"
        )
    for suffix in ("_atd.xml", "_atd.txt", "_smart_tagging.xml"):
        required = data_dir / "tactical" / f"BAR-ATH_tactical{suffix}"
        if not required.is_file():
            parser.error(f"required tracking file not found: {required}")

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MTAH_DATA_DIR"] = str(data_dir)

    if args.keep_intermediates:
        temp_root = ROOT / "work" / "audio"
        temp_root.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="unpundits-")
        temp_root = Path(cleanup.name)

    wav = temp_root / "tactical_16k_mono.wav"
    strong = temp_root / "whistles_strong.csv"
    weak = temp_root / "whistles_weak.csv"
    intervals = temp_root / "dead_ball_intervals.csv"
    events = temp_root / "weak_events.csv"

    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        run([
            ffmpeg, "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", wav,
        ])
        run([
            sys.executable, ROOT / "work" / "whistle_audio_experiment.py", wav,
            "--out", strong,
        ])
        run([
            sys.executable, ROOT / "work" / "whistle_audio_experiment.py", wav,
            "--out", weak, "--ratio-threshold", "0.035",
            "--tonality-threshold", "0.12",
        ])
        run([
            sys.executable, ROOT / "work" / "dead_ball_multievidence_experiment.py",
            "--data-dir", data_dir,
            "--strong-whistles", strong,
            "--weak-whistles", weak,
            "--out", out,
            "--intervals", intervals,
            "--events", events,
        ], env=env)
        run([
            sys.executable, ROOT / "evaluation" / "validate_submission.py", out,
        ], env=env)
        print(f"\nSubmission ready: {out}")
    finally:
        if cleanup is not None:
            cleanup.cleanup()


if __name__ == "__main__":
    main()

