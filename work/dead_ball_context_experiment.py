#!/usr/bin/env python3
"""Dead-ball v3: accept weak whistles only when pitch geometry corroborates them.

This experiment extends the conservative v2 latch.  It does not read organizer labels and does
not contain match-specific timestamps.  A short/quiet whistle candidate is accepted only if the
tracked ball recently approached one pitch boundary and subsequently crosses that *same* boundary.
This recovers low-volume referee signals without treating every tonal stadium sound as a whistle.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from games import GAME, GAMES, require, source_frames  # noqa: E402
from starter.baseline import team_of_each_track, write_submission  # noqa: E402
from sequence_possession_experiment import (  # noqa: E402
    load_tracking_evidence, planar_motion, possession_memory, probable_contacts,
)
from dead_ball_experiment import (  # noqa: E402
    dead_ball_latch, load_player_positions, load_strong_whistles, pitch_exit_events,
    team_activity, write_intervals,
)


WEAK_MIN_DURATION_S = 0.12
WEAK_MIN_MEAN_RATIO = 0.035
WEAK_MIN_MAX_RATIO = 0.055
WEAK_MIN_TONALITY = 0.30
BOUNDARY_MARGIN_M = 2.0
BOUNDARY_LOOKBACK_S = 4.0
BOUNDARY_CONFIRM_S = 4.0
# A weak audio cue receives less trust than a normal whistle.  Require the recent boundary moment
# to look like a ball that settled/was trapped at the line, not an ordinary live restart or attack.
BOUNDARY_MAX_APPROACH_SPEED_MPS = 1.0


def boundary_side(point, pitch):
    """Return the nearest pitch boundary and signed distance to it."""
    x, y = point
    if not np.isfinite(x) or not np.isfinite(y):
        return None, math.inf
    length, width = pitch
    distances = {
        "x_min": x,
        "x_max": length - x,
        "y_min": y,
        "y_max": width - y,
    }
    side = min(distances, key=distances.get)
    return side, float(distances[side])


def exit_side(frame, ball_xy, pitch):
    """Return which boundary an outside tracked point has crossed."""
    x, y = ball_xy[frame]
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    length, width = pitch
    if x < 0:
        return "x_min"
    if x > length:
        return "x_max"
    if y < 0:
        return "y_min"
    if y > width:
        return "y_max"
    return None


def load_contextual_weak_whistles(path, ball_xy, ball_speed, pitch, fps,
                                  boundary_margin=BOUNDARY_MARGIN_M,
                                  lookback_seconds=BOUNDARY_LOOKBACK_S,
                                  confirmation_seconds=BOUNDARY_CONFIRM_S,
                                  max_approach_speed=BOUNDARY_MAX_APPROACH_SPEED_MPS):
    """Accept quiet whistles only with recent and future same-boundary evidence."""
    exits = pitch_exit_events(ball_xy, pitch, fps)
    events = []
    lookback = int(round(lookback_seconds * fps))
    confirmation = int(round(confirmation_seconds * fps))

    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            duration = float(row["duration_seconds"])
            mean_ratio = float(row["mean_ratio"])
            max_ratio = float(row["max_ratio"])
            tonality = float(row["max_tonality"])
            if (duration < WEAK_MIN_DURATION_S or mean_ratio < WEAK_MIN_MEAN_RATIO
                    or max_ratio < WEAK_MIN_MAX_RATIO or tonality < WEAK_MIN_TONALITY):
                continue

            start = float(row["candidate_start_seconds"])
            end = float(row["candidate_end_seconds"])
            frame = min(len(ball_xy) - 1, max(0, int(math.ceil(start * fps))))

            recent_side = None
            recent_distance = math.inf
            recent_frame = None
            for prior in range(max(0, frame - lookback), frame + 1):
                side, distance = boundary_side(ball_xy[prior], pitch)
                if distance < recent_distance:
                    recent_side, recent_distance, recent_frame = side, distance, prior
            if recent_side is None or recent_distance > boundary_margin:
                continue
            if (recent_frame is None or not np.isfinite(ball_speed[recent_frame])
                    or ball_speed[recent_frame] > max_approach_speed):
                continue

            matching_exit = next((event for event in exits
                                  if frame <= event["frame"] <= frame + confirmation
                                  and exit_side(event["frame"], ball_xy, pitch) == recent_side),
                                 None)
            if matching_exit is None:
                continue

            events.append({
                "frame": frame,
                "source": "contextual_weak_whistle",
                "start": start,
                "end": end,
                "boundary": recent_side,
                "boundary_distance": recent_distance,
                "boundary_speed": float(ball_speed[recent_frame]),
                "confirming_exit": matching_exit["frame"],
            })
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR"))
    parser.add_argument("--strong-whistles", default=os.path.join(
        HERE, "audio", "full_whistle_candidates_v3.csv"))
    parser.add_argument("--weak-whistles", default=os.path.join(
        HERE, "audio", "full_whistle_candidates_weak_v4.csv"))
    parser.add_argument("--out", default=os.path.join(
        ROOT, "evaluation", "results", "dead_ball_v3.csv"))
    parser.add_argument("--intervals", default=os.path.join(
        HERE, "dead_ball_v3_intervals.csv"))
    args = parser.parse_args()

    fps = GAMES[GAME]["fps"]
    pitch = GAMES[GAME]["pitch"]
    n_frames = source_frames("tactical")
    meta_path, raw_path = require(["atd_meta", "atd_raw"], data_dir=args.data_dir)
    teams = team_of_each_track(meta_path)
    ball_xy, nearest_team, nearest_home, nearest_away = load_tracking_evidence(
        raw_path, teams, n_frames, pitch
    )
    speed, _ = planar_motion(ball_xy, fps)
    contacts = probable_contacts(nearest_home, nearest_away)
    base = possession_memory(nearest_team, fps)
    player_xy = load_player_positions(raw_path, n_frames, len(teams), pitch)
    activity = team_activity(player_xy, fps)

    strong = load_strong_whistles(args.strong_whistles, fps, n_frames)
    exits = pitch_exit_events(ball_xy, pitch, fps)
    weak = load_contextual_weak_whistles(
        args.weak_whistles, ball_xy, speed, pitch, fps
    )
    labels, intervals = dead_ball_latch(
        base, strong + exits + weak, nearest_team, contacts, speed, ball_xy, activity,
        nearest_home, nearest_away, fps, pitch=pitch,
    )
    write_submission(args.out, labels)
    write_intervals(args.intervals, intervals, fps)
    print(f"strong whistle stops: {len(strong)}")
    print(f"pitch-exit stops: {len(exits)}")
    print(f"contextual weak-whistle stops: {len(weak)}")
    for event in weak:
        print("  weak stop {:.3f}s, boundary {}, {:.2f}m at {:.2f}m/s, confirmed {:.3f}s".format(
            event["start"], event["boundary"], event["boundary_distance"],
            event["boundary_speed"],
            event["confirming_exit"] / fps,
        ))
    print(f"latched intervals: {len(intervals)}")
    print(f"wrote submission: {args.out}")
    print(f"wrote intervals: {args.intervals}")


if __name__ == "__main__":
    main()
