#!/usr/bin/env python3
"""Dead-ball v9: multi-evidence stops plus control-aware live possession.

A quiet whistle creates only a POSSIBLE_STOP.  It becomes a stop event when subsequent tracking
shows a sequence break, non-competitive ball behaviour, and a later restart that survives the
existing future-play checks.  Boundary evidence is an independent optional branch imported from
v3a; it is never required for an in-field foul.

The detector is offline by design: future frames may confirm and backdate either a stoppage or a
live possession boundary. Missing automatic ball coordinates are treated as unknown rather than
dead, while independently confirmed whistle/boundary intervals remain protected. No organizer
labels or match-specific timestamps are read here.
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

from games import (  # noqa: E402
    GAME, GAMES, anchor_video_frame, half_video_blocks, require, source_frames,
)
from starter.baseline import team_names, team_of_each_track, write_submission  # noqa: E402
from sequence_possession_experiment import (  # noqa: E402
    enforce_match_envelope, load_tracking_evidence, planar_motion,
    possession_chain_postprocess,
    possession_memory, probable_contacts,
)
from dead_ball_experiment import (  # noqa: E402
    MIN_WHISTLE_DURATION_S, MIN_WHISTLE_MAX_RATIO, MIN_WHISTLE_MEAN_RATIO,
    contact_restart_candidates, dead_ball_latch, load_player_positions,
    load_strong_whistles, pitch_exit_events, team_activity, write_intervals,
)
from dead_ball_context_experiment import (  # noqa: E402
    WEAK_MIN_DURATION_S, WEAK_MIN_MAX_RATIO, WEAK_MIN_MEAN_RATIO,
    WEAK_MIN_TONALITY, load_contextual_weak_whistles,
)


ACTIVITY_PRE_START_S = -2.0
ACTIVITY_PRE_END_S = -0.20
ACTIVITY_POST_START_S = 0.40
ACTIVITY_POST_END_S = 2.0
ACTIVITY_LATE_START_S = 2.0
ACTIVITY_LATE_END_S = 4.0
MIN_ACTIVITY_DROP_MPS = 0.35
MAX_ACTIVITY_RETAINED = 0.85
STOPPED_ACTIVITY_MPS = 1.50

NONCOMPETITIVE_START_S = 0.40
NONCOMPETITIVE_END_S = 2.0
MIN_BALL_PLAYER_DISTANCE_M = 2.50
MIN_MISSING_BALL_FRACTION = 0.30

MIN_RESTART_DELAY_S = 0.40
MAX_RESTART_SEARCH_S = 25.0
MIN_SEQUENCE_MEAN_RATIO = 0.060
MIN_SEQUENCE_MAX_RATIO = 0.100
MIN_MISSING_AT_ONSET_S = 2.0


def window(values, frame, start_seconds, end_seconds, fps):
    start = max(0, frame + int(round(start_seconds * fps)))
    end = min(len(values), frame + int(round(end_seconds * fps)))
    return values[start:end]


def finite_median(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else math.nan


def sequence_stop_evidence(frame, nearest_team, nearest_home, nearest_away, activity, fps):
    """Measure whether play visibly de-intensifies and the ball ceases to be contested."""
    pre = finite_median(window(
        activity, frame, ACTIVITY_PRE_START_S, ACTIVITY_PRE_END_S, fps
    ))
    early = finite_median(window(
        activity, frame, ACTIVITY_POST_START_S, ACTIVITY_POST_END_S, fps
    ))
    late = finite_median(window(
        activity, frame, ACTIVITY_LATE_START_S, ACTIVITY_LATE_END_S, fps
    ))
    if not np.isfinite(pre) or not np.isfinite(early) or not np.isfinite(late):
        return None

    activity_drop = (pre - early >= MIN_ACTIVITY_DROP_MPS
                     and early <= MAX_ACTIVITY_RETAINED * pre)
    stopped_level = min(early, late) <= STOPPED_ACTIVITY_MPS

    nearest = np.minimum(nearest_home, nearest_away)
    nearest_window = window(
        nearest, frame, NONCOMPETITIVE_START_S, NONCOMPETITIVE_END_S, fps
    )
    nearest_median = finite_median(nearest_window)
    team_window = window(
        nearest_team, frame, NONCOMPETITIVE_START_S, NONCOMPETITIVE_END_S, fps
    )
    missing_fraction = float(np.mean(team_window < 0)) if team_window.size else 0.0
    detached_ball = (np.isfinite(nearest_median)
                     and nearest_median >= MIN_BALL_PLAYER_DISTANCE_M)
    noncompetitive = detached_ball or missing_fraction >= MIN_MISSING_BALL_FRACTION

    return {
        "pre_activity": pre,
        "early_activity": early,
        "late_activity": late,
        "activity_drop": activity_drop,
        "stopped_level": stopped_level,
        "nearest_median": nearest_median,
        "missing_fraction": missing_fraction,
        "noncompetitive": noncompetitive,
    }


def first_confirmed_restart(frame, contacts, speed, ball_xy, activity,
                            nearest_home, nearest_away, fps):
    """Return a future-confirmed restart sufficiently after a possible whistle."""
    limit = min(len(contacts), frame + int(round(MAX_RESTART_SEARCH_S * fps)))
    minimum = frame + int(round(MIN_RESTART_DELAY_S * fps))
    for candidate in contact_restart_candidates(
            frame, limit, contacts, speed, ball_xy, activity,
            nearest_home, nearest_away, fps):
        if candidate["contact"] >= minimum:
            return candidate
    return None


def load_sequence_confirmed_weak_whistles(path, nearest_team, nearest_home, nearest_away,
                                           contacts, speed, ball_xy, activity, fps):
    """Confirm quiet in-field whistles from stop-pattern plus restart evidence."""
    events = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            duration = float(row["duration_seconds"])
            mean_ratio = float(row["mean_ratio"])
            max_ratio = float(row["max_ratio"])
            tonality = float(row["max_tonality"])
            if (duration < WEAK_MIN_DURATION_S or mean_ratio < WEAK_MIN_MEAN_RATIO
                    or max_ratio < WEAK_MIN_MAX_RATIO or tonality < WEAK_MIN_TONALITY):
                continue
            # Strong candidates already have their own conservative path.  Do not use the more
            # permissive sequence branch to move a strong whistle's onset earlier.
            if (duration >= MIN_WHISTLE_DURATION_S
                    and mean_ratio >= MIN_WHISTLE_MEAN_RATIO
                    and max_ratio >= MIN_WHISTLE_MAX_RATIO):
                continue
            if mean_ratio < MIN_SEQUENCE_MEAN_RATIO or max_ratio < MIN_SEQUENCE_MAX_RATIO:
                continue

            start = float(row["candidate_start_seconds"])
            end = float(row["candidate_end_seconds"])
            frame = min(len(nearest_team) - 1, max(0, int(math.ceil(start * fps))))
            if nearest_team[frame] < 0:
                first_valid = frame
                while first_valid < len(nearest_team) and nearest_team[first_valid] < 0:
                    first_valid += 1
                if ((first_valid - frame) / fps < MIN_MISSING_AT_ONSET_S):
                    continue
            evidence = sequence_stop_evidence(
                frame, nearest_team, nearest_home, nearest_away, activity, fps
            )
            if evidence is None:
                continue
            if not (evidence["activity_drop"] and evidence["stopped_level"]
                    and evidence["noncompetitive"]):
                continue

            restart = first_confirmed_restart(
                frame, contacts, speed, ball_xy, activity,
                nearest_home, nearest_away, fps,
            )
            if restart is None:
                continue

            events.append({
                "frame": frame,
                "source": "sequence_weak_whistle",
                "start": start,
                "end": end,
                "evidence": evidence,
                "confirming_contact": int(restart["contact"]),
                "confirming_release": int(restart["release"]),
                "restart_override": {
                    "release": int(restart["release"]),
                    "contact": int(restart["contact"]),
                    "team": int(restart["team"]),
                    "method": "sequence_future_confirmation",
                },
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
        ROOT, "evaluation", "results", "dead_ball_v9.csv"))
    parser.add_argument("--intervals", default=os.path.join(
        HERE, "dead_ball_v9_intervals.csv"))
    parser.add_argument("--events", default=os.path.join(
        HERE, "dead_ball_v9_weak_events.csv"))
    args = parser.parse_args()

    fps = GAMES[GAME]["fps"]
    pitch = GAMES[GAME]["pitch"]
    n_frames = source_frames("tactical")
    meta_path, raw_path, smart_path = require(
        ["atd_meta", "atd_raw", "smart_tagging"], data_dir=args.data_dir
    )
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
    boundary_weak = load_contextual_weak_whistles(
        args.weak_whistles, ball_xy, speed, pitch, fps
    )
    sequence_weak = load_sequence_confirmed_weak_whistles(
        args.weak_whistles, nearest_team, nearest_home, nearest_away,
        contacts, speed, ball_xy, activity, fps,
    )

    labels, intervals = dead_ball_latch(
        base, strong + exits + boundary_weak + sequence_weak,
        nearest_team, contacts, speed, ball_xy, activity,
        nearest_home, nearest_away, fps, pitch=pitch,
    )
    labels, possession_events = possession_chain_postprocess(
        labels, nearest_team, nearest_home, nearest_away, contacts, speed, fps,
        smart_tagging_path=smart_path, club_names=team_names(meta_path),
        dead_intervals=intervals,
    )
    blocks = half_video_blocks(GAME)
    first_half_start, first_half_end = blocks[0][1], blocks[0][2]
    second_half_end = blocks[1][2]
    second_half_kickoff = anchor_video_frame(GAMES[GAME]["kickoff_2nd"])
    labels = enforce_match_envelope(
        labels, first_half_start, first_half_end,
        second_half_kickoff, second_half_end,
    )
    write_submission(args.out, labels)
    write_intervals(args.intervals, intervals, fps)

    with open(args.events, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "frame", "time", "source", "pre_activity", "early_activity",
            "late_activity", "nearest_median", "missing_fraction",
            "confirming_contact", "confirming_contact_time",
        ])
        for event in sequence_weak:
            evidence = event["evidence"]
            writer.writerow([
                event["frame"], f"{event['frame'] / fps:.3f}", event["source"],
                f"{evidence['pre_activity']:.3f}",
                f"{evidence['early_activity']:.3f}",
                f"{evidence['late_activity']:.3f}",
                f"{evidence['nearest_median']:.3f}",
                f"{evidence['missing_fraction']:.3f}",
                event["confirming_contact"],
                f"{event['confirming_contact'] / fps:.3f}",
            ])

    print(f"strong whistle stops: {len(strong)}")
    print(f"pitch-exit stops: {len(exits)}")
    print(f"boundary-assisted weak stops: {len(boundary_weak)}")
    print(f"sequence-confirmed weak stops: {len(sequence_weak)}")
    for event in sequence_weak:
        evidence = event["evidence"]
        print("  sequence stop {:.3f}s: activity {:.2f}->{:.2f}, nearest {:.2f}m, "
              "missing {:.0%}, restart {:.3f}s".format(
                  event["start"], evidence["pre_activity"],
                  evidence["early_activity"], evidence["nearest_median"],
                  evidence["missing_fraction"], event["confirming_contact"] / fps,
              ))
    print(f"latched intervals: {len(intervals)}")
    print(f"possession-chain repairs: {len(possession_events)}")
    print(f"wrote submission: {args.out}")
    print(f"wrote intervals: {args.intervals}")
    print(f"wrote weak-event diagnostics: {args.events}")


if __name__ == "__main__":
    main()
