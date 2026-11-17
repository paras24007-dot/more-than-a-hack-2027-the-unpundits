#!/usr/bin/env python3
"""Conservative dead-ball latch and offline restart-confirmation experiment.

The module never reads organizer ground truth.  It starts from the sequence-possession baseline,
adds high-confidence whistle and pitch-exit stoppage triggers, then keeps D latched until a restart
candidate survives future-frame checks.  Intermediate touches used to return or position the ball
remain D.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import warnings

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from games import GAME, GAMES, require, source_frames  # noqa: E402
from starter.baseline import team_of_each_track, write_submission  # noqa: E402
from sequence_possession_experiment import (  # noqa: E402
    AWAY, CONTACT_RADIUS_M, HOME, NOBODY, TEAM_LABEL, load_tracking_evidence,
    planar_motion, possession_memory, probable_contacts,
)


# Recall-oriented audio candidates are filtered into conservative whistle triggers.
MIN_WHISTLE_DURATION_S = 0.50
MIN_WHISTLE_MEAN_RATIO = 0.12
MIN_WHISTLE_MAX_RATIO = 0.20

# Restart evidence.  Values are generic physical/time priors and are exposed as CLI arguments.
CONTACT_PRE_WINDOW_S = 0.32
RESTART_LOOKAHEAD_S = 3.0
RESTART_SPEED_MPS = 4.5
RESTART_MIN_DISPLACEMENT_M = 4.0
BOUNDARY_RESTART_LOOKBACK_S = 3.0
BOUNDARY_RESTART_MARGIN_M = 6.0
RECEIVER_MAX_DISTANCE_M = 4.0
RECEIVER_MIN_CLOSURE_M = CONTACT_RADIUS_M
RECEIVER_MIN_CLOSURE_FRACTION = 0.35
ACTIVITY_FUTURE_START_S = 2.0
ACTIVITY_FUTURE_END_S = 4.0
ACTIVITY_MIN_MPS = 1.30
ACTIVITY_INCREASE_MPS = 0.35
MISSING_RESUME_CONFIRM_S = 0.40
MAX_LATCH_S = 45.0


def load_player_positions(raw_path, n_frames, n_tracks, pitch):
    """Load all player centres in metres for movement-intensity confirmation."""
    length, width = pitch
    positions = np.full((n_frames, n_tracks, 2), np.nan, dtype=np.float32)
    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if frame < 0 or frame >= n_frames:
                continue
            for slot, group in enumerate(parts[1].split(";")[:n_tracks]):
                values = group.split(",")
                if values[0] == "NaN" or values[1] == "NaN":
                    continue
                positions[frame, slot] = (float(values[0]) * length,
                                          float(values[1]) * width)
    return positions


def team_activity(player_xy, fps, smooth_seconds=0.40):
    """Return robust whole-match activity as median smoothed player speed per frame."""
    velocity = np.diff(player_xy, axis=0, prepend=np.nan) * fps
    speed = np.linalg.norm(velocity, axis=2)
    window = max(1, int(round(smooth_seconds * fps)))
    smoothed = np.full_like(speed, np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for frame in range(window - 1, len(speed)):
            smoothed[frame] = np.nanmedian(speed[frame - window + 1:frame + 1], axis=0)
        activity = np.nanmedian(smoothed, axis=1)
    return activity


def load_strong_whistles(path, fps, n_frames,
                         min_duration=MIN_WHISTLE_DURATION_S,
                         min_mean_ratio=MIN_WHISTLE_MEAN_RATIO,
                         min_max_ratio=MIN_WHISTLE_MAX_RATIO):
    """Convert enriched spectral candidates into conservative stop events."""
    events = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("mean_ratio"):
                raise ValueError("whistle CSV lacks spectral summaries; regenerate with v3 script")
            duration = float(row["duration_seconds"])
            mean_ratio = float(row["mean_ratio"])
            max_ratio = float(row["max_ratio"])
            if (duration < min_duration or mean_ratio < min_mean_ratio
                    or max_ratio < min_max_ratio):
                continue
            start = float(row["candidate_start_seconds"])
            end = float(row["candidate_end_seconds"])
            frame = min(n_frames - 1, max(0, int(math.ceil(end * fps))))
            events.append({"frame": frame, "source": "whistle", "start": start, "end": end})
    return events


def pitch_exit_events(ball_xy, pitch, fps, confirm_seconds=0.12):
    """Find sustained inside-to-outside ball crossings."""
    length, width = pitch
    valid = ~np.isnan(ball_xy[:, 0])
    outside = valid & (
        (ball_xy[:, 0] < 0.0) | (ball_xy[:, 0] > length)
        | (ball_xy[:, 1] < 0.0) | (ball_xy[:, 1] > width)
    )
    confirm = max(1, int(math.ceil(confirm_seconds * fps)))
    events = []
    for frame in range(1, len(ball_xy) - confirm + 1):
        if not outside[frame] or outside[frame - 1]:
            continue
        if np.sum(outside[frame:frame + confirm]) < confirm:
            continue
        x, y = ball_xy[frame]
        length, width = pitch
        if x < 0.0:
            boundary = "x_min"
        elif x > length:
            boundary = "x_max"
        elif y < 0.0:
            boundary = "y_min"
        else:
            boundary = "y_max"
        events.append({"frame": frame, "source": "pitch_exit",
                       "start": frame / fps, "end": frame / fps,
                       "boundary": boundary})
    return events


def distance_to_boundary(point, boundary, pitch):
    """Signed in-pitch distance from one point to the named pitch boundary."""
    x, y = point
    if not np.isfinite(x) or not np.isfinite(y):
        return math.inf
    length, width = pitch
    return {
        "x_min": x,
        "x_max": length - x,
        "y_min": y,
        "y_max": width - y,
    }.get(boundary, math.inf)


def boundary_restart_receiving_phase(stop, confirmed_restart, boundary, pitch,
                                     nearest_team, contacts, speed, ball_xy,
                                     nearest_home, nearest_away, fps):
    """Backtrack a confirmed boundary restart to its receiving/control phase.

    A corner, throw-in or goal kick can be confirmed only after a later reception and broader
    resumption of play.  The later touch is evidence that the restart was genuine; it is not
    necessarily the restart boundary itself.  This helper looks for the preceding departure from
    the relevant pitch boundary, then releases D when the same-team receiver is materially closing
    on the travelling ball.  It deliberately applies only to boundary/set-piece events: an in-field
    foul may contain several administrative passes before the true restart.
    """
    if boundary is None or confirmed_restart is None:
        return None
    team = int(confirmed_restart["team"])
    contact = int(confirmed_restart["contact"])
    distances = nearest_home if team == HOME else nearest_away
    lookback = int(round(BOUNDARY_RESTART_LOOKBACK_S * fps))
    first = max(stop + 1, contact - lookback)

    departures = []
    for frame in range(first, contact):
        same_team_contact = int(contacts[frame]) == team
        tracking_resumed = (int(nearest_team[frame]) == team
                            and frame > 0 and int(nearest_team[frame - 1]) == NOBODY)
        if not (same_team_contact or tracking_resumed):
            continue
        if distance_to_boundary(ball_xy[frame], boundary, pitch) > BOUNDARY_RESTART_MARGIN_M:
            continue
        departures.append(frame)

    # Prefer the latest plausible pass into the already-confirmed receiver.  Earlier contacts can
    # merely be players positioning the ball for the set piece.
    for departure in reversed(departures):
        segment = ball_xy[departure:contact + 1]
        if len(segment) < 3 or np.isnan(segment[:, 0]).any():
            continue
        net_displacement = float(np.linalg.norm(segment[-1] - segment[0]))
        if net_displacement < RESTART_MIN_DISPLACEMENT_M:
            continue
        speed_segment = speed[departure:contact + 1]
        if np.isnan(speed_segment).all() or float(np.nanmax(speed_segment)) < RESTART_SPEED_MPS:
            continue

        peak = -math.inf
        for frame in range(departure, contact + 1):
            distance = distances[frame]
            if not np.isfinite(distance):
                continue
            peak = max(peak, float(distance))
            required_closure = max(
                RECEIVER_MIN_CLOSURE_M,
                RECEIVER_MIN_CLOSURE_FRACTION * peak,
            )
            if (distance <= RECEIVER_MAX_DISTANCE_M
                    and peak - distance >= required_closure):
                return {
                    "release": frame,
                    "departure": departure,
                    "confirming_contact": contact,
                    "peak_receiver_distance": peak,
                    "receiver_distance": float(distance),
                }
    return None


def first_valid_after_missing(stop, nearest_team, fps):
    """Return a conservative release when a stopped interval ends with tracking reappearing."""
    if nearest_team[stop] != NOBODY:
        return None
    frame = stop
    while frame < len(nearest_team) and nearest_team[frame] == NOBODY:
        frame += 1
    if frame >= len(nearest_team):
        return None
    return min(len(nearest_team) - 1,
               frame + int(math.ceil(MISSING_RESUME_CONFIRM_S * fps)))


def contact_restart_candidates(stop, limit, contacts, speed, ball_xy, activity,
                               nearest_home, nearest_away, fps,
                               speed_threshold=RESTART_SPEED_MPS):
    """Yield contacts that lead into sustained ball and multi-player activity."""
    pre = max(1, int(round(CONTACT_PRE_WINDOW_S * fps)))
    displacement_frames = int(round(RESTART_LOOKAHEAD_S * fps))
    future_a = int(round(ACTIVITY_FUTURE_START_S * fps))
    future_b = int(round(ACTIVITY_FUTURE_END_S * fps))

    for contact in range(stop + 1, min(limit, len(contacts))):
        team = int(contacts[contact])
        if team == NOBODY:
            continue
        end = contact + future_b
        displacement_end = contact + displacement_frames
        if end >= len(activity) or displacement_end >= len(ball_xy):
            continue
        before_speed = speed[max(0, contact - pre):contact + 1]
        after_speed = speed[contact:min(len(speed), contact + future_a + 1)]
        if np.isnan(before_speed).all() or np.isnan(after_speed).all():
            continue
        if float(np.nanmin(before_speed)) > 2.0 or float(np.nanmax(after_speed)) < 3.0:
            continue
        if np.isnan(ball_xy[[contact, displacement_end], 0]).any():
            continue
        displacement = float(np.linalg.norm(ball_xy[displacement_end] - ball_xy[contact]))
        if displacement < RESTART_MIN_DISPLACEMENT_M:
            continue

        baseline_slice = activity[max(stop, contact - int(2 * fps)):contact + 1]
        future_slice = activity[contact + future_a:contact + future_b + 1]
        if np.isnan(baseline_slice).all() or np.isnan(future_slice).all():
            continue
        baseline = float(np.nanmedian(baseline_slice))
        future = float(np.nanmedian(future_slice))
        if future < ACTIVITY_MIN_MPS or future - baseline < ACTIVITY_INCREASE_MPS:
            continue

        release = None
        for frame in range(contact, min(len(speed), contact + future_a + 1)):
            nearest = nearest_home[frame] if team == HOME else nearest_away[frame]
            if (not np.isnan(speed[frame]) and speed[frame] >= speed_threshold
                    and not np.isnan(nearest) and nearest >= 1.5):
                release = frame
                break
        if release is None:
            release = contact + int(round(0.8 * fps))
        yield {
            "contact": contact,
            "release": release,
            "team": team,
            "baseline_activity": baseline,
            "future_activity": future,
            "displacement": displacement,
        }


def find_restart(stop, nearest_team, contacts, speed, ball_xy, activity,
                 nearest_home, nearest_away, fps, event=None, pitch=None):
    """Find the first restart that survives offline confirmation after a stop."""
    missing_release = first_valid_after_missing(stop, nearest_team, fps)
    if missing_release is not None:
        return {"release": missing_release, "contact": missing_release,
                "team": int(nearest_team[missing_release]), "method": "tracking_resume"}

    limit = min(len(nearest_team), stop + int(round(MAX_LATCH_S * fps)))
    candidates = contact_restart_candidates(
        stop, limit, contacts, speed, ball_xy, activity, nearest_home, nearest_away, fps
    )
    try:
        candidate = next(candidates)
    except StopIteration:
        return None
    candidate["method"] = "contact_future_confirmation"
    boundary = (event or {}).get("boundary")
    if boundary is not None and pitch is not None:
        receiving = boundary_restart_receiving_phase(
            stop, candidate, boundary, pitch, nearest_team, contacts, speed, ball_xy,
            nearest_home, nearest_away, fps,
        )
        if receiving is not None and receiving["release"] < candidate["release"]:
            candidate["release"] = receiving["release"]
            candidate["method"] = "boundary_receiving_phase"
            candidate["departure"] = receiving["departure"]
            candidate["confirming_contact"] = receiving["confirming_contact"]
    return candidate


def dead_ball_latch(base_labels, stop_events, nearest_team, contacts, speed, ball_xy,
                    activity, nearest_home, nearest_away, fps, pitch=None):
    """Overlay confirmed dead intervals while preserving the untouched baseline elsewhere."""
    labels = np.array(base_labels, copy=True)
    intervals = []
    covered_until = -1
    for event in sorted(stop_events, key=lambda row: row["frame"]):
        stop = int(event["frame"])
        if stop <= covered_until:
            continue
        restart = event.get("restart_override")
        if restart is None:
            restart = find_restart(
                stop, nearest_team, contacts, speed, ball_xy, activity,
                nearest_home, nearest_away, fps, event=event, pitch=pitch,
            )
        if restart is None or restart["release"] <= stop:
            continue
        release = min(len(labels), int(restart["release"]))
        labels[stop:release] = "D"
        covered_until = release - 1
        intervals.append({
            "stop": stop,
            "release": release,
            "source": event["source"],
            "restart_method": restart["method"],
            "restart_team": TEAM_LABEL.get(int(restart["team"]), "D"),
            "contact": int(restart["contact"]),
        })
    return labels, intervals


def write_intervals(path, intervals, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stop_frame", "stop_time", "release_frame", "release_time",
                         "duration_s", "source", "restart_method", "restart_team",
                         "candidate_contact_frame", "candidate_contact_time"])
        for row in intervals:
            writer.writerow([
                row["stop"], f"{row['stop'] / fps:.3f}",
                row["release"], f"{row['release'] / fps:.3f}",
                f"{(row['release'] - row['stop']) / fps:.3f}", row["source"],
                row["restart_method"], row["restart_team"], row["contact"],
                f"{row['contact'] / fps:.3f}",
            ])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR"))
    parser.add_argument("--whistles", default=os.path.join(HERE, "audio",
                                                            "full_whistle_candidates_v3.csv"))
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results",
                                                       "dead_ball_v2.csv"))
    parser.add_argument("--intervals", default=os.path.join(HERE,
                                                             "dead_ball_v2_intervals.csv"))
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

    events = load_strong_whistles(args.whistles, fps, n_frames)
    events += pitch_exit_events(ball_xy, pitch, fps)
    labels, intervals = dead_ball_latch(
        base, events, nearest_team, contacts, speed, ball_xy, activity,
        nearest_home, nearest_away, fps, pitch=pitch,
    )
    write_submission(args.out, labels)
    write_intervals(args.intervals, intervals, fps)
    print(f"strong whistle stops: {sum(e['source'] == 'whistle' for e in events)}")
    print(f"pitch-exit stops: {sum(e['source'] == 'pitch_exit' for e in events)}")
    print(f"latched intervals: {len(intervals)}")
    print(f"wrote submission: {args.out}")
    print(f"wrote intervals: {args.intervals}")


if __name__ == "__main__":
    main()
