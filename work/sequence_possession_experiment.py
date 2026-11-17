#!/usr/bin/env python3
"""Tracking-only possession-memory experiment.

This script deliberately does not read public or private ground truth.  It turns the automatic
tracking into a full-match submission using two separate concepts:

* instantaneous evidence: which team has the nearest tracked player to the ball;
* confirmed possession: the team whose evidence has survived a short confirmation period.

The separation prevents a pass in flight from changing possession merely because an opponent is
temporarily a few centimetres nearer to the ball.  It is a first experiment, not a final model:
dead-ball inference and video evidence are intentionally left for later modules.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, GAMES, require, source_frames  # noqa: E402
from starter.baseline import team_of_each_track, write_submission  # noqa: E402


HOME, AWAY, NOBODY = 0, 1, -1
TEAM_LABEL = {HOME: "H", AWAY: "A", NOBODY: "D"}

# Generic football priors, not fitted to the public five-minute labels.
CONTACT_RADIUS_M = 1.5
CONTACT_LOCAL_WINDOW = 2
TURNOVER_CONFIRM_SECONDS = 0.8

# Offline sequence repair. These are generic physical priors, not match timestamps.
SHORT_MISSING_GAP_SECONDS = 0.20
PASS_CONTACT_SPEED_MPS = 8.0
PASS_LOOKBACK_SECONDS = 1.5
PASS_RECEPTION_LOOKAHEAD_SECONDS = 3.0
CONTROL_DEPARTURE_SPEED_MPS = 6.5
CONTROL_RECEPTION_SPEED_MPS = 1.0
CONTROL_RECEPTION_DISTANCE_M = 1.1
KICKOFF_BACKFILL_SECONDS = 0.08
LONG_BALL_LANDING_LOOKBACK_SECONDS = 2.0
LONG_BALL_LANDING_LOOKAHEAD_SECONDS = 0.5
LONG_BALL_PRE_SPEED_MPS = 8.0
LONG_BALL_LANDING_SPEED_MPS = 6.0
LONG_BALL_CONFIRM_SECONDS = 0.8
LONG_BALL_CONFIRM_LOOKAHEAD_SECONDS = 6.0

# A missing ball coordinate means "unknown", not "dead".  These limits are used only when no
# independently detected dead-ball interval overlaps the gap.
MISSING_LIVE_LOOKBACK_SECONDS = 3.0
MAX_MISSING_LIVE_GAP_SECONDS = 12.0
POST_MISSING_TURNOVER_CONFIRM_SECONDS = 0.60

# Controlled-turnover timing.  Nearest-player evidence may lead a real interception in a press,
# or lag it when the automatic ball track drifts.  A low-speed control phase followed by a
# same-team action is stronger evidence than proximity alone.
TURNOVER_ACTION_LOOKBACK_SECONDS = 2.0
TURNOVER_ACTION_LOOKAHEAD_SECONDS = 3.0
DELIBERATE_DEPARTURE_SPEED_MPS = 10.0
MAX_DEPARTURE_BOUNDARY_SHIFT_SECONDS = 0.60
CONTROL_TROUGH_SPEED_MPS = 1.2
CONTROL_PARITY_MARGIN_M = 0.5
UNCONFIRMED_SPELL_MAX_SECONDS = 3.2
UNCONFIRMED_SPELL_MAX_CONTACTS = 2
INTERCEPTION_CONFIRM_LOOKBACK_SECONDS = 2.0
INTERCEPTION_CONFIRM_LOOKAHEAD_SECONDS = 0.5
INTERCEPTION_MIN_CONTACT_SEPARATION_SECONDS = 0.4
INTERCEPTION_CONTACT_BACKFILL_SECONDS = 0.08
INTERCEPTION_FIRST_CONTACT_DISTANCE_M = 0.5


def load_tracking_evidence(raw_path, teams, n_frames, pitch):
    """Return ball geometry and nearest-team evidence for every tactical-video frame."""
    length, width = pitch
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float64)
    nearest_team = np.full(n_frames, NOBODY, dtype=np.int8)
    nearest_home = np.full(n_frames, np.nan, dtype=np.float64)
    nearest_away = np.full(n_frames, np.nan, dtype=np.float64)

    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if frame < 0 or frame >= n_frames:
                continue

            ball = parts[2].split(",")
            if ball[0] == "NaN" or ball[1] == "NaN":
                continue
            bx, by = float(ball[0]) * length, float(ball[1]) * width
            ball_xy[frame] = (bx, by)

            best = [math.inf, math.inf]
            for slot, group in enumerate(parts[1].split(";")):
                values = group.split(",")
                if values[0] == "NaN" or values[1] == "NaN":
                    continue
                px, py = float(values[0]) * length, float(values[1]) * width
                distance = math.hypot(px - bx, py - by)
                team = int(teams[slot])
                if distance < best[team]:
                    best[team] = distance

            if math.isfinite(best[HOME]):
                nearest_home[frame] = best[HOME]
            if math.isfinite(best[AWAY]):
                nearest_away[frame] = best[AWAY]
            if math.isfinite(best[HOME]) or math.isfinite(best[AWAY]):
                nearest_team[frame] = HOME if best[HOME] <= best[AWAY] else AWAY

    return ball_xy, nearest_team, nearest_home, nearest_away


def planar_motion(ball_xy, fps):
    """Derive ground-plane speed and turn angle; the supplied ball speed is often NaN."""
    n_frames = len(ball_xy)
    velocity = np.full((n_frames, 2), np.nan, dtype=np.float64)
    valid_pair = (~np.isnan(ball_xy[1:, 0])) & (~np.isnan(ball_xy[:-1, 0]))
    velocity[1:][valid_pair] = (ball_xy[1:][valid_pair] - ball_xy[:-1][valid_pair]) * fps
    speed = np.linalg.norm(velocity, axis=1)

    turn = np.full(n_frames, np.nan, dtype=np.float64)
    for frame in range(2, n_frames):
        before, after = velocity[frame - 1], velocity[frame]
        if np.isnan(before[0]) or np.isnan(after[0]):
            continue
        denom = np.linalg.norm(before) * np.linalg.norm(after)
        if denom <= 1e-9:
            continue
        cosine = np.clip(float(np.dot(before, after) / denom), -1.0, 1.0)
        turn[frame] = math.degrees(math.acos(cosine))
    return speed, turn


def probable_contacts(nearest_home, nearest_away, radius=CONTACT_RADIUS_M,
                      local_window=CONTACT_LOCAL_WINDOW):
    """Infer contact candidates from a team's local distance minimum.

    This is intentionally labelled probable: 2D tracking has no touch flag and no ball height.
    """
    n_frames = len(nearest_home)
    out = np.full(n_frames, NOBODY, dtype=np.int8)
    for frame in range(local_window, n_frames - local_window):
        candidates = []
        for team, distances in ((HOME, nearest_home), (AWAY, nearest_away)):
            distance = distances[frame]
            if np.isnan(distance) or distance > radius:
                continue
            neighbourhood = distances[frame - local_window:frame + local_window + 1]
            if np.isnan(neighbourhood).all() or distance > np.nanmin(neighbourhood):
                continue
            candidates.append((distance, team))
        if candidates:
            out[frame] = min(candidates)[1]
    return out


def possession_memory(nearest_team, fps, confirm_seconds=TURNOVER_CONFIRM_SECONDS):
    """Debounce nearest-team evidence into a confirmed possession sequence.

    Missing-ball frames remain D in this first experiment.  Within each observed-ball block, an
    opposing nearest-team run must survive `confirm_seconds` before it changes confirmed ownership.
    No labels are rewritten from ground truth and no match-specific frame is referenced.
    """
    confirm_frames = max(1, int(math.ceil(confirm_seconds * fps)))
    labels = np.full(len(nearest_team), "D", dtype="<U1")
    confirmed = NOBODY
    candidate = NOBODY
    candidate_start = -1

    for frame, nearest in enumerate(nearest_team):
        if nearest == NOBODY:
            confirmed = NOBODY
            candidate = NOBODY
            candidate_start = -1
            continue

        if confirmed == NOBODY:
            confirmed = int(nearest)
            labels[frame] = TEAM_LABEL[confirmed]
            continue

        if nearest == confirmed:
            candidate = NOBODY
            candidate_start = -1
            labels[frame] = TEAM_LABEL[confirmed]
            continue

        if nearest != candidate:
            candidate = int(nearest)
            candidate_start = frame

        candidate_frames = frame - candidate_start + 1
        if candidate_frames >= confirm_frames:
            confirmed = candidate
            candidate = NOBODY
            candidate_start = -1

        labels[frame] = TEAM_LABEL[confirmed]

    return labels


def bridge_short_missing_gaps(labels, fps, max_seconds=SHORT_MISSING_GAP_SECONDS):
    """Fill a tiny D gap only when the same live team appears on both sides.

    Missing automatic ball tracking is not evidence that the referee stopped play. A gap longer
    than this conservative limit, or one spanning a team change, is left untouched.
    """
    out = np.array(labels, copy=True)
    limit = max(1, int(round(max_seconds * fps)))
    frame = 0
    while frame < len(out):
        if out[frame] != "D":
            frame += 1
            continue
        start = frame
        while frame < len(out) and out[frame] == "D":
            frame += 1
        stop = frame
        if (stop - start <= limit and start > 0 and stop < len(out)
                and out[start - 1] in ("H", "A") and out[start - 1] == out[stop]):
            out[start:stop] = out[start - 1]
    return out


def _protected_dead_mask(n_frames, dead_intervals):
    """Frames explicitly declared dead by whistle/boundary evidence."""
    protected = np.zeros(n_frames, dtype=bool)
    for row in dead_intervals or []:
        start = max(0, int(row["stop"]))
        stop = min(n_frames, int(row["release"]))
        if stop > start:
            protected[start:stop] = True
    return protected


def recover_untracked_live_play(labels, nearest_team, contacts, fps,
                                dead_intervals=None, smart_context=None):
    """Carry confirmed ownership through missing automatic ball coordinates.

    The tracking provider sometimes loses the ball for several seconds during completely live
    play.  We use the last probable player interaction as the owner, preserve any independently
    detected dead-ball interval, and require fresh opponent evidence after tracking returns.
    Nothing here reads organizer labels or match timestamps.
    """
    out = np.array(labels, copy=True)
    protected = _protected_dead_mask(len(out), dead_intervals)
    lookback = max(1, int(round(MISSING_LIVE_LOOKBACK_SECONDS * fps)))
    max_gap = max(1, int(round(MAX_MISSING_LIVE_GAP_SECONDS * fps)))
    post_confirm = max(1, int(round(POST_MISSING_TURNOVER_CONFIRM_SECONDS * fps)))
    events = []

    frame = 0
    while frame < len(out):
        if int(nearest_team[frame]) != NOBODY:
            frame += 1
            continue
        start = frame
        while frame < len(out) and int(nearest_team[frame]) == NOBODY:
            frame += 1
        stop = frame
        if stop - start > max_gap:
            continue

        context_team = NOBODY
        if smart_context is not None:
            context_values = smart_context[start:stop]
            known = context_values[context_values != NOBODY]
            if known.size:
                counts = np.bincount(known, minlength=2)
                context_team = int(np.argmax(counts))

        # A tiny dropout with the same observed team on both sides is stronger than a broad,
        # automatically generated tactical-phase tag.
        short_same_team = NOBODY
        short_limit = max(1, int(round(SHORT_MISSING_GAP_SECONDS * fps)))
        if (stop - start <= short_limit and start > 0 and stop < len(out)
                and int(nearest_team[start - 1]) != NOBODY
                and int(nearest_team[start - 1]) == int(nearest_team[stop])):
            short_same_team = int(nearest_team[start - 1])

        contact = None
        for candidate in range(start - 1, max(-1, start - lookback - 1), -1):
            if int(contacts[candidate]) != NOBODY:
                contact = candidate
                break
        if short_same_team != NOBODY:
            owner = short_same_team
            owner_start = start
        elif context_team != NOBODY:
            owner = context_team
            owner_start = start
        elif contact is not None:
            owner = int(contacts[contact])
            owner_start = contact
        elif start > 0 and int(nearest_team[start - 1]) != NOBODY:
            owner = int(nearest_team[start - 1])
            owner_start = start
        else:
            continue

        owner_label = TEAM_LABEL[owner]
        changed = []
        # Recover only portions not protected by a real stop.  Starting at the confirming contact
        # also repairs a false nearest-player turnover during the pass that led into the dropout.
        for candidate in range(owner_start, stop):
            if not protected[candidate] and out[candidate] != owner_label:
                out[candidate] = owner_label
                changed.append(candidate)

        # If the ball reappears nearer the opponent, do not treat that first noisy frame as an
        # instantaneous turnover.  Preserve the old owner for 0.60 s; a sustained opponent run
        # then changes ownership on the following frame.
        if stop < len(out) and not protected[stop]:
            observed = int(nearest_team[stop])
            if observed not in (NOBODY, owner):
                hold_stop = min(len(out), stop + post_confirm - 1)
                for candidate in range(stop, hold_stop):
                    if protected[candidate] or int(nearest_team[candidate]) == NOBODY:
                        break
                    if int(nearest_team[candidate]) == owner:
                        break
                    if out[candidate] != owner_label:
                        out[candidate] = owner_label
                        changed.append(candidate)

        if changed:
            events.append({
                "method": "untracked_live_play",
                "team": owner_label,
                "gap_start": start,
                "gap_stop": stop,
                "first_changed": min(changed),
                "last_changed": max(changed),
            })
    return out, events


def smart_tagging_live_context(path, club_names, n_frames, fps):
    """Team context for missing tracking, from broad automatic live-play phases only.

    Progression and attacking-transition tags are not possession labels and are never allowed to
    overwrite observed tracking.  They are used solely to choose between teams while the ball
    coordinate itself is absent.
    """
    context = np.full(n_frames, NOBODY, dtype=np.int8)
    priority = np.zeros(n_frames, dtype=np.int8)
    code_priority = {"progression": 1, "attacking transition": 2}
    for instance in ET.parse(path).iter("instance"):
        code = (instance.findtext("code") or "").strip().lower()
        if code not in code_priority:
            continue
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        team = _tag_team_side(tags.get("Team"), club_names)
        if team == NOBODY:
            continue
        start = max(0, int(round(float(instance.findtext("start")) * fps)))
        stop = min(n_frames, int(round(float(instance.findtext("end")) * fps)) + 1)
        rank = code_priority[code]
        replace = priority[start:stop] < rank
        context[start:stop][replace] = team
        priority[start:stop][replace] = rank
    return context


def controlled_turnover_boundary_repairs(labels, nearest_team, nearest_home,
                                         nearest_away, contacts, speed, fps):
    """Place live turnovers at a football action rather than a proximity threshold.

    Two conservative corrections are made:

    * delay a nearest-player turnover when the old team acted last and the new team only later
      produces a close touch followed by a deliberate departure;
    * backdate a delayed turnover to a low-speed control trough when a later same-team contact
      confirms the new owner.
    """
    out = np.array(labels, copy=True)
    events = []
    lookback = max(1, int(round(TURNOVER_ACTION_LOOKBACK_SECONDS * fps)))
    lookahead = max(1, int(round(TURNOVER_ACTION_LOOKAHEAD_SECONDS * fps)))
    max_departure_shift = max(1, int(round(MAX_DEPARTURE_BOUNDARY_SHIFT_SECONDS * fps)))

    # Snapshot transitions first so edits in this pass do not manufacture more candidates.
    transitions = [frame for frame in range(1, len(out))
                   if out[frame] in ("H", "A") and out[frame - 1] in ("H", "A")
                   and out[frame] != out[frame - 1]]
    for transition in transitions:
        old_label, new_label = out[transition - 1], out[transition]
        old_team = HOME if old_label == "H" else AWAY
        new_team = HOME if new_label == "H" else AWAY

        old_contact = next((frame for frame in range(transition - 1,
                            max(-1, transition - lookback - 1), -1)
                            if int(contacts[frame]) == old_team), None)
        new_contact = next((frame for frame in range(transition,
                            min(len(out), transition + lookahead + 1))
                            if int(contacts[frame]) == new_team), None)
        if old_contact is None or new_contact is None or new_contact <= transition:
            continue

        departure = next((frame for frame in range(new_contact, min(len(out),
                          new_contact + int(round(0.8 * fps)) + 1))
                          if np.isfinite(speed[frame])
                          and speed[frame] >= DELIBERATE_DEPARTURE_SPEED_MPS), None)
        # Larger shifts are too ambiguous: they often represent a genuine turnover followed by a
        # later pass, not a delayed confirmation of the same turnover.
        if (departure is None or departure <= transition
                or departure - transition >= max_departure_shift):
            continue
        out[transition:departure] = old_label
        events.append({
            "method": "deliberate_departure_boundary",
            "team": new_label,
            "old_transition": transition,
            "new_transition": departure,
            "confirming_contact": new_contact,
        })

    # Recompute after the delay pass.  Future same-team action can confirm an earlier control
    # trough even when the supplied ball coordinate is several metres away from every player.
    transitions = [frame for frame in range(1, len(out))
                   if out[frame] in ("H", "A") and out[frame - 1] in ("H", "A")
                   and out[frame] != out[frame - 1]]
    for transition in transitions:
        old_label, new_label = out[transition - 1], out[transition]
        new_team = HOME if new_label == "H" else AWAY
        new_distance = nearest_home if new_team == HOME else nearest_away
        old_distance = nearest_away if new_team == HOME else nearest_home
        confirming_contact = next((frame for frame in range(transition,
                                  min(len(out), transition + lookahead + 1))
                                  if int(contacts[frame]) == new_team), None)
        if confirming_contact is None:
            continue

        onset = None
        for frame in range(max(0, transition - lookback), transition):
            if (not np.isfinite(speed[frame])
                    or speed[frame] > CONTROL_TROUGH_SPEED_MPS
                    or not np.isfinite(new_distance[frame])
                    or not np.isfinite(old_distance[frame])):
                continue
            if new_distance[frame] - old_distance[frame] < CONTROL_PARITY_MARGIN_M:
                onset = frame
                break
        if onset is None or np.any(out[onset:transition] == "D"):
            continue
        out[onset:transition] = new_label
        events.append({
            "method": "future_control_confirmation",
            "team": new_label,
            "old_transition": transition,
            "new_transition": onset,
            "confirming_contact": confirming_contact,
        })
    return out, events


def collapse_unconfirmed_sandwich_spells(labels, contacts, fps, support_events=None):
    """Remove a brief proximity-only possession surrounded by the same opponent.

    A pass in flight, deflection or contested touch often creates a short A-B-A label spell even
    though B never controls the ball.  The spell is retained when a stronger sequence repair has
    independently confirmed it (for example, reception followed by a pass).  Otherwise, at most
    two isolated contact candidates are insufficient to overturn the surrounding possession.
    """
    out = np.array(labels, copy=True)
    limit = max(1, int(round(UNCONFIRMED_SPELL_MAX_SECONDS * fps)))
    support_events = support_events or []
    events = []
    frame = 1
    while frame < len(out) - 1:
        start = frame
        label = out[frame]
        while frame < len(out) and out[frame] == label:
            frame += 1
        stop = frame
        if (label not in ("H", "A") or stop - start > limit or start <= 0
                or stop >= len(out) or out[start - 1] != out[stop]
                or out[stop] not in ("H", "A")):
            continue

        team = HOME if label == "H" else AWAY
        contact_count = int(np.sum(contacts[start:stop] == team))
        if contact_count > UNCONFIRMED_SPELL_MAX_CONTACTS:
            continue

        supported = False
        for event in support_events:
            if event.get("team") != label:
                continue
            event_start = event.get("onset", event.get("live", event.get("new_transition")))
            event_stop = event.get(
                "opponent_reception",
                event.get("confirmation_start", event.get("confirming_contact")),
            )
            if event_start is None or event_stop is None:
                continue
            if int(event_start) < stop and int(event_stop) >= start:
                supported = True
                break
        if supported:
            continue

        surrounding = out[start - 1]
        out[start:stop] = surrounding
        events.append({
            "method": "unconfirmed_sandwich_spell",
            "removed_team": label,
            "restored_team": surrounding,
            "start": start,
            "stop": stop,
            "contact_count": contact_count,
        })
    return out, events


def future_confirmed_interception_repairs(labels, contacts, nearest_home,
                                          nearest_away, fps):
    """Backdate a late turnover when two same-team actions confirm the interception.

    In a crowded tackle the nearest-team signal may flicker and delay ownership. A first contact
    followed by another same-team action, with no intervening opponent contact, confirms that the
    turnover began at the interception rather than at the later debounce boundary.
    """
    out = np.array(labels, copy=True)
    events = []
    before = max(1, int(round(INTERCEPTION_CONFIRM_LOOKBACK_SECONDS * fps)))
    after = max(1, int(round(INTERCEPTION_CONFIRM_LOOKAHEAD_SECONDS * fps)))
    separation = max(1, int(round(INTERCEPTION_MIN_CONTACT_SEPARATION_SECONDS * fps)))
    backfill = max(0, int(round(INTERCEPTION_CONTACT_BACKFILL_SECONDS * fps)))
    transitions = [frame for frame in range(1, len(out))
                   if out[frame] in ("H", "A") and out[frame - 1] in ("H", "A")
                   and out[frame] != out[frame - 1]]
    for transition in transitions:
        old_label, new_label = out[transition - 1], out[transition]
        old_team = HOME if old_label == "H" else AWAY
        new_team = HOME if new_label == "H" else AWAY
        candidates = [frame for frame in range(max(0, transition - before),
                      min(len(out), transition + after + 1))
                      if int(contacts[frame]) == new_team]
        first = next((frame for frame in candidates if frame < transition), None)
        if first is None:
            continue
        new_distance = nearest_home if new_team == HOME else nearest_away
        if (not np.isfinite(new_distance[first])
                or new_distance[first] > INTERCEPTION_FIRST_CONTACT_DISTANCE_M):
            continue
        second = next((frame for frame in candidates if frame - first >= separation), None)
        if second is None:
            continue
        if np.any(contacts[first:second + 1] == old_team):
            continue
        onset = max(0, first - backfill)
        if np.any(out[onset:transition] == "D"):
            continue
        out[onset:transition] = new_label
        events.append({
            "method": "future_confirmed_interception",
            "team": new_label,
            "onset": onset,
            "first_contact": first,
            "confirming_contact": second,
            "old_transition": transition,
        })
    return out, events


def enforce_match_envelope(labels, first_half_start, first_half_end,
                           second_half_kickoff, second_half_end):
    """Force pre-match, halftime and post-match frames to dead ball."""
    out = np.array(labels, copy=True)
    out[:max(0, first_half_start)] = "D"
    out[max(0, first_half_end + 1):min(len(out), second_half_kickoff)] = "D"
    out[min(len(out), second_half_end + 1):] = "D"
    return out


def confirmed_pass_chain_repairs(labels, nearest_team, nearest_home, nearest_away,
                                 contacts, speed, fps):
    """Recover a short possession that nearest-player debounce discarded.

    A later high-speed touch/pass can confirm that a team had already received and controlled the
    ball. We look back to the preceding low-speed control trough and acceleration, then keep that
    team through its pass until a slow, close opponent reception. This is offline confirmation:
    the later pass is evidence for the earlier possession boundary, not the boundary itself.
    """
    out = np.array(labels, copy=True)
    events = []
    lookback = max(1, int(round(PASS_LOOKBACK_SECONDS * fps)))
    lookahead = max(1, int(round(PASS_RECEPTION_LOOKAHEAD_SECONDS * fps)))

    for contact in range(len(out)):
        team = int(contacts[contact])
        if team == NOBODY or not np.isfinite(speed[contact]):
            continue
        if speed[contact] < PASS_CONTACT_SPEED_MPS:
            continue
        team_label = TEAM_LABEL[team]
        if out[contact] in ("D", team_label):
            continue

        first = max(0, contact - lookback)
        history = speed[first:contact + 1]
        if not np.isfinite(history).any():
            continue
        trough = first + int(np.nanargmin(history))
        onset = None
        for frame in range(trough, contact + 1):
            if (np.isfinite(speed[frame])
                    and speed[frame] >= CONTROL_DEPARTURE_SPEED_MPS
                    and int(nearest_team[frame]) == team):
                onset = frame
                break
        if onset is None or onset >= contact:
            continue

        opponent = HOME if team == AWAY else AWAY
        opponent_distance = nearest_home if opponent == HOME else nearest_away
        reception = None
        for frame in range(contact + 1, min(len(out), contact + lookahead + 1)):
            if (np.isfinite(speed[frame])
                    and speed[frame] <= CONTROL_RECEPTION_SPEED_MPS
                    and int(nearest_team[frame]) == opponent
                    and np.isfinite(opponent_distance[frame])
                    and opponent_distance[frame] <= CONTROL_RECEPTION_DISTANCE_M):
                reception = frame
                break
        if reception is None or np.any(out[onset:reception] == "D"):
            continue

        out[onset:reception] = team_label
        events.append({
            "method": "confirmed_pass_chain",
            "team": team_label,
            "onset": onset,
            "confirming_contact": contact,
            "opponent_reception": reception,
        })
    return out, events


def _normalise_team_name(value):
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())


def _tag_team_side(value, club_names):
    """Map a Smart Tagging club string to H/A, tolerating small spelling differences."""
    import difflib
    wanted = _normalise_team_name(value)
    if not wanted or wanted == "na":
        return NOBODY
    scores = [difflib.SequenceMatcher(None, wanted, _normalise_team_name(name)).ratio()
              for name in club_names]
    side = int(np.argmax(scores))
    return side if scores[side] >= 0.50 else NOBODY


def kickoff_long_ball_repairs(labels, smart_tagging_path, club_names, nearest_team,
                              contacts, speed, fps):
    """Keep kickoff ownership through an overlapping long ball and backdate its reception.

    Smart Tagging supplies coarse *context*, never the answer. Tracking supplies the first moving
    interaction, physical deceleration supplies the long-ball arrival, and a later sustained
    opponent run confirms who received it. No organizer labels or fixed match timestamps enter.
    """
    out = np.array(labels, copy=True)
    events = []
    parsed = []
    for instance in ET.parse(smart_tagging_path).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        parsed.append({
            "code": (instance.findtext("code") or "").strip().lower(),
            "start": int(round(start_s * fps)),
            "end": int(round(end_s * fps)),
            "tags": {label.findtext("group"): label.findtext("text")
                     for label in instance.findall("label")},
        })
    kickoffs = [row for row in parsed if row["code"] == "set pieces"
                and "kick off" in (row["tags"].get("Type") or "").lower()]
    long_balls = [row for row in parsed if row["code"] == "long ball"]
    confirm = max(1, int(round(LONG_BALL_CONFIRM_SECONDS * fps)))
    backfill = max(0, int(round(KICKOFF_BACKFILL_SECONDS * fps)))

    for kickoff in kickoffs:
        overlaps = [row for row in long_balls
                    if row["start"] <= kickoff["end"] and row["end"] >= kickoff["start"]]
        for long_ball in overlaps:
            team = _tag_team_side(long_ball["tags"].get("Team"), club_names)
            if team == NOBODY:
                continue
            opponent = HOME if team == AWAY else AWAY
            live = None
            for frame in range(max(0, kickoff["start"]),
                               min(len(out), kickoff["end"] + fps + 1)):
                if int(contacts[frame]) == team:
                    live = frame
                    break
            if live is None:
                continue
            for _ in range(backfill):
                if live <= 0 or int(nearest_team[live - 1]) != NOBODY:
                    break
                live -= 1

            search_start = max(live, long_ball["end"]
                               - int(round(LONG_BALL_LANDING_LOOKBACK_SECONDS * fps)))
            search_stop = min(len(out), long_ball["end"]
                              + int(round(LONG_BALL_LANDING_LOOKAHEAD_SECONDS * fps)) + 1)
            landing = None
            pre_window = max(1, int(round(0.5 * fps)))
            post_window = max(1, int(round(0.2 * fps)))
            for frame in range(search_start, search_stop):
                pre = speed[max(live, frame - pre_window):frame]
                post = speed[frame:min(len(speed), frame + post_window + 1)]
                if (not np.isfinite(speed[frame]) or not np.isfinite(pre).any()
                        or not np.isfinite(post).any()):
                    continue
                if (speed[frame] <= LONG_BALL_LANDING_SPEED_MPS
                        and np.nanmax(pre) >= LONG_BALL_PRE_SPEED_MPS
                        and np.nanmedian(post) <= LONG_BALL_LANDING_SPEED_MPS):
                    landing = frame
                    break
            if landing is None:
                continue

            confirmation_start = None
            limit = min(len(out), long_ball["end"]
                        + int(round(LONG_BALL_CONFIRM_LOOKAHEAD_SECONDS * fps)))
            frame = landing + 1
            while frame < limit:
                if out[frame] != TEAM_LABEL[opponent]:
                    frame += 1
                    continue
                run_start = frame
                while frame < limit and out[frame] == TEAM_LABEL[opponent]:
                    frame += 1
                if frame - run_start >= confirm:
                    confirmation_start = run_start
                    break
            if confirmation_start is None:
                continue

            out[live:landing + 1] = TEAM_LABEL[team]
            out[landing + 1:confirmation_start] = TEAM_LABEL[opponent]
            events.append({
                "method": "kickoff_long_ball",
                "team": TEAM_LABEL[team],
                "live": live,
                "landing": landing,
                "receiver": TEAM_LABEL[opponent],
                "confirmation_start": confirmation_start,
            })
    return out, events


def possession_chain_postprocess(labels, nearest_team, nearest_home, nearest_away,
                                 contacts, speed, fps, smart_tagging_path=None,
                                 club_names=None, dead_intervals=None):
    """Apply conservative offline sequence repairs after dead-ball inference."""
    out = np.array(labels, copy=True)
    events = []
    smart_context = None
    if smart_tagging_path and club_names:
        smart_context = smart_tagging_live_context(
            smart_tagging_path, club_names, len(out), fps
        )
        out, kickoff_events = kickoff_long_ball_repairs(
            out, smart_tagging_path, club_names, nearest_team, contacts, speed, fps
        )
        events.extend(kickoff_events)
    out, turnover_events = controlled_turnover_boundary_repairs(
        out, nearest_team, nearest_home, nearest_away, contacts, speed, fps
    )
    events.extend(turnover_events)
    out, pass_events = confirmed_pass_chain_repairs(
        out, nearest_team, nearest_home, nearest_away, contacts, speed, fps
    )
    events.extend(pass_events)
    out, missing_events = recover_untracked_live_play(
        out, nearest_team, contacts, fps, dead_intervals=dead_intervals,
        smart_context=smart_context,
    )
    events.extend(missing_events)
    out, sandwich_events = collapse_unconfirmed_sandwich_spells(
        out, contacts, fps, support_events=events
    )
    events.extend(sandwich_events)
    out, interception_events = future_confirmed_interception_repairs(
        out, contacts, nearest_home, nearest_away, fps
    )
    events.extend(interception_events)
    out = bridge_short_missing_gaps(out, fps)
    return out, events


def write_diagnostic(path, start, end, fps, labels, nearest_team, nearest_home,
                     nearest_away, contacts, speed, turn):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "frame", "video_time", "sequence_label", "nearest_team",
            "nearest_home_m", "nearest_away_m", "probable_contact_team",
            "planar_ball_speed_mps", "turn_degrees",
        ])
        for frame in range(max(0, start), min(len(labels), end + 1)):
            minute, rem = divmod(frame, fps * 60)
            second, subframe = divmod(rem, fps)
            writer.writerow([
                frame,
                f"{minute:02d}:{second:02d}.{round(1000 * subframe / fps):03d}",
                labels[frame],
                TEAM_LABEL[int(nearest_team[frame])],
                "" if np.isnan(nearest_home[frame]) else f"{nearest_home[frame]:.3f}",
                "" if np.isnan(nearest_away[frame]) else f"{nearest_away[frame]:.3f}",
                "" if contacts[frame] == NOBODY else TEAM_LABEL[int(contacts[frame])],
                "" if np.isnan(speed[frame]) else f"{speed[frame]:.3f}",
                "" if np.isnan(turn[frame]) else f"{turn[frame]:.3f}",
            ])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.environ.get("MTAH_DATA_DIR"))
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results",
                                                       "sequence_memory_v1.csv"))
    parser.add_argument("--diagnostic", default=os.path.join(ROOT, "work",
                                                              "sequence_target_diagnostic.csv"))
    parser.add_argument("--diagnostic-start", type=int, default=12620)
    parser.add_argument("--diagnostic-end", type=int, default=12680)
    parser.add_argument("--confirm-seconds", type=float, default=TURNOVER_CONFIRM_SECONDS)
    args = parser.parse_args()

    fps = GAMES[GAME]["fps"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"], data_dir=args.data_dir)
    n_frames = source_frames("tactical")
    evidence = load_tracking_evidence(
        raw_path, team_of_each_track(meta_path), n_frames, GAMES[GAME]["pitch"]
    )
    ball_xy, nearest_team, nearest_home, nearest_away = evidence
    speed, turn = planar_motion(ball_xy, fps)
    contacts = probable_contacts(nearest_home, nearest_away)
    labels = possession_memory(nearest_team, fps, args.confirm_seconds)

    write_submission(args.out, labels)
    write_diagnostic(
        args.diagnostic, args.diagnostic_start, args.diagnostic_end, fps, labels,
        nearest_team, nearest_home, nearest_away, contacts, speed, turn,
    )
    counts = {label: int(np.sum(labels == label)) for label in ("H", "A", "D")}
    print(f"wrote submission: {args.out}")
    print(f"wrote diagnostic: {args.diagnostic}")
    print(f"turnover confirmation: {args.confirm_seconds:.2f}s")
    print(f"counts: {counts}")


if __name__ == "__main__":
    main()
