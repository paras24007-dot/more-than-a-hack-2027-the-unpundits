#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dead_ball_experiment import (  # noqa: E402
    boundary_restart_receiving_phase, contact_restart_candidates,
    first_valid_after_missing,
)
from dead_ball_context_experiment import (  # noqa: E402
    boundary_side, exit_side,
)
from dead_ball_multievidence_experiment import sequence_stop_evidence  # noqa: E402
from sequence_possession_experiment import AWAY, HOME, NOBODY  # noqa: E402


class DeadBallExperimentTest(unittest.TestCase):
    def test_sequence_evidence_accepts_in_field_stoppage(self):
        fps = 25
        n = 250
        frame = 75
        activity = np.full(n, 2.5)
        activity[frame + 10:frame + 100] = 1.0
        nearest_team = np.full(n, HOME, dtype=np.int8)
        nearest_home = np.full(n, 4.0)
        nearest_away = np.full(n, 5.0)
        evidence = sequence_stop_evidence(
            frame, nearest_team, nearest_home, nearest_away, activity, fps
        )
        self.assertTrue(evidence["activity_drop"])
        self.assertTrue(evidence["stopped_level"])
        self.assertTrue(evidence["noncompetitive"])

    def test_sequence_evidence_rejects_continuous_competitive_play(self):
        fps = 25
        n = 250
        frame = 75
        activity = np.full(n, 2.2)
        nearest_team = np.full(n, HOME, dtype=np.int8)
        nearest_home = np.full(n, 0.8)
        nearest_away = np.full(n, 1.2)
        evidence = sequence_stop_evidence(
            frame, nearest_team, nearest_home, nearest_away, activity, fps
        )
        self.assertFalse(evidence["activity_drop"])
        self.assertFalse(evidence["noncompetitive"])

    def test_boundary_helpers_preserve_crossing_side(self):
        pitch = (105.3, 68.0)
        side, distance = boundary_side(np.array([1.8, 55.0]), pitch)
        self.assertEqual(side, "x_min")
        self.assertAlmostEqual(distance, 1.8)
        points = np.array([[1.8, 55.0], [-0.2, 30.0]])
        self.assertEqual(exit_side(1, points, pitch), "x_min")

    def test_missing_tracking_requires_confirmation_after_it_returns(self):
        nearest = np.array([HOME] * 5 + [NOBODY] * 10 + [AWAY] * 20, dtype=np.int8)
        release = first_valid_after_missing(5, nearest, fps=25)
        self.assertEqual(release, 25)  # first valid 15 + ten confirmation frames

    def test_transport_touch_fails_but_later_restart_survives_lookahead(self):
        n = 220
        contacts = np.full(n, NOBODY, dtype=np.int8)
        contacts[30] = HOME       # administrative ball movement
        contacts[100] = AWAY     # actual restart candidate
        speed = np.ones(n, dtype=float)
        speed[30:50] = 5.0
        speed[100:120] = 5.0
        ball_xy = np.column_stack((np.arange(n, dtype=float) * 0.1, np.zeros(n)))
        activity = np.ones(n, dtype=float)
        activity[150:201] = 2.0  # sustained team-wide response only after second candidate
        nearest_home = np.full(n, 2.0)
        nearest_away = np.full(n, 2.0)

        candidates = list(contact_restart_candidates(
            10, 180, contacts, speed, ball_xy, activity,
            nearest_home, nearest_away, fps=25,
        ))

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["contact"], 100)
        self.assertEqual(candidates[0]["team"], AWAY)

    def test_boundary_restart_releases_in_receiving_phase_not_at_kick(self):
        n = 100
        fps = 25
        stop = 5
        departure = 30
        reception = 56
        contacts = np.full(n, NOBODY, dtype=np.int8)
        contacts[reception] = AWAY
        nearest_team = np.full(n, NOBODY, dtype=np.int8)
        nearest_team[departure:] = AWAY
        nearest_home = np.full(n, 12.0)
        nearest_away = np.full(n, 0.2)
        # After the kick, the nearest away-player distance first grows and then falls as a
        # teammate receives the pass.  The receiving phase begins before the final contact.
        nearest_away[departure:reception + 1] = np.concatenate((
            np.linspace(0.2, 4.0, 11),
            np.linspace(3.8, 0.2, reception - departure - 10),
        ))
        ball_xy = np.column_stack((
            np.linspace(0.2, 20.0, n),
            np.full(n, 67.0),
        ))
        speed = np.full(n, 6.0)
        confirmed = {"contact": reception, "release": 70, "team": AWAY}

        result = boundary_restart_receiving_phase(
            stop, confirmed, "y_max", (105.3, 68.0), nearest_team,
            contacts, speed, ball_xy, nearest_home, nearest_away, fps,
        )

        self.assertIsNotNone(result)
        self.assertGreater(result["release"], departure)
        self.assertLess(result["release"], reception)


if __name__ == "__main__":
    unittest.main()
