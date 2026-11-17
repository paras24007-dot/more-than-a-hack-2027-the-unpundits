#!/usr/bin/env python3
"""Small synthetic checks for the possession-memory experiment."""

import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sequence_possession_experiment import (  # noqa: E402
    AWAY, HOME, NOBODY, bridge_short_missing_gaps,
    collapse_unconfirmed_sandwich_spells, confirmed_pass_chain_repairs,
    controlled_turnover_boundary_repairs,
    enforce_match_envelope, future_confirmed_interception_repairs,
    possession_memory, recover_untracked_live_play,
)


class PossessionMemoryTest(unittest.TestCase):
    def test_brief_opponent_run_does_not_change_possession(self):
        nearest = np.array([HOME] * 10 + [AWAY] * 8 + [HOME] * 10, dtype=np.int8)
        labels = possession_memory(nearest, fps=25, confirm_seconds=0.8)
        self.assertEqual(set(labels), {"H"})

    def test_sustained_opponent_run_changes_only_after_confirmation(self):
        nearest = np.array([HOME] * 5 + [AWAY] * 25, dtype=np.int8)
        labels = possession_memory(nearest, fps=25, confirm_seconds=0.8)
        self.assertTrue(np.all(labels[:24] == "H"))
        self.assertTrue(np.all(labels[24:] == "A"))

    def test_recovery_cancels_turnover_candidate(self):
        nearest = np.array(
            [HOME] * 5 + [AWAY] * 12 + [HOME] * 4 + [AWAY] * 7 + [HOME] * 5,
            dtype=np.int8,
        )
        labels = possession_memory(nearest, fps=25, confirm_seconds=0.8)
        self.assertEqual(set(labels), {"H"})

    def test_missing_ball_is_dead_in_first_experiment(self):
        nearest = np.array([HOME, HOME, NOBODY, AWAY], dtype=np.int8)
        labels = possession_memory(nearest, fps=25, confirm_seconds=0.8)
        self.assertEqual(labels.tolist(), ["H", "H", "D", "A"])

    def test_short_missing_gap_is_bridged_only_for_same_team(self):
        labels = np.array(list("HHDDDHHHAAADDDHH"), dtype="<U1")
        repaired = bridge_short_missing_gaps(labels, fps=10, max_seconds=0.3)
        self.assertEqual("".join(repaired[:8]), "HHHHHHHH")
        self.assertEqual("".join(repaired[11:]), "DDDHH")

    def test_later_pass_backdates_discarded_short_possession(self):
        n = 100
        labels = np.full(n, "H", dtype="<U1")
        nearest = np.zeros(n, dtype=np.int8)
        nearest[35:51] = AWAY
        home = np.full(n, 3.0)
        away = np.full(n, 3.0)
        contacts = np.full(n, NOBODY, dtype=np.int8)
        contacts[50] = AWAY
        speed = np.full(n, 4.0)
        speed[30] = 2.0
        speed[35:51] = np.linspace(7.0, 10.0, 16)
        speed[80] = 0.8
        home[80] = 1.0
        repaired, events = confirmed_pass_chain_repairs(
            labels, nearest, home, away, contacts, speed, fps=25
        )
        self.assertTrue(np.all(repaired[35:80] == "A"))
        self.assertEqual(repaired[80], "H")
        self.assertEqual(len(events), 1)

    def test_missing_tracking_keeps_last_confirmed_owner_but_not_through_stop(self):
        labels = np.array(list("AAHHDDDDHHHHHHHH"), dtype="<U1")
        nearest = np.array(
            [AWAY, AWAY, HOME, HOME, NOBODY, NOBODY, NOBODY, NOBODY]
            + [HOME] * 8,
            dtype=np.int8,
        )
        contacts = np.full(len(labels), NOBODY, dtype=np.int8)
        contacts[1] = AWAY
        repaired, _ = recover_untracked_live_play(
            labels, nearest, contacts, fps=10, dead_intervals=[]
        )
        self.assertEqual("".join(repaired), "AAAAAAAAAAAAAHHH")

        protected, _ = recover_untracked_live_play(
            labels, nearest, contacts, fps=10,
            dead_intervals=[{"stop": 5, "release": 8}],
        )
        self.assertEqual("".join(protected[5:8]), "DDD")

    def test_turnover_waits_for_new_team_departure(self):
        n = 60
        labels = np.array(list("H" * 15 + "A" * 45), dtype="<U1")
        nearest = np.array([HOME] * 10 + [AWAY] * 50, dtype=np.int8)
        home = np.full(n, 3.0)
        away = np.full(n, 3.0)
        contacts = np.full(n, NOBODY, dtype=np.int8)
        contacts[8] = HOME
        contacts[20] = AWAY
        speed = np.full(n, 4.0)
        speed[20:26] = [2.0, 3.0, 5.0, 7.0, 9.0, 11.0]
        repaired, _ = controlled_turnover_boundary_repairs(
            labels, nearest, home, away, contacts, speed, fps=25
        )
        self.assertTrue(np.all(repaired[:25] == "H"))
        self.assertTrue(np.all(repaired[25:] == "A"))

    def test_unconfirmed_short_spell_is_removed_but_supported_spell_survives(self):
        labels = np.array(list("H" * 10 + "A" * 20 + "H" * 10), dtype="<U1")
        contacts = np.full(len(labels), NOBODY, dtype=np.int8)
        contacts[15] = AWAY
        repaired, _ = collapse_unconfirmed_sandwich_spells(
            labels, contacts, fps=10
        )
        self.assertEqual(set(repaired), {"H"})

        supported, _ = collapse_unconfirmed_sandwich_spells(
            labels, contacts, fps=10,
            support_events=[{
                "team": "A", "onset": 10, "confirming_contact": 15,
                "opponent_reception": 30,
            }],
        )
        self.assertTrue(np.all(supported[10:30] == "A"))

    def test_match_envelope_keeps_pregame_halftime_and_postmatch_dead(self):
        labels = np.full(30, "H", dtype="<U1")
        repaired = enforce_match_envelope(
            labels, first_half_start=5, first_half_end=12,
            second_half_kickoff=18, second_half_end=25,
        )
        self.assertTrue(np.all(repaired[:5] == "D"))
        self.assertTrue(np.all(repaired[5:13] == "H"))
        self.assertTrue(np.all(repaired[13:18] == "D"))
        self.assertTrue(np.all(repaired[18:26] == "H"))
        self.assertTrue(np.all(repaired[26:] == "D"))

    def test_future_same_team_action_backdates_interception(self):
        labels = np.array(list("H" * 50 + "A" * 20), dtype="<U1")
        contacts = np.full(len(labels), NOBODY, dtype=np.int8)
        contacts[20] = AWAY
        contacts[45] = AWAY
        home = np.full(len(labels), 3.0)
        away = np.full(len(labels), 3.0)
        away[20] = 0.2
        repaired, events = future_confirmed_interception_repairs(
            labels, contacts, home, away, fps=25
        )
        self.assertTrue(np.all(repaired[18:] == "A"))
        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
