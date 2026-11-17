#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from whistle_audio_experiment import candidate_intervals, spectral_features  # noqa: E402


class WhistleAudioExperimentTest(unittest.TestCase):
    def test_tone_is_detected_but_broad_noise_is_not(self):
        rate = 16000
        rng = np.random.default_rng(7)
        time = np.arange(rate * 2) / rate
        samples = rng.normal(0, 0.015, len(time))
        whistle = (time >= 0.6) & (time < 1.2)
        samples[whistle] += 0.35 * np.sin(2 * np.pi * 3800 * time[whistle])

        times, ratios, tonalities, _ = spectral_features(samples, rate)
        intervals = candidate_intervals(times, ratios, tonalities)

        self.assertEqual(len(intervals), 1)
        self.assertLess(intervals[0][0], 0.7)
        self.assertGreater(intervals[0][1], 1.1)


if __name__ == "__main__":
    unittest.main()
