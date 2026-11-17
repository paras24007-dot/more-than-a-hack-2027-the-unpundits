#!/usr/bin/env python3
"""Experimental referee-whistle detector for tactical-video audio.

This module scans audio without reading possession ground truth. It intentionally produces
*whistle candidates*, not final dead-ball labels: stadium audio can contain chants, PA tones,
and other high-frequency sounds. Candidate intervals must later be combined with tracking and
restart evidence.
"""

from __future__ import annotations

import argparse
import csv
import math
import wave

import numpy as np


def read_mono_wav(path):
    """Read 16-bit PCM WAV and return normalized mono samples plus sample rate."""
    with wave.open(path, "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM WAV")
        rate = handle.getframerate()
        channels = handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def spectral_features(samples, rate, window_seconds=0.08, hop_seconds=0.02,
                      whistle_low_hz=3000.0, whistle_high_hz=4500.0):
    """Measure high-frequency concentration and narrow-band tonality per short audio window."""
    size = max(64, int(round(window_seconds * rate)))
    hop = max(1, int(round(hop_seconds * rate)))
    if len(samples) < size:
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0)

    window = np.hanning(size)
    freqs = np.fft.rfftfreq(size, 1.0 / rate)
    whistle_band = (freqs >= whistle_low_hz) & (freqs <= whistle_high_hz)
    useful_band = (freqs >= 500.0) & (freqs <= min(7000.0, rate / 2.0))
    times, ratios, tonalities, peaks = [], [], [], []

    for start in range(0, len(samples) - size + 1, hop):
        frame = samples[start:start + size]
        power = np.abs(np.fft.rfft(frame * window)) ** 2
        total = float(power[useful_band].sum()) + 1e-15
        band = power[whistle_band]
        band_total = float(band.sum())
        times.append((start + size / 2) / rate)
        ratios.append(band_total / total)
        tonalities.append(float(band.max()) / (band_total + 1e-15) if band.size else 0.0)
        peaks.append(float(freqs[whistle_band][int(np.argmax(band))]) if band.size else math.nan)

    return np.asarray(times), np.asarray(ratios), np.asarray(tonalities), np.asarray(peaks)


def candidate_intervals(times, ratios, tonalities, ratio_threshold=0.08,
                        tonality_threshold=0.07, max_gap_seconds=0.10,
                        min_duration_seconds=0.12):
    """Join nearby whistle-like windows into candidate sound intervals."""
    active = (ratios >= ratio_threshold) & (tonalities >= tonality_threshold)
    active_times = times[active]
    if not len(active_times):
        return []

    groups = []
    start = previous = float(active_times[0])
    for timestamp in active_times[1:]:
        timestamp = float(timestamp)
        if timestamp - previous > max_gap_seconds:
            groups.append((start, previous))
            start = timestamp
        previous = timestamp
    groups.append((start, previous))

    half_window = 0.04
    return [
        (max(0.0, start - half_window), end + half_window)
        for start, end in groups
        if end - start + 2 * half_window >= min_duration_seconds
    ]


def interval_statistics(intervals, times, ratios, tonalities, peaks):
    """Return spectral summaries used to reject weak stadium/PA candidates later."""
    summaries = []
    for start, end in intervals:
        mask = (times >= start) & (times <= end)
        summaries.append({
            "max_ratio": float(np.max(ratios[mask])),
            "mean_ratio": float(np.mean(ratios[mask])),
            "max_tonality": float(np.max(tonalities[mask])),
            "median_peak_hz": float(np.median(peaks[mask])),
            "peak_std_hz": float(np.std(peaks[mask])),
        })
    return summaries


def write_candidates(path, intervals, offset_seconds=0.0, summaries=None):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["candidate_start_seconds", "candidate_end_seconds", "duration_seconds"]
        if summaries is not None:
            header += ["max_ratio", "mean_ratio", "max_tonality", "median_peak_hz",
                       "peak_std_hz"]
        writer.writerow(header)
        for index, (start, end) in enumerate(intervals):
            row = [
                f"{start + offset_seconds:.3f}",
                f"{end + offset_seconds:.3f}",
                f"{end - start:.3f}",
            ]
            if summaries is not None:
                summary = summaries[index]
                row += [
                    f"{summary['max_ratio']:.6f}",
                    f"{summary['mean_ratio']:.6f}",
                    f"{summary['max_tonality']:.6f}",
                    f"{summary['median_peak_hz']:.3f}",
                    f"{summary['peak_std_hz']:.3f}",
                ]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wav")
    parser.add_argument("--out", default="whistle_candidates.csv")
    parser.add_argument("--offset-seconds", type=float, default=0.0)
    # Deliberately recall-oriented defaults. Contextual tracking/restart evidence will reject
    # stadium and PA false positives; missing a real whistle here would be unrecoverable later.
    parser.add_argument("--ratio-threshold", type=float, default=0.08)
    parser.add_argument("--tonality-threshold", type=float, default=0.07)
    args = parser.parse_args()

    samples, rate = read_mono_wav(args.wav)
    times, ratios, tonalities, peaks = spectral_features(samples, rate)
    intervals = candidate_intervals(
        times, ratios, tonalities,
        ratio_threshold=args.ratio_threshold,
        tonality_threshold=args.tonality_threshold,
    )
    summaries = interval_statistics(intervals, times, ratios, tonalities, peaks)
    write_candidates(args.out, intervals, args.offset_seconds, summaries)
    print(f"scanned {len(samples) / rate:.3f}s at {rate} Hz")
    print(f"wrote {len(intervals)} candidate intervals: {args.out}")


if __name__ == "__main__":
    main()
