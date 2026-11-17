# The Unpundits — Technical Document

## 1. Summary

Our system assigns `H` (home possession), `A` (away possession) or `D` (dead ball) to every frame
of the tactical video. The design treats possession as a temporal state rather than an isolated
nearest-player decision. It also treats dead ball as a persistent state: once a stoppage is
confirmed, ordinary ball movement cannot unlock it until a restart survives future checks.

The final pipeline is deterministic and requires no trained model. It uses the supplied tactical
tracking, automatic Smart Tagging and the tactical video's audio. The repository contains one
command that extracts audio, generates whistle candidates, processes the supplied signals and
writes the complete 199,319-row submission.

## 2. Assets used

### Used

- **Tactical ATD tracking:** two-dimensional ball and player positions. Player-to-ball distances,
  pitch exits, planar ball motion and player activity are derived from it.
- **Tactical video audio:** converted to 16 kHz mono PCM and scanned for narrow-band,
  high-frequency whistle candidates.
- **Organizer metadata:** team assignment for anonymous tracking slots, pitch dimensions, frame
  rate and frame count.
- **Smart Tagging:** kickoff and overlapping long-ball intervals are used as coarse context. They
  never become possession labels directly; tracking motion and future control must confirm them.

### Investigated but not used in the final system

- **Homography:** useful for debugging tracking/video disagreement, but not required by the final
  inference path.
- **Broadcast video:** excluded because its timeline is not synchronized to the tactical source.

## 3. Possession inference

For each observed ball frame, the system calculates the nearest tracked home and away players.
This instantaneous signal is deliberately separated from confirmed possession.

The confirmed team is retained while the ball travels or becomes temporarily nearer to an
opponent. An opposing nearest-team run must persist for 0.8 seconds before ownership changes. This
reduces false turnovers during passes, deflections and short loose-ball phases. Probable contacts
are inferred from a local player-ball distance minimum within a 1.5-metre radius; they are treated
as evidence rather than literal touch annotations because the supplied tracking has no touch flag
or ball-height coordinate.

Seven offline sequence repairs handle cases where local proximity is misleading. First, a kickoff
that overlaps an automatic long-ball tag retains the kicking team while the ball is travelling;
physical deceleration identifies the arrival and later sustained control confirms and backdates
the receiver. Second, a later high-speed pass can confirm that a short possession had already
started at the preceding control/acceleration phase. The state is then retained through the pass
until a slow, close opponent reception. Third, live turnovers are placed at a credible action:
a close touch followed by deliberate ball departure, or a low-speed control phase later confirmed
by a same-team action. This prevents a nearby presser from taking possession merely by remaining
closest for several frames.

Fourth, a missing automatic ball coordinate means unknown rather than dead. A dropout of at most
12 seconds may retain the team from the last credible interaction when no independently confirmed
whistle or boundary interval overlaps it. Automatic progression/attacking-transition tags are
used only as supporting team context while the coordinate is absent, never to overwrite observed
tracking. When tracking returns near the opponent, a 0.60-second confirmation guard prevents an
instantaneous false turnover. Fifth, a tiny gap of at most 0.20 seconds is bridged directly when
the same observed team appears on both sides. These rules use no fixed match timestamps.

Sixth, a brief opponent spell surrounded by the same team is removed when it contains at most two
isolated contact candidates and no stronger sequence evidence confirms control. This preserves
ownership through long passes, deflections and contested touches. A genuine short possession is
retained when a reception/pass-chain repair independently supports it.

Seventh, an interception can be confirmed retrospectively when an opponent makes a genuinely close
first contact (within 0.5 metres), that team produces a second action shortly afterwards, and the
old team has no intervening contact. Once confirmed, the change is backdated by 0.08 seconds to the
original interception phase instead of waiting for the later pass that proves control.

After every inference rule, the known recording envelope is enforced: pre-match, the complete
halftime interval through the exact second-half kickoff, and post-match are always `D`.

## 4. Dead-ball inference

Dead-ball detection combines independent evidence rather than requiring every stoppage to have the
same cause.

### Strong whistles

Audio is analysed in 80 ms windows with a 20 ms hop. Energy concentration between 3 and 4.5 kHz
and narrow-band tonality create whistle candidates. Strong candidates become direct stoppage
events.

### Quiet whistles

A quiet candidate creates a possible stop. It is accepted only when later evidence confirms a
sequence break, such as reduced collective player activity, non-competitive ball behaviour and a
future valid restart. Boundary-assisted quiet whistles are an independent branch for corners,
goal kicks and throw-ins; an in-field foul never requires a boundary crossing.

### Pitch exits

A sustained tracked crossing of a touchline or goal line creates a boundary stoppage candidate.
This branch is independent of whistle loudness.

### Persistent stopped state

Once a stop is confirmed, the state remains `D`. Momentum after the whistle, a goalkeeper passing
the ball back, or players transporting the ball toward a free-kick location cannot release it.
Each possible restart is checked using future frames for ball departure, displacement and broader
resumption of play. This offline confirmation prevents an administrative touch from becoming a
false restart.

For boundary set pieces, a later same-team reception can confirm that an earlier kick was the real
restart. The later touch is confirmation, not automatically the possession boundary. After the
sequence is confirmed, the system changes from `D` when the receiver is materially closing into
the ball-control phase. This avoids both releasing at the first unconfirmed kick and waiting until
a later second pass.

## 5. Training data

No trained model and no labelled training dataset are used. No human labels from the provided
match are embedded in the submission, and no match-specific timestamps are present in the code.
The public five-minute reference was used only for organizer-provided pipeline verification and
error analysis.

## 6. Public verification

On the organizer's 7,500 labelled public frames, the final version obtains:

- accuracy: 0.9881;
- macro F1: 0.9889;
- alive/dead F1: 0.9984;
- dead-ball F1: 0.9935;
- spell F1: 0.8000.

These public numbers are diagnostic only; the final ranking uses the private full-match reference.
The generated CSV passes the organizer-compatible validator with exactly 199,319 ordered rows and
no invalid or blank labels.

## 7. Limitations

- The supplied ball track is absent for long periods. The system repairs only short, independently
  live-looking gaps and remains conservative for absences longer than 12 seconds.
- Two-dimensional tracking cannot reliably distinguish an aerial ball, a deflection and controlled
  contact in every case.
- Possession remains more fragmented than the public reference because some extended challenges
  and occlusions provide neither reliable proximity nor a later interaction strong enough for
  offline confirmation.
- Audio whistle detection is affected by crowd noise, public-address tones and quiet referee
  whistles.
- The system does not use a learned visual ball or referee detector; those would be the main next
  improvements.

## 8. Reproduction

Install the pinned dependencies and run:

```bash
python generate_submission.py --data-dir "/path/to/dataset"
```

This extracts tactical audio, generates both strong and recall-oriented whistle candidates, runs
the complete tracking/state pipeline, writes `deliverables/The_Unpundits.csv`, and invokes the
submission validator. Tests can be run with:

```bash
python -m unittest discover -s work -p 'test_*.py'
```

