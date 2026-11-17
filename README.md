# The Unpundits — More Than A Hack 2027

Reproducible Phase 1 submission for frame-level football possession inference.

The system produces one `H`, `A`, `D` or `X` label for each of the 199,319 frames of the supplied
BAR–ATH tactical video. It combines automatic player/ball tracking, Smart Tagging kickoff/long-ball
context, temporal possession memory, pitch-boundary events, referee-whistle candidates from
tactical audio, control-aware turnover timing, live-play recovery for short tracking dropouts,
long-ball ownership continuity, a protected match/halftime envelope, and a persistent dead-ball
state with future-confirmed restarts.

## Reproduce the submission

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python generate_submission.py \
  --data-dir "/path/to/More Than A Hack 2027 - Initial Challenge - Dataset"
```

If the tactical MP4 is stored separately, add
`--video "/path/to/BAR-ATH_tactical-002.mp4"`.

The data directory must contain:

```text
tactical/
  BAR-ATH_tactical.mp4
  BAR-ATH_tactical_atd.xml
  BAR-ATH_tactical_atd.txt
  BAR-ATH_tactical_smart_tagging.xml
```

The command writes `deliverables/The_Unpundits.csv` and validates it using the same format checks
as the official scorer. To choose another output location, add `--out /path/to/output.csv`.

The first run downloads the platform-specific FFmpeg executable supplied by `imageio-ffmpeg` if it
is not already cached. Match assets and generated frame-level files are excluded from version
control because the competition data may not be redistributed.

## Tests

```bash
python -m unittest discover -s work -p 'test_*.py'
```

## Method

See [The_Unpundits.md](The_Unpundits.md) or [The_Unpundits.pdf](The_Unpundits.pdf) for the full
technical description, implemented possession rules, limitations and reproduction details.

## Repository layout

- `generate_submission.py` — one-command end-to-end reproduction.
- `work/sequence_possession_experiment.py` — tracking evidence and temporal possession memory.
- `work/whistle_audio_experiment.py` — deterministic whistle-candidate extraction.
- `work/dead_ball_experiment.py` — stoppage latch and restart confirmation.
- `work/dead_ball_context_experiment.py` — boundary-assisted quiet-whistle evidence.
- `work/dead_ball_multievidence_experiment.py` — final multi-evidence pipeline.
- `evaluation/validate_submission.py` — official-compatible submission validation.

## Data and licence

No match video, tracking, homography, Smart Tagging, public labels or generated match annotations
are committed here. The supplied match remains subject to the competition's data licence. Code
derived from the organizer starter repository retains its MIT licence; see `LICENSE`.

