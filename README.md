# pitch-modeling

Per-pitch **Stuff**, **Pitching** and **Location** run values for MLB pitches, from a set of
double/debiased machine learning (DML) pitch models trained on 2023–25 Statcast data.

* **Stuff**: what the pitch's physical characteristics are worth, independent of where it
  was thrown and in what count.
* **Pitching**: what the pitch is worth at its actual location and count.
* **Location**: the difference between the two, i.e. what the location added to (or took away
  from) the stuff.

Everything runs from one standalone script, `score_pitches.py` (numpy, pandas, lightgbm).

## Install

```bash
pip install -r requirements.txt
```

`numba` is optional. With it, a full season (~700k pitches) scores in about a minute. Without
it, a float32 numpy fallback is used, spread across all cores.

## Usage

```bash
python score_pitches.py pitches.parquet --out values.csv
python score_pitches.py pitches.parquet --pitcher "Nolan McLean" --parts --out mclean.csv
```

| option | effect |
|---|---|
| `--out` | output file, `.csv` or `.parquet` |
| `--pitcher` | score only these pitchers (names or MLBAM ids) |
| `--parts` | add each value's balls / strikes / batted-ball split (the parts sum to the value) |
| `--probs` | add the nine outcome probabilities behind each value |
| `--count pre` | the input's `balls`/`strikes` are already the pre-pitch count |
| `--models`, `--constants` | alternative model / constant directories |
| `--kernel`, `--workers` | `numba` or `numpy` scoring kernel, and processes for the numpy kernel |

## Input

One row per pitch, in Statcast / MLB Stats API ("statfast") format, as a `.parquet` or `.csv`
file.

| | columns |
|---|---|
| required | `pitcher`, `pitch_type`, `p_throws`, `stand` |
| | `release_speed`, `release_extension`, `release_pos_x`, `release_pos_z`, `release_spin_rate`, `spin_axis` |
| | `vx0`, `vy0`, `vz0`, `ax`, `ay`, `az` (the 9-parameter fit at y = 50 ft) |
| | `plate_x`, `plate_z`, `sz_top`, `sz_bot`, `balls`, `strikes` |
| | `game_pk`, `at_bat_index`, `pitch_number`, and `game_date` or `season` |
| optional | `call_code`, `event_desc`, `events`: swing/take filter and `observed_rv` |
| | `home_team`: park elevation for spin efficiency (default 517 ft, the league average) |
| | `pitcher_name`, `batter` |

Notes:

* **Count.** `balls`/`strikes` are read as the count *after* the pitch (as the MLB Stats API
  reports it). The pre-pitch count is rebuilt from the previous pitch of the plate
  appearance, so pass every pitch of each plate appearance. Use `--count pre` if the file
  already has pre-pitch counts.
* **Whole outings.** Each pitch's differences from the pitcher's primary fastball (velocity,
  movement) come from the pitches passed in, so pass whole outings or seasons.
* **What gets scored.** With `call_code`, only swing/take decisions are scored, as in
  training: no pitchouts, bunts, or automatic or intentional balls. Pitch types outside the
  Fastball (FF, SI, primary FC), Breaking (SL, ST, SV, CU, KC, secondary FC) and Offspeed (CH,
  FS, FO) groups are dropped.

## Output

One row per scored pitch: ids, count, pitch type and group, plate location, the observed
outcome (with `call_code`), and these values. All are **runs per pitch from the pitcher's
side** (+ = good for the pitcher); multiply by 100 for runs per 100 pitches ("rv100").

| column | meaning |
|---|---|
| `stuff_rv` | Stuff, count-neutral: the pitch over the league's location mix, averaged over the league's count mix, with count-neutral run values |
| `stuff_rv_asused` | Stuff at the pitch's actual count (league location mix) |
| `pitching_rv` | the pitch at its actual location and count |
| `location_rv` | `pitching_rv − stuff_rv_asused` |
| `observed_rv` | the run value of what actually happened, in that count |

## Plus scales

Average a value over a unit's pitches, convert it to rv100, and scale it:

```
plus = 100 + 15 × (rv100 − mean) / sd
```

`constants/plus_scale_constants.json` has the mean and sd (rv100) of each value for four
aggregations. They are pooled over every 2023–26 unit scored by this script (2.8M pitches) and
weighted by pitches. 2023–25 are the models' training seasons.

| aggregation | keys | 2023–26 units |
|---|---|---|
| `pitcher_season` | pitcher, season | 3,278 |
| `pitcher_season_pitch_type` | pitcher, season, `pt` | 15,068 |
| `pitcher_game` | pitcher, `game_pk` | 82,158 |
| `pitcher_game_pitch_type` | pitcher, `game_pk`, `pt` | 297,065 |

Each covers `stuff_rv`, `stuff_rv_asused`, `pitching_rv` and `location_rv`. Smaller units
have wider SDs, because more of their spread is noise, so use the constants for the
aggregation you computed.

```python
import json
import pandas as pd

v = pd.read_csv("values.csv")
scale = json.load(open("constants/plus_scale_constants.json"))
c = scale["aggregations"]["pitcher_game"]["columns"]
g = v.groupby(["pitcher", "game_pk"])[["stuff_rv", "pitching_rv", "location_rv"]].mean() * 100
for col, name in (("stuff_rv", "Stuff+"), ("pitching_rv", "Pitching+"), ("location_rv", "Location+")):
    g[name] = 100 + 15 * (g[col] - c[col]["mean"]) / c[col]["sd"]
```

## How it works

Four chained models cover the pitch outcomes:

1. swing / take
2. called strike / ball
3. whiff / foul / in play
4. batted-ball result (out, 1B, 2B, 3B, HR)

Each is fit separately for Fastball, Breaking and Offspeed pitches, against same- and
opposite-handed batters. Every stage is a sequential logit with a DML design:

* A location model l(W) uses plate location, count and handedness.
* A stuff index D(S) uses velocity, movement, release, extension, spin, spin efficiency,
  seam-shifted wake and differences from the primary fastball. Pitch type is never an input.
  D(S) is fit to what location does not explain, and its effect is estimated per count.

The chained probabilities for the nine outcomes (ball, called strike, whiff, foul, and five
batted-ball results) are valued with linear weights.

* **Stuff** averages each pitch over stored league location samples, and over counts.
* **Pitching** uses the pitch's own location and count.

| path | contents |
|---|---|
| `score_pitches.py` | the scorer |
| `models/*_logit_2325.pkl` | the four fitted models (plain dicts of LightGBM model strings and numpy arrays) |
| `constants/count_mix.json` | league share of each pre-pitch count |
| `constants/run_values.csv` | count-neutral run value of each outcome |
| `constants/run_values_by_count.csv` | run value of each outcome in each count |
| `constants/plus_scale_constants.json` | plus-scale constants: pitcher season / season × pitch type / game / game × pitch type (2023–26) |

On all 700,262 scored 2026 pitches, this script reproduces the reference pipeline's values
to within 0.013 rv100 per pitcher × pitch type. The small differences come from the constant
fit plane.
