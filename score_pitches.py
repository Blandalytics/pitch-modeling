"""Per-pitch Stuff, Pitching and Location run values from the trained pitch models.

Standalone: numpy, pandas and lightgbm (plus pyarrow for parquet, and numba if installed). The
models are plain dicts of numpy arrays and LightGBM model strings, and everything below mirrors
the project's dml_stuff_swing.py / combine.py / pitch_values.py (the function each mirrors is
named in its docstring).

Values, in runs per pitch from the pitcher's side (+ = good for the pitcher; x 100 for the
per-100-pitch "rv100" scale):

    stuff_rv          Stuff, count-neutral: the pitch's stuff over the league's location mix,
                      averaged over the league count mix and valued with count-neutral weights
    stuff_rv_asused   Stuff at the pitch's actual count (league location mix)
    pitching_rv       the pitch at its actual location and count (location + stuff models)
    location_rv       pitching_rv - stuff_rv_asused: what the location added to the stuff
    observed_rv       the actual outcome's run value in its count (needs call_code / events)

Input: statfast-format pitches (.parquet or .csv), one row per pitch:

    required  pitcher pitch_type p_throws stand release_speed release_extension
              release_pos_x release_pos_z vx0 vy0 vz0 ax ay az release_spin_rate spin_axis
              plate_x plate_z sz_top sz_bot balls strikes
              game_date (or season), game_pk, at_bat_index, pitch_number
    optional  call_code, event_desc, events   swing/take filter and observed_rv
              home_team                       park elevation for spin efficiency
              hb, ivb, release_pos_y          statsapi movement for the pitch groups
              pitcher_name, batter

release_pos_y is not used: statfast's release_pos_x/z are the 9-parameter fit at the y = 50 ft
plane, which this script takes as the constant 50.0 when propagating back to the release point
(the reported value only ever differs from 50 by < 0.01 ft).

balls / strikes are statsapi's count *after* the pitch (statfast); the count the pitch was
thrown in is rebuilt from the previous pitch of the plate appearance (--count pre if the file
already has pre-pitch counts). With call_code, only swing/take decisions are scored, as in
training (no pitchouts, bunts, automatic or intentional balls). Pitch groups come from
pitch_groups.py: the Level 1 classifier (pitch_l1.py, arm-adjusted field only) per
pitcher-game-pitch type, else the Statcast pitch type (without hb/ivb, pitch type only).
Pitches outside the Fastball / Breaking / Offspeed groups are dropped. The primary fastball,
the arsenal deltas and the groups come from the pitches passed in, so pass whole outings.

2026 (abs_2026.py, abs_2026.json in the constants): 2026 rows are scored with each batter's
pose-convention reference zone instead of statfast's fixed ABS zone (unless the input already
carries sz_top_abs / sz_bot_abs, i.e. was swapped upstream). Their location-aware swing,
called-strike and whiff stages are recalibrated to 2026's calls and swings, and the as-used
stages get the matching league-wide shift per count. Pitching, Stuff as used and Location
change; count-neutral Stuff does not. --no-abs scores 2026 as-is. In-process callers pass
``env=abs_2026.load(...)`` to score(); the default (None) applies none of it, which is right
for any level but MLB.

    python score_pitches.py pitches.parquet --out values.csv [--parts] [--probs]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import abs_2026
import pitch_groups

HERE = Path(__file__).resolve().parent


def _default_dir(name: str) -> Path:
    """The repo layout (models/, constants/), else the project's output/."""
    return HERE / name if (HERE / name).is_dir() else HERE / "output"


MODEL_FILES = {  # chained in this order; the class order is checked on load
    "swing_take": ("swing_take_logit_2325.pkl", ("swing", "take")),
    "take_outcome": ("take_outcome_logit_2325.pkl", ("called_strike", "ball")),
    "swing_outcome": ("swing_outcome_logit_2325.pkl", ("whiff", "foul", "in_play")),
    "in_play": ("in_play_logit_2325.pkl", ("field_out", "single", "double", "triple", "home_run")),
}
BATTED = MODEL_FILES["in_play"][1]
OUTCOMES = ("ball", "called_strike", "swinging_strike", "foul", *BATTED)
RV_SUBSETS = {"balls": ("ball",), "strikes": ("called_strike", "swinging_strike", "foul"),
              "bbe": BATTED}  # fmt: skip
N_COUNT = 12  # balls 0-3 x strikes 0-2, index balls * 3 + strikes
CHUNK = 2048  # rows per numpy-kernel call
TASK_ROWS_MIN = 2048

STUFF_COLS = ["pitcher", "pitch_type", "p_throws", "release_speed", "release_extension",
              "release_pos_x", "release_pos_z", "vx0", "vy0", "vz0", "ax", "ay", "az",
              "release_spin_rate", "spin_axis"]  # fmt: skip
LOCATION_COLS = ["stand", "plate_x", "plate_z", "sz_top", "sz_bot", "balls", "strikes"]
PA_KEYS = ["game_pk", "at_bat_index", "pitch_number"]
# dml_stuff_swing.SWING_TAKE: swing (S W F T X D E) or take (B *B C H), bunts dropped
DECISIONS = frozenset(["S", "W", "F", "T", "X", "D", "E", "B", "*B", "C", "H"])
IN_PLAY_CODES = frozenset(["X", "D", "E"])
OBSERVED = {"B": "ball", "*B": "ball", "H": "ball", "C": "called_strike", "S": "swinging_strike",
            "W": "swinging_strike", "T": "swinging_strike", "F": "foul"}  # fmt: skip
HITS = {"single": "single", "double": "double", "triple": "triple", "home_run": "home_run"}

# ---- stuff features (mirror dml_stuff_swing.py) ---------------------------------------------
FB_CANDIDATES = frozenset(["FF", "SI", "FC"])
MATCHUPS = {1: "Same Hand", 0: "Opposite Hand"}  # platoon flag -> model name suffix
STUFF_NUM = ["velo", "ax_m", "az", "rel_x_m", "rel_z", "extension", "spin_rate", "spin_eff",
             "axis_diff", "velo_diff", "ax_diff", "az_diff"]  # fmt: skip
G = 32.174  # ft/s^2
Y_FIT = 50.0  # ft: the plane of statfast's 9-parameter fit (release_pos_x/z, vx0.., ax..)
Y_PLATE = 17 / 12
BALL_R = 0.1208
BALL_A = np.pi * BALL_R**2
BALL_M = 0.3203
RHO_SEA = 0.0747  # lb/ft^3, ~70F
SCALE_HEIGHT = 27_500.0  # ft
PARK_ELEVATION = {
    "AZ": 1082, "ARI": 1082, "ATL": 1050, "BAL": 33, "BOS": 20, "CHC": 595, "CWS": 595,
    "CIN": 490, "CLE": 653, "COL": 5190, "DET": 600, "HOU": 43, "KC": 750, "LAA": 160,
    "LAD": 515, "MIA": 7, "MIL": 635, "MIN": 815, "NYM": 20, "NYY": 55, "OAK": 25, "ATH": 25,
    "PHI": 20, "PIT": 730, "SD": 20, "SF": 0, "SEA": 10, "STL": 465, "TB": 45, "TEX": 551,
    "TOR": 270, "WSH": 25,
}  # fmt: skip
DEFAULT_ELEVATION = 517  # ft: pitch-weighted mean park elevation over 2023-26, rounded


def _col(df: pd.DataFrame, c: str) -> np.ndarray:
    return df[c].to_numpy(np.float64)


def _flight_times(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """_flight_times: release and plate times relative to the y = 50 ft fit plane (Y_FIT)."""
    vy0, ay = _col(df, "vy0"), _col(df, "ay")
    ay = np.where(np.abs(ay) < 1e-6, 1e-6, ay)
    y_rel = 60.5 - _col(df, "release_extension")
    vy_r = -np.sqrt(vy0**2 + 2 * ay * (y_rel - Y_FIT))
    vy_f = -np.sqrt(vy0**2 + 2 * ay * (Y_PLATE - Y_FIT))
    return (vy_r - vy0) / ay, (vy_f - vy0) / ay


def _magnus(df: pd.DataFrame, t_mid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """_magnus: spin-induced acceleration (drag and gravity removed) and speed, mid-flight."""
    a = np.column_stack([_col(df, "ax"), _col(df, "ay"), _col(df, "az") + G])
    v0 = np.column_stack([_col(df, "vx0"), _col(df, "vy0"), _col(df, "vz0")])
    v = v0 + a * t_mid[:, None]
    speed = np.linalg.norm(v, axis=1)
    vhat = v / speed[:, None]
    return a - (a * vhat).sum(1, keepdims=True) * vhat, speed


def _spin_efficiency(df: pd.DataFrame, mag: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """_spin_efficiency: transverse spin implied by the Magnus acceleration / total spin."""
    park = df["home_team"].astype("string").map(PARK_ELEVATION)
    rho = RHO_SEA * np.exp(-park.fillna(DEFAULT_ELEVATION).to_numpy(float) / SCALE_HEIGHT)
    cl = np.linalg.norm(mag, axis=1) / (0.5 * rho * BALL_A / BALL_M * speed**2)
    spin_param = np.where(cl < 0.15, cl / 1.5, (cl - 0.09) / 0.6)
    transverse_rpm = spin_param * speed / BALL_R * 60 / (2 * np.pi)
    return np.clip(transverse_rpm / _col(df, "release_spin_rate").clip(1), 0, 1.25)


def physics(df: pd.DataFrame) -> pd.DataFrame:
    """_physics: release point, glove-side-negative mirrored x, gravity-free az, spin
    efficiency, axis diff."""
    t_rel, t_plate = _flight_times(df)
    mirror = np.where(df["p_throws"].to_numpy() == "R", -1.0, 1.0)
    x_rel = _col(df, "release_pos_x") + _col(df, "vx0") * t_rel + 0.5 * _col(df, "ax") * t_rel**2
    z_rel = _col(df, "release_pos_z") + _col(df, "vz0") * t_rel + 0.5 * _col(df, "az") * t_rel**2
    mag, speed = _magnus(df, 0.5 * (t_rel + t_plate))
    inferred = (180 + np.degrees(np.arctan2(-mag[:, 0], mag[:, 2]))) % 360
    diff = (_col(df, "spin_axis") - inferred + 180) % 360 - 180
    return df.assign(
        velo=_col(df, "release_speed"), ax_m=_col(df, "ax") * mirror, az=_col(df, "az") + G,
        rel_x_m=x_rel * mirror, rel_z=z_rel, extension=_col(df, "release_extension"),
        spin_rate=_col(df, "release_spin_rate"), spin_eff=_spin_efficiency(df, mag, speed),
        axis_diff=diff * np.where(mirror < 0, 1.0, -1.0),
    )  # fmt: skip


def _primary_fastballs(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """_primary_fastballs: most-thrown FF/SI/FC per key group (ties -> harder)."""
    fb = df[df["pt"].isin(FB_CANDIDATES)]
    agg = fb.groupby([*keys, "pt"], observed=True).agg(
        n=("velo", "size"), fb_velo=("velo", "mean"), fb_ax=("ax_m", "mean"), fb_az=("az", "mean")
    )
    agg = agg.reset_index().sort_values(["n", "fb_velo"]).groupby(keys).tail(1)
    return agg.rename(columns={"pt": "fb_type"}).drop(columns="n")


def arsenal(df: pd.DataFrame) -> pd.DataFrame:
    """_arsenal: primary-fastball flag and deltas vs the same-game primary fastball (season
    fallback), and the platoon flag."""
    game = _primary_fastballs(df, ["pitcher", "game_pk"])
    season = _primary_fastballs(df, ["pitcher", "season"])
    out = df.merge(game, on=["pitcher", "game_pk"], how="left")
    fill = out[["pitcher", "season"]].merge(season, on=["pitcher", "season"], how="left")
    for c in ("fb_type", "fb_velo", "fb_ax", "fb_az"):
        out[c] = out[c].fillna(fill[c])
    primary = (out["pt"] == out["fb_type"]).fillna(False).to_numpy(bool)
    out["is_primary"] = primary.astype(np.int8)
    for c, ref in (("velo", "fb_velo"), ("ax", "fb_ax"), ("az", "fb_az")):
        src = "ax_m" if c == "ax" else c
        out[f"{c}_diff"] = np.where(primary, 0.0, out[src] - out[ref])
    out["platoon"] = (out["p_throws"] == out["stand"]).astype(np.int8)
    return out


def pitch_group(df: pd.DataFrame) -> pd.Series:
    """pitch_group: pitch_groups.assign, the Level 1 family of the pitcher-game-pitch type
    (pitch_l1.py, arm-adjusted field), else the Statcast pitch type; run on whole outings."""
    return pitch_groups.assign(df)


def _pre_pitch_count(df: pd.DataFrame) -> pd.DataFrame:
    """_pre_pitch_count: statsapi's balls/strikes are the count after the pitch; the count a
    pitch was thrown in is the previous pitch's (0-0 for the first of the plate appearance).
    Runs on every row passed in, so pitchouts, bunts etc. still advance the count."""
    df = df.sort_values(PA_KEYS, kind="stable")
    pa = df.groupby(["game_pk", "at_bat_index"], sort=False)
    for c in ("balls", "strikes"):
        df[c] = pa[c].shift(1).fillna(0).astype(np.int8)
    return df


def features(raw: pd.DataFrame, count: str = "post") -> pd.DataFrame:
    """prepare(raw, SWING_TAKE): stuff and location inputs, pitch group and model for every
    scoreable pitch. ``input_row`` is the pitch's row in ``raw``."""
    need = [*STUFF_COLS, *LOCATION_COLS, *PA_KEYS]
    missing = [c for c in need if c not in raw] + (
        [] if {"game_date", "season"} & set(raw.columns) else ["game_date or season"])
    if missing:
        raise ValueError(f"missing required column(s): {missing}")
    df = raw.reset_index(drop=True).assign(input_row=lambda d: np.arange(len(d)))
    if count == "post":
        df = _pre_pitch_count(df)
    df["group"] = pitch_group(df)  # Level 1 works on whole outings: before any filter
    ok = df[[*STUFF_COLS, *LOCATION_COLS]].notna().all(axis=1)
    ok &= (df["balls"] <= 3) & (df["strikes"] <= 2) & (df["sz_top"] > df["sz_bot"])
    ok &= df["p_throws"].isin(["L", "R"]) & df["stand"].isin(["L", "R"])
    df = df[ok].reset_index(drop=True)
    df["pt"] = df["pitch_type"].astype("string").fillna("UN")
    if "season" in raw:
        df["season"] = df["season"].astype(np.int16)
    else:
        df["season"] = pd.to_datetime(df["game_date"]).dt.year.astype(np.int16)
    if "home_team" not in df:
        df["home_team"] = pd.NA
    df = arsenal(physics(df))
    df = df[np.isfinite(df["spin_eff"])].reset_index(drop=True)
    if "call_code" in df:  # SWING_TAKE.label: swing/take decisions only, bunts dropped
        code = df["call_code"].astype("string")
        keep = code.isin(list(DECISIONS)).fillna(False).to_numpy(bool)
        desc = df["event_desc"] if "event_desc" in df else pd.Series(pd.NA, index=df.index)
        bunt = desc.astype("string").str.contains("bunt", case=False, na=False)
        keep = keep & ~(code.isin(list(IN_PLAY_CODES)) & bunt).fillna(False).to_numpy(bool)
        df = df[keep].reset_index(drop=True)
    # _location: batter-relative plate x, zone-normalised height
    df["x_b"] = _col(df, "plate_x") * np.where(df["stand"].to_numpy() == "R", -1.0, 1.0)
    df["z_n"] = (_col(df, "plate_z") - _col(df, "sz_bot")) / (
        _col(df, "sz_top") - _col(df, "sz_bot"))  # fmt: skip
    matchup = df["platoon"].map(MATCHUPS).to_numpy()
    return df.assign(model=df["group"].astype(str).to_numpy() + " vs " + matchup)


def observed(df: pd.DataFrame) -> pd.Series:
    """combine.observed: the actual outcome; a ball in play is its batted-ball result (errors,
    fielder's choices, sacrifices and double plays count as field outs)."""
    code = df["call_code"].astype(str)
    obs = code.map(OBSERVED)
    bip = code.isin(list(IN_PLAY_CODES)).to_numpy()
    events = df["events"] if "events" in df else pd.Series(pd.NA, index=df.index)
    obs[bip] = events[bip].astype(str).map(HITS).fillna("field_out")
    return obs


# ---- models ---------------------------------------------------------------------------------
def load_models(model_dir: Path) -> list[dict]:
    """The saved models (plain dicts), with every booster loaded once."""
    models = []
    for name, (file, classes) in MODEL_FILES.items():
        blob = pickle.loads((model_dir / file).read_bytes())
        if tuple(blob["classes"]) != classes or blob.get("link") != "logit":
            raise ValueError(f"{file}: expected a logit {name} model with classes {classes}")
        for gm in blob["groups"].values():
            if gm.get("link", "sequential") != "sequential":
                raise ValueError(f"{file}: only the sequential link is supported")
            if not gm.get("loc_models"):
                raise ValueError(f"{file}: no location models (needed for Pitching)")
            for k in ("boosters", "loc_models", "r_models"):
                gm[f"_{k}"] = [lgb.Booster(model_str=b) for b in gm[k]]
        models.append(blob)
    return models


def _season(gm: dict, df: pd.DataFrame) -> np.ndarray:
    """Seasons after the last training season are scored as that season."""
    return np.minimum(df["season"].to_numpy(), max(gm["seasons"]))


def _season_code(gm: dict, df: pd.DataFrame) -> list[np.ndarray]:
    seasons = gm["seasons"]
    return [np.searchsorted(seasons, _season(gm, df)).astype(np.float32)] if len(seasons) > 1 else []


def _stuff_index(gm: dict, df: pd.DataFrame) -> np.ndarray:
    """LogitGroupModel.indices / stuff_matrix: per-stage stuff log-odds D(S)."""
    X = [df[c].to_numpy(np.float32) for c in STUFF_NUM]
    X += [df["is_primary"].to_numpy(np.float32), df["platoon"].to_numpy(np.float32)]
    X = np.column_stack([*X, *_season_code(gm, df)])
    return np.column_stack([b.predict(X, raw_score=True) for b in gm["_boosters"]])


def _location_matrix(gm: dict, df: pd.DataFrame) -> np.ndarray:
    """location_matrix: W = batter-relative x, zone-normalised z, count, hands (and season)."""
    W = [df["x_b"].to_numpy(np.float32), df["z_n"].to_numpy(np.float32),
         df["balls"].to_numpy(np.float32), df["strikes"].to_numpy(np.float32),
         (df["p_throws"] == "L").to_numpy(np.float32), (df["stand"] == "L").to_numpy(np.float32)]  # fmt: skip
    return np.column_stack([*W, *_season_code(gm, df)])


def _count(df: pd.DataFrame) -> np.ndarray:
    return df["balls"].to_numpy(np.int64) * 3 + df["strikes"].to_numpy(np.int64)


def _sequential(q: np.ndarray) -> np.ndarray:
    """sequential: class probabilities from stage probabilities q (rows x stages)."""
    stay, probs = np.ones(len(q)), []
    for s in range(q.shape[1]):
        probs.append(stay * q[:, s])
        stay = stay * (1 - q[:, s])
    return np.column_stack([*probs, stay])


def _full_probs(gm: dict, df: pd.DataFrame, D: np.ndarray) -> np.ndarray:
    """LogitGroupModel.location_probs (full): per stage, logit l(W) + r(W) + theta_c (D - r) at
    the pitch's actual location and count, chained into class probabilities."""
    W, c = _location_matrix(gm, df), _count(df)
    q = []
    for s, (lm, rm, th) in enumerate(
        zip(gm["_loc_models"], gm["_r_models"], gm["thetas"], strict=True)
    ):
        ell = np.clip(lm.predict(W), 1e-4, 1 - 1e-4)
        r = rm.predict(W)
        q.append(1 / (1 + np.exp(-(np.log(ell / (1 - ell)) + r + th[c] * (D[:, s] - r)))))
    return _sequential(np.column_stack(q))


def _class_probs_numpy(t: np.ndarray, Gd: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """class_probs: sequential-logit class probabilities averaged over location draws, in
    float32 with preallocated buffers (the fallback when numba is not installed)."""
    tt = (t + shift).astype(np.float32)
    G32 = np.ascontiguousarray(Gd, dtype=np.float32)
    n, S = t.shape
    out = np.empty((n, S + 1))
    stay = np.ones((n, len(Gd)), np.float32)
    q = np.empty_like(stay)
    for s in range(S):
        np.add(tt[:, [s]], G32[None, :, s], out=q)
        np.negative(q, out=q)
        np.exp(q, out=q)
        q += 1
        np.reciprocal(q, out=q)
        q *= stay
        out[:, s] = q.mean(1)
        stay -= q
    out[:, S] = stay.mean(1)
    return out


try:  # the compiled kernel: exact float64, rows split over threads
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _class_probs_numba(t, Gd, shift):
        n, S = t.shape
        m = Gd.shape[0]
        out = np.zeros((n, S + 1))
        for i in prange(n):
            for g in range(m):
                stay = 1.0
                for s in range(S):
                    q = 1.0 / (1.0 + np.exp(-(t[i, s] + Gd[g, s] + shift[s])))
                    out[i, s] += stay * q
                    stay *= 1.0 - q
                out[i, S] += stay
            for k in range(S + 1):
                out[i, k] /= m
        return out

    KERNELS = {"numba": (_class_probs_numba, None), "numpy": (_class_probs_numpy, CHUNK)}
except ImportError:
    KERNELS = {"numpy": (_class_probs_numpy, CHUNK)}
DEFAULT_KERNEL = "numba" if "numba" in KERNELS else "numpy"


def _draws(gm: dict, cell: tuple) -> np.ndarray:
    """combine._draw_key: the cell's league location draws, else the same count and hands in
    the nearest season that has them."""
    for season in sorted(gm["seasons"], key=lambda sn: abs(sn - cell[0])):
        key = (season, *cell[1:])
        if key in gm["draws"]:
            return gm["draws"][key]
    raise KeyError(f"no location draws for {cell}")


def _neutral(model: dict, gm: dict, t: np.ndarray, cells: dict, kernel, delta=None) -> np.ndarray:
    """neutral_probs: class probabilities for stuff log-odds t, averaged over each cell's
    league location draws. ``cells`` maps (season, balls, strikes, p_throws, stand) to rows; a
    sixth key element True marks rows of the abs_2026 season, whose first-stage league shift gets
    delta[count] (combine.abs_asused)."""
    out = np.full((len(t), t.shape[1] + 1), np.nan)
    fn, chunk = kernel
    base = np.asarray(model["shift"], np.float64)
    for cell, idx in cells.items():
        Gd = _draws(gm, cell[:5])
        shift = base
        if delta is not None and len(cell) > 5 and cell[5]:
            shift = base.copy()
            shift[0] += delta[cell[1] * 3 + cell[2]]
        for start in range(0, len(idx), chunk or len(idx)):
            rows = idx[start : start + (chunk or len(idx))]
            out[rows] = fn(np.ascontiguousarray(t[rows]), Gd, shift)
    return out


def _chain(swing_take, take_outcome, swing_outcome, in_play) -> np.ndarray:
    """combine._chain: the nine OUTCOMES from the four models' class probabilities."""
    swing, take = swing_take[:, 0], swing_take[:, 1]
    return np.column_stack([
        take * take_outcome[:, 1], take * take_outcome[:, 0],
        swing * swing_outcome[:, 0], swing * swing_outcome[:, 1],
        (swing * swing_outcome[:, 2])[:, None] * in_play,
    ])  # fmt: skip


# ---- count-neutral Stuff (the slow part: every pitch at all 12 counts) -----------------------
_WORKER: dict = {}


def _slim(models: list[dict]) -> list[dict]:
    keep = ("thetas", "seasons", "draws")
    return [{"shift": model["shift"],
             "groups": {n: {k: gm[k] for k in keep} for n, gm in model["groups"].items()}}
            for model in models]  # fmt: skip


def _init_worker(models: list[dict], kernel: str) -> None:
    _WORKER.update(models=models, kernel=KERNELS[kernel])


def _neutral_block(task) -> np.ndarray:
    """combine.count_average for one block: nine-outcome probabilities at each count, weighted
    by mix[c]."""
    name, D, hands, mix = task
    models, kernel = _WORKER["models"], _WORKER["kernel"]
    out = np.zeros((len(hands), len(OUTCOMES)))
    groups = hands.groupby(["season", "p_throws", "stand"], observed=True).indices
    for c in np.flatnonzero(mix):
        probs = []
        for model, d in zip(models, D, strict=True):
            gm = model["groups"][name]
            t = np.column_stack([th[c] * d[:, s] for s, th in enumerate(gm["thetas"])])
            cells = {(int(sn), c // 3, c % 3, str(pt), str(st)): idx
                     for (sn, pt, st), idx in groups.items()}  # fmt: skip
            probs.append(_neutral(model, gm, t, cells, kernel))
        out += mix[c] * _chain(*probs)
    return out


def count_neutral(models, df, D_by_group, mix, workers=None, kernel=None) -> np.ndarray:
    """combine.count_neutral: nine-outcome probabilities at each count, weighted by the league
    count mix. With numba one process (threads use every core), else one process per core."""
    kernel = kernel or DEFAULT_KERNEL
    workers = workers or (1 if kernel == "numba" else os.cpu_count() or 1)
    task_rows = max(TASK_ROWS_MIN, -(-len(df) // (4 * workers)))
    pairs = []
    for name, rows in df.groupby("model", sort=False).indices.items():
        sub = df.iloc[rows]
        gm = models[0]["groups"][name]
        hands = pd.DataFrame({"season": _season(gm, sub), "p_throws": sub["p_throws"].to_numpy(),
                              "stand": sub["stand"].to_numpy()})  # fmt: skip
        D = D_by_group[name]
        for start in range(0, len(rows), task_rows):
            block = slice(start, start + task_rows)
            task = (name, [d[block] for d in D], hands.iloc[block].reset_index(drop=True), mix)
            pairs.append((rows[block], task))
    tasks = [task for _, task in pairs]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(min(workers, len(tasks)), initializer=_init_worker,
                                 initargs=(_slim(models), kernel)) as pool:  # fmt: skip
            results = list(pool.map(_neutral_block, tasks))
    else:
        _init_worker(_slim(models), kernel)
        results = [_neutral_block(task) for task in tasks]
    out = np.zeros((len(df), len(OUTCOMES)))
    for (rows, _), block in zip(pairs, results, strict=True):
        out[rows] = block
    return out


# ---- scoring --------------------------------------------------------------------------------
def load_constants(constants_dir: Path) -> tuple[np.ndarray, pd.Series, pd.DataFrame]:
    """League count mix, count-neutral outcome values and outcome values by count."""
    mix = json.loads((constants_dir / "count_mix.json").read_text())
    mix = np.array([mix[f"{c // 3}-{c % 3}"] for c in range(N_COUNT)])
    values = pd.read_csv(constants_dir / "run_values.csv", index_col="outcome")["run value"]
    by_count = pd.read_csv(constants_dir / "run_values_by_count.csv", index_col="outcome")
    return mix, values.reindex(list(OUTCOMES)), by_count.reindex(list(OUTCOMES))


def score(raw: pd.DataFrame, models: list[dict], mix, values, by_count, count="post", env=None,
          **par):  # fmt: skip
    """Per pitch: the nine outcome probabilities of each kind (count-neutral Stuff, Stuff at the
    actual count, Pitching) and the run values. Returns (pitches, rows dropped as Other). With
    ``env`` (abs_2026.load), its season's rows get the reference zone, the recalibrated swing,
    called-strike and whiff stages (Pitching) and the matching as-used shifts (Stuff as used;
    count-neutral Stuff is unchanged)."""
    if env:
        raw = abs_2026.apply_zone(raw, env)
    df = features(raw, count)
    modeled = df["model"].isin(set.intersection(*(set(m["groups"]) for m in models)))
    dropped = int((~modeled).sum())
    df = df[modeled].reset_index(drop=True)
    kernel = KERNELS[par.get("kernel") or DEFAULT_KERNEL]
    D_by_group, asused, full = {}, np.zeros((len(df), 9)), np.zeros((len(df), 9))
    for name, rows in df.groupby("model", sort=False).indices.items():
        sub = df.iloc[rows].reset_index(drop=True)
        gms = [m["groups"][name] for m in models]
        D = [_stuff_index(gm, sub) for gm in gms]
        D_by_group[name] = D
        c, in_env = _count(sub), sub["season"].to_numpy() == (env or {}).get("season")
        keys = pd.DataFrame({"season": _season(gms[0], sub), "balls": sub["balls"].to_numpy(),
                             "strikes": sub["strikes"].to_numpy(),
                             "p_throws": sub["p_throws"].to_numpy(),
                             "stand": sub["stand"].to_numpy(),
                             "env": in_env})  # fmt: skip
        cells = {(int(k[0]), int(k[1]), int(k[2]), str(k[3]), str(k[4]), bool(k[5])): idx
                 for k, idx in keys.groupby(list(keys.columns)).indices.items()}  # fmt: skip
        used, probs = [], []
        for stage, model, gm, d in zip(MODEL_FILES, models, gms, D, strict=True):
            t = np.column_stack([th[c] * d[:, s] for s, th in enumerate(gm["thetas"])])
            delta = abs_2026.asused_delta(stage, env)
            used.append(_neutral(model, gm, t, cells, kernel, delta))
            probs.append(abs_2026.adjust(stage, _full_probs(gm, sub, d), sub, env))
        asused[rows] = _chain(*used)
        full[rows] = _chain(*probs)
    neutral = count_neutral(models, df, D_by_group, mix, par.get("workers"), par.get("kernel"))
    at_count = by_count.to_numpy()[:, _count(df)].T  # pitches x outcomes
    parts = {"stuff_rv": -(neutral * values.to_numpy()), "stuff_rv_asused": -(asused * at_count),
             "pitching_rv": -(full * at_count)}  # fmt: skip
    parts["location_rv"] = parts["pitching_rv"] - parts["stuff_rv_asused"]
    out = df
    for name, rv in parts.items():
        out[name] = rv.sum(axis=1)
        for sub, outcomes in RV_SUBSETS.items():
            out[f"{name}_{sub}"] = rv[:, [OUTCOMES.index(o) for o in outcomes]].sum(axis=1)
    for kind, probs in (("stuff", neutral), ("stuff_asused", asused), ("pitching", full)):
        for k, o in enumerate(OUTCOMES):
            out[f"p_{o}_{kind}"] = probs[:, k]
    if "call_code" in out:
        out["observed"] = observed(out)
        obs = pd.Categorical(out["observed"], categories=list(OUTCOMES)).codes
        rv = -at_count[np.arange(len(out)), np.maximum(obs, 0)]
        out["observed_rv"] = np.where(obs >= 0, rv, np.nan)
    return out, dropped


VALUES = ["stuff_rv", "stuff_rv_asused", "pitching_rv", "location_rv"]
ID_COLS = ["game_pk", "game_date", "season", "at_bat_index", "pitch_number", "pitcher",
           "pitcher_name", "batter", "p_throws", "stand", "balls", "strikes", "pt", "group",
           "plate_x", "plate_z", "observed"]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data", help="statfast-format pitches (.parquet or .csv)")
    ap.add_argument("--out", required=True, help="per-pitch values (.parquet or .csv)")
    ap.add_argument("--parts", action="store_true", help="add each value's balls/strikes/bbe split")
    ap.add_argument("--probs", action="store_true", help="add the nine outcome probabilities")
    ap.add_argument("--pitcher", nargs="+", help="score only these pitchers (names or MLBAM ids)")
    ap.add_argument("--count", choices=["post", "pre"], default="post",
                    help="balls/strikes are the count after the pitch (statfast) or before it")
    ap.add_argument("--models", type=Path, default=_default_dir("models"))
    ap.add_argument("--constants", type=Path, default=_default_dir("constants"))
    ap.add_argument("--workers", type=int, help="processes (default: 1 with numba, else cores)")
    ap.add_argument("--kernel", choices=sorted(KERNELS), default=DEFAULT_KERNEL)
    ap.add_argument("--no-abs", action="store_true",
                    help="score 2026 as-is (statfast's ABS zone, no 2026 recalibration)")
    a = ap.parse_args()
    raw = pd.read_parquet(a.data) if a.data.endswith(".parquet") else pd.read_csv(a.data)
    if a.pitcher:
        names = raw["pitcher_name"].astype(str) if "pitcher_name" in raw else raw["pitcher"]
        raw = raw[names.isin(a.pitcher) | raw["pitcher"].astype(str).isin(a.pitcher)]
    models = load_models(a.models)
    mix, values, by_count = load_constants(a.constants)
    env = None if a.no_abs else abs_2026.load(a.constants / abs_2026.FILE)
    pitches, dropped = score(raw, models, mix, values, by_count, a.count, env,
                             workers=a.workers, kernel=a.kernel)  # fmt: skip
    cols = [c for c in ID_COLS if c in pitches] + VALUES
    cols += ["observed_rv"] if "observed_rv" in pitches else []
    if a.parts:
        cols += [f"{v}_{s}" for v in VALUES for s in RV_SUBSETS]
    if a.probs:
        cols += [f"p_{o}_{k}" for k in ("stuff", "stuff_asused", "pitching") for o in OUTCOMES]
    out = pitches[cols]
    if a.out.endswith(".parquet"):
        out.to_parquet(a.out, index=False)
    else:
        out.to_csv(a.out, index=False, encoding="utf-8-sig")
    per100 = {v: round(100 * float(out[v].mean()), 3) for v in VALUES}
    print(f"{len(raw):,} input rows: {len(out):,} pitches scored, {dropped:,} outside the "
          f"modeled groups dropped -> {a.out}")  # fmt: skip
    print("mean per 100 pitches:", per100)


if __name__ == "__main__":
    main()
