"""Level 1 pitch groups (Fastball / Breaking Ball / Offspeed) for any statfast pull.

    import statfast, pitch_l1
    df = statfast.pitcher_game("Cam Schlittler", game_date="2026-09-20")
    out = pitch_l1.classify(df)          # adds l1_pitch (per pitch) and l1 (per game-pitch type)

Per pitch
---------
1. hb_arm      statsapi hb, arm side + for both hands (LHP flipped)
2. arm angle   always estimated: OLS on [height, extension, signed release x, release z]
               plus the pitcher's stored offset (per season, else overall, else none);
               the model's mean when an input is missing. Release x/z are extrapolated
               from statsapi's y = 50 ft state back to the release plane.
3. rotation    the arm_angle/arm_adjusted_movement.py transform (phi = 90 - arm angle)
               -> horz_arm_adj, vert_arm_adj
4. dspeed      release_speed - 95th pct release_speed of that pitcher in that game
5. soft membership in two frozen 3-feature TopoTagger fields (2023-2026 MLB):
               A = {horz_arm_adj, vert_arm_adj, dspeed}, B = {hb_arm, ivb, dspeed}

Per pitcher-game-pitch type (the 0.75-threshold blend)
------------------------------------------------------
6. H_A = entropy (bits) of A's non-Unassigned families in the group;
   w = min(0.5, s * H_A) if H_A >= 0.75 else 0          (s = 0.68)
   l1_pitch = argmax((1 - w) p_A + w p_B); Unassigned only if both fields abstain
7. l1 = the group's most common non-Unassigned l1_pitch (ties: larger summed blend
   score); Unassigned only if every pitch in the group is.

CLI
---
    python pitch_l1.py day 2026-09-27 [-t R] [-o out.parquet]
    python pitch_l1.py pitcher-game "Cam Schlittler" --date 2026-09-20
    python pitch_l1.py season 2026 -o l1_2026.parquet

The frozen model (models/pitch_l1_v1.npz) is built by build_pitch_l1.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MODEL = ROOT / "models" / "pitch_l1_v1.npz"
STATFAST_DIR = Path(r"C:\Users\krbla\claude_sessions\savant_scraper")
FAMS = ["Fastball", "Breaking Ball", "Offspeed"]
LABELS = np.array(["Unassigned"] + FAMS, dtype=object)
REQUIRED = ["game_pk", "game_date", "pitcher", "p_throws", "pitch_type", "release_speed",
            "release_extension", "hb", "ivb", "release_pos_x", "release_pos_y", "release_pos_z",
            "vx0", "vy0", "vz0", "ax", "ay", "az"]
# statfast columns to request when this module does the pull itself
PULL_COLUMNS = REQUIRED + ["at_bat_index", "pitch_number", "pitcher_name", "batter", "stand",
                           "inning", "balls", "strikes"]


# ---------------------------------------------------------------------------- model
class Model:
    """Frozen artifacts: two soft-membership fields, blend parameters, arm-angle estimator."""

    def __init__(self, path: Path = MODEL):
        z = np.load(path, allow_pickle=False)
        self.params = json.loads(str(z["params"]))
        self.fields = {k: dict(p=z[f"{k}_p"], hard=z[f"{k}_hard"], lo=z[f"{k}_lo"],
                               width=(z[f"{k}_hi"] - z[f"{k}_lo"]) / z[f"{k}_bins"], bins=z[f"{k}_bins"])
                       for k in ("A", "B")}
        self.coef, self.intercept = z["aa_coef"], float(z["aa_intercept"])
        self.aa_mean = float(z["aa_mean"])
        self.off_ps = pd.Series(z["off_ps_val"], index=pd.MultiIndex.from_arrays(
            [z["off_ps_pitcher"], z["off_ps_season"]]))
        self.off_p = pd.Series(z["off_p_val"], index=z["off_p_pitcher"])
        self.height = pd.Series(z["height_val"], index=z["height_pitcher"])


_MODEL: Model | None = None


def load_model(path: Path = MODEL) -> Model:
    global _MODEL
    if _MODEL is None or path != MODEL:
        _MODEL = Model(path)
    return _MODEL


# ---------------------------------------------------------------------------- features
def _heights(model: Model, pitchers: np.ndarray) -> pd.Series:
    """Height (ft) per pitcher: model cache, then statsapi for any new pitcher."""
    out = pd.Series(model.height.reindex(pitchers).to_numpy(), index=pitchers)
    missing = out.index[out.isna()].tolist()
    if missing:
        try:
            import requests

            for k in range(0, len(missing), 150):
                r = requests.get("https://statsapi.mlb.com/api/v1/people",
                                 params={"personIds": ",".join(map(str, missing[k:k + 150])),
                                         "fields": "people,id,height"}, timeout=30).json()
                for p in r.get("people", []):
                    ft = pd.Series([p.get("height", "")]).str.extract(r"(\d+)'\s*(\d+)").astype(float)
                    out[p["id"]] = ft.iloc[0, 0] + ft.iloc[0, 1] / 12
        except Exception as exc:  # offline: the arm-angle estimate falls back to the mean
            print(f"pitch_l1: height lookup failed for {len(missing)} pitchers ({exc})", file=sys.stderr)
    return out


def features(df: pd.DataFrame, model: Model) -> pd.DataFrame:
    """Add hb_arm, release point, arm angle, arm-adjusted movement and dspeed."""
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"statfast frame is missing {miss}; pull with columns=pitch_l1.PULL_COLUMNS")
    out = df.copy()
    f = {c: out[c].astype(float) for c in REQUIRED if c not in
         ("game_pk", "game_date", "pitcher", "p_throws", "pitch_type")}
    lhp = out["p_throws"].astype(str).eq("L").to_numpy()
    out["hb_arm"] = np.where(lhp, -f["hb"], f["hb"])

    y_rel = 60.5 - f["release_extension"]
    t = (-f["vy0"] - np.sqrt(f["vy0"] ** 2 - 2 * f["ay"] * (f["release_pos_y"] - y_rel))) / f["ay"]
    out["rel_x"] = f["release_pos_x"] + f["vx0"] * t + 0.5 * f["ax"] * t ** 2
    out["rel_z"] = f["release_pos_z"] + f["vz0"] * t + 0.5 * f["az"] * t ** 2

    pitcher = out["pitcher"].astype(np.int64).to_numpy()
    season = pd.to_datetime(out["game_date"]).dt.year.to_numpy()
    h = _heights(model, np.unique(pitcher))
    X = np.column_stack([h.reindex(pitcher).to_numpy(), f["release_extension"],
                         out["rel_x"] * np.where(lhp, 1.0, -1.0), out["rel_z"]])
    est = X @ model.coef + model.intercept
    off = model.off_ps.reindex(pd.MultiIndex.from_arrays([pitcher, season])).to_numpy()
    off = np.where(np.isnan(off), model.off_p.reindex(pitcher).to_numpy(), off)
    est = est + np.nan_to_num(off)
    out["arm_angle_l1"] = np.where(np.isnan(est), model.aa_mean, est)
    out["arm_angle_l1_source"] = np.where(np.isnan(est), "mean", "estimated").astype(object)

    # arm_angle/arm_adjusted_movement.py: rotate so the arm points straight up
    phi = np.radians(90.0 - out["arm_angle_l1"].to_numpy(float))
    horz, vert = out["hb_arm"].to_numpy(float), f["ivb"].to_numpy()
    out["horz_arm_adj"] = horz * np.cos(phi) - vert * np.sin(phi)
    out["vert_arm_adj"] = horz * np.sin(phi) + vert * np.cos(phi)

    p95 = out.groupby(["pitcher", "game_pk"])["release_speed"].transform(
        lambda s: s.astype(float).quantile(model.params["dspeed_q"]))
    out["dspeed"] = f["release_speed"] - p95.astype(float)
    return out


def _lookup(field: dict, X: np.ndarray):
    finite = np.isfinite(X).all(axis=1)  # pitches with no tracking data stay unassigned
    idx = np.floor(np.nan_to_num((X - field["lo"]) / field["width"], nan=-1.0)).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < field["bins"]), axis=1) & finite
    idx = np.clip(idx, 0, field["bins"] - 1)
    P = field["p"][tuple(idx.T)].astype(float)
    P[~inside] = 0.0
    return P


# ---------------------------------------------------------------------------- classify
def classify(df: pd.DataFrame, model: Model | None = None) -> pd.DataFrame:
    """Return df with Level 1 columns added:

    arm_angle_l1, arm_angle_l1_source, hb_arm, horz_arm_adj, vert_arm_adj, dspeed,
    l1_H_A   arm-adjusted entropy of the pitcher-game-pitch type (bits)
    l1_w_raw weight on the raw-movement field for that group
    l1_pitch blended per-pitch family
    l1       modal family of the pitcher-game-pitch type (the label to use)
    """
    model = model or load_model()
    prm = model.params
    out = features(df, model).reset_index(drop=True)
    PA = _lookup(model.fields["A"], out[["horz_arm_adj", "vert_arm_adj", "dspeed"]].to_numpy(float))
    PB = _lookup(model.fields["B"], out[["hb_arm", "ivb", "dspeed"]].to_numpy(float))

    keys = pd.MultiIndex.from_frame(out[["pitcher", "game_pk"]].astype(np.int64).assign(
        pitch_type=out["pitch_type"].astype("string").fillna("NA")))
    code, _ = pd.factorize(keys)
    G = code.max() + 1

    lab_A = np.where(PA.max(1) > 0, PA.argmax(1) + 1, 0)
    cnt = np.zeros((G, 3))
    ok = lab_A > 0
    np.add.at(cnt, (code[ok], lab_A[ok] - 1), 1)
    q = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(q > 0, q * np.log2(q), 0).sum(1)
    w = np.where(H >= prm["T"], np.minimum(prm["cap"], prm["s"] * H), 0.0)

    S = (1 - w[code])[:, None] * PA + w[code][:, None] * PB + 1e-9 * PA
    lab = np.where(S.max(1) > 0, S.argmax(1) + 1, 0)
    out["l1_H_A"], out["l1_w_raw"] = H[code], w[code]
    out["l1_pitch"] = LABELS[lab]

    # modal family per pitcher-game-pitch type; ties -> larger summed blend score
    votes = np.zeros((G, 3))
    score = np.zeros((G, 3))
    ok = lab > 0
    np.add.at(votes, (code[ok], lab[ok] - 1), 1)
    np.add.at(score, code, S)
    key = votes * 1e6 + score  # votes dominate; score only breaks ties
    modal = np.where(votes.sum(1) > 0, key.argmax(1) + 1, 0)
    out["l1"] = LABELS[modal[code]]
    return out


def mix(out: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Pitch mix table: MLB pitch types with their Level 1 group, counts and shares."""
    by = by or (["pitcher_name"] if "pitcher_name" in out else ["pitcher"])
    g = out.groupby(by + ["pitch_type", "l1"], observed=True, dropna=False)
    t = g.agg(n=("release_speed", "size"), velo=("release_speed", "mean"),
              hb=("hb_arm", "mean"), ivb=("ivb", "mean")).reset_index()
    t["pct"] = 100 * t["n"] / t.groupby(by)["n"].transform("sum")
    return t.sort_values(by + ["n"], ascending=[True] * len(by) + [False]).round(1)


# ---------------------------------------------------------------------------- CLI
def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Level 1 pitch groups for statfast pulls")
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("day", "pitcher-game", "season"):
        p = sub.add_parser(name)
        p.add_argument("target", help="date (day), pitcher name/id (pitcher-game) or year (season)")
        p.add_argument("--date", help="game date for pitcher-game")
        p.add_argument("-t", "--game-type", default="R")
        p.add_argument("-o", "--out", help="write .parquet / .csv")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(STATFAST_DIR))
    import statfast

    kw = {"game_type": a.game_type, "columns": PULL_COLUMNS}
    if a.mode == "day":
        df = statfast.mlb_day(a.target, **kw)
    elif a.mode == "pitcher-game":
        df = statfast.pitcher_game(a.target, game_date=a.date, **kw)
    else:
        df = statfast.mlb_season(int(a.target), **kw)
    if df.empty:
        print("no pitches")
        return 1
    out = classify(df)
    print(f"{len(out):,} pitches, {out.game_pk.nunique()} games, {out.pitcher.nunique()} pitchers")
    print(out["l1"].value_counts(normalize=True).round(3).to_string())
    if a.mode == "pitcher-game":
        print(mix(out).to_string(index=False))
    if a.out:
        (out.to_parquet if a.out.endswith(".parquet") else out.to_csv)(a.out, index=False)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
