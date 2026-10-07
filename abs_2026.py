"""2026 adjustment for location-aware scoring (Pitching, Location): the ABS zone and its calls.

    python abs_2026.py      # fit on 2026 MLB decisions, cross-validate -> output/abs_2026.json

Two 2026 changes bias Pitching and Location when the 2023-25 models score 2026 pitches as-is:

1. **Zone definition.** statfast's 2026 sz_top/sz_bot are the fixed ABS zone (27% and 53.5% of
   height, one per batter-season), about 0.2 ft shorter than the pose-tracked 2023-25 zones the
   location models were trained on. ``apply_zone`` swaps in each batter's pose-convention
   reference zone from the projections project (src/ingest/zone.py, data/interim/zone_ref.parquet;
   copied into the JSON). The ABS zone is kept as sz_top_abs / sz_bot_abs.
2. **Called zone (ABS challenges).** With the reference zone, in-zone calls match 2025, but the
   called zone is sharper: more strikes called well inside, far fewer on the edges. Batters
   swing less outside the zone, and the chases they still make miss more often. Three
   recalibrations of the location-aware stage log-odds L (location model + stuff, at the pitch's
   location and count), fitted on 2026:

     take_outcome   P(called strike | take):  logit = a_count + s(L) + f(z_n) + g(z_abs) + h(xe)
     swing_take     P(swing):                 logit = L + a_count + f(z_n) + g(z_abs) + h(xe)
     swing_outcome  P(whiff | swing):         logit = L + a_count + f(z_n) + g(z_abs) + h(xe)

   z_n is the height in the reference zone (the models' input), z_abs the height in the ABS zone
   (the zone being called), xe = |plate_x| - 0.83 ft (+ = off the plate, ball radius included).
   Each spline is a restricted cubic spline (linear beyond its end knots). The take stage
   re-slopes L (the 2026 zone is sharper); the swing and whiff stages keep L's slope of 1 so
   differences in stuff are untouched and only the location response moves. a_count is an
   intercept per pre-pitch count: 2026 batters take more in hitters' counts (3-1 swings 51% vs
   54% predicted). Recalibrating the whiff stage leaves foul vs in play (the next stage) as is.

3. **As-used baseline.** Location is Pitching minus Stuff as used, which averages each stage over
   the league's location draws. Left in the 2023-25 response, every 2026 pitch would be charged
   the league-wide change in calls and swings as "location". So each recalibrated stage also gets
   a per-count logit shift on its as-used probability, solved so that its mean change on the 2026
   rows (takes, decisions, swings) equals the recalibration's mean change to the location-aware
   probability there (``asused_delta``).

Only rows in 2026 change: Pitching, Stuff as used and Location. Count-neutral Stuff and the
location-only check are untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASON = 2026
FILE = "abs_2026.json"  # in output/ (the constants dir)
ZONE_REF = Path(r"C:\Users\krbla\claude_sessions\projections\data\interim\zone_ref.parquet")
ZONE_CONVENTIONS = ZONE_REF.with_name("zone_conventions.parquet")
CONVENTION = 2025  # the season the models score 2026 as
ABS_BOT = 0.27  # ABS zone bottom, share of height (for batters without a reference zone)
EDGE_X = 17 / 24 + 0.1208  # ft: plate half-width plus ball radius
CLIP = {"L": (-12.0, 12.0), "z_n": (-1.5, 2.5), "z_abs": (-1.5, 2.5), "xe": (-1.0, 1.5)}
KNOTS = {  # take_outcome: called-strike edges; swing_take: the wider swing surface
    "take_outcome": {
        "z_n": [-0.4, -0.2, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6],
        "z_abs": [-0.4, -0.2, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6],
        "xe": [-0.5, -0.25, -0.12, -0.04, 0.04, 0.12, 0.25, 0.5],
    },
    "swing_take": {
        "z_n": [-0.6, -0.3, -0.1, 0.1, 0.5, 0.9, 1.1, 1.3, 1.6],
        "z_abs": [-0.6, -0.3, -0.1, 0.1, 0.5, 0.9, 1.1, 1.3, 1.6],
        "xe": [-0.5, -0.25, -0.05, 0.1, 0.3, 0.6],
    },
}
KNOTS["swing_outcome"] = KNOTS["swing_take"]
L_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)  # take_outcome's spline in L: knots at these
SWINGS = frozenset(["S", "W", "F", "T", "X", "D", "E"])
TAKES = frozenset(["B", "*B", "C", "H"])
STAGES = {  # model -> (call codes of its first class, the rows it is fitted on)
    "take_outcome": (frozenset(["C"]), TAKES),  # called strike | take
    "swing_take": (SWINGS, SWINGS | TAKES),  # swing | decision
    "swing_outcome": (frozenset(["S", "W", "T"]), SWINGS),  # whiff | swing (foul tips are whiffs)
}
FOLDS = 10  # cross-validation: blocks of consecutive game dates
N_COUNT = 12
RIDGE = 1e-3


# ---- zone -----------------------------------------------------------------------------------
def apply_zone(raw: pd.DataFrame, env: dict) -> pd.DataFrame:
    """Swap the 2026 rows' ABS zone (kept as sz_top_abs / sz_bot_abs) for the batter's
    pose-convention reference zone. A batter without one gets the 2025 convention at the height
    his ABS zone implies (bottom = 27% of height). Data that already carries sz_top_abs (zone
    swapped upstream, e.g. projections' zone.apply) passes unchanged."""
    if "sz_top_abs" in raw:
        return raw
    season = _seasons(raw)
    rows = season == env["season"]
    if not rows.any():
        return raw
    z = env["zone"]
    ref = pd.DataFrame(z["ref"], columns=["batter", "top", "bot"]).set_index("batter")
    out = raw.copy()
    height = raw["sz_bot"].to_numpy(float) / ABS_BOT  # ft
    for side in ("top", "bot"):
        fixed = raw[f"sz_{side}"].to_numpy(float)
        swap = out["batter"].map(ref[side]).to_numpy(float)
        swap = np.where(np.isnan(swap), z[f"{side}_a"] + z[f"{side}_b"] * height, swap)
        out[f"sz_{side}_abs"] = np.where(rows, fixed, np.nan)  # other seasons: no ABS zone
        out[f"sz_{side}"] = np.where(rows, swap, fixed)
    return out


def _seasons(df: pd.DataFrame) -> np.ndarray:
    if "season" in df:
        return df["season"].to_numpy()
    return pd.to_datetime(df["game_date"]).dt.year.to_numpy()


# ---- recalibration --------------------------------------------------------------------------
def rcs(x: np.ndarray, knots) -> np.ndarray:
    """Restricted cubic spline basis (Harrell): x and len(knots) - 2 nonlinear columns."""
    t = np.asarray(knots, float)
    k = len(t)

    def p(u):
        return np.clip(u, 0, None) ** 3

    cols = [x]
    for j in range(k - 2):
        b = (p(x - t[j]) - p(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
             + p(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))  # fmt: skip
        cols.append(b / (t[k - 1] - t[0]) ** 2)
    return np.column_stack(cols)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.clip(np.log(p / (1 - p)), *CLIP["L"])


def inputs(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """z_n (reference zone), z_abs (ABS zone) and xe (ft off the plate edge), clipped, and the
    pre-pitch count index (balls * 3 + strikes)."""
    z, x = df["plate_z"].to_numpy(float), df["plate_x"].to_numpy(float)
    top, bot = df["sz_top_abs"].to_numpy(float), df["sz_bot_abs"].to_numpy(float)
    ref_top, ref_bot = df["sz_top"].to_numpy(float), df["sz_bot"].to_numpy(float)
    raw = {"z_n": (z - ref_bot) / (ref_top - ref_bot), "z_abs": (z - bot) / (top - bot),
           "xe": np.abs(x) - EDGE_X}  # fmt: skip
    count = df["balls"].to_numpy(int) * 3 + df["strikes"].to_numpy(int)
    return {k: np.clip(v, *CLIP[k]) for k, v in raw.items()} | {"count": count}


def design(spec: dict, L: np.ndarray, x: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(design matrix, offset): intercept, count intercepts (0-0 the reference), s(L) when the
    stage re-slopes L, then f(z_n), g(z_abs), h(xe). Without s(L), L is the offset."""
    cols = [np.ones((len(L), 1)), np.eye(N_COUNT)[x["count"]][:, 1:]]
    if spec.get("L_knots"):
        cols.append(rcs(L, spec["L_knots"]))
    cols += [rcs(x[k], spec["knots"][k]) for k in ("z_n", "z_abs", "xe")]
    return np.column_stack(cols), np.zeros(len(L)) if spec.get("L_knots") else L


def adjust(stage: str, P: np.ndarray, df: pd.DataFrame, env: dict | None) -> np.ndarray:
    """One model's location-aware class probabilities (rows x classes, the recalibrated stage's
    class first) with the 2026 recalibration on the rows of ``df`` in the env's season. The other
    classes keep their shares of the rest, i.e. the later sequential stages are unchanged. Other
    rows, and models without a recalibration, pass unchanged."""
    if not env or stage not in env.get("stages", {}):
        return P
    rows = _seasons(df) == env["season"]
    if not rows.any():
        return P
    spec = env["stages"][stage]
    X, off = design(spec, _logit(P[rows, 0]), inputs(df[rows]))
    p = 1 / (1 + np.exp(-(off + X @ np.asarray(spec["coef"]))))
    rest = P[rows, 1:]
    out = np.array(P, dtype=float)
    out[rows, 0] = p
    out[rows, 1:] = rest / rest.sum(axis=1, keepdims=True) * (1 - p)[:, None]
    return out


def asused_delta(stage: str, env: dict | None) -> np.ndarray | None:
    """Per pre-pitch count (balls * 3 + strikes), the logit shift added to the stage's first
    league shift when its as-used (league location draws) probabilities are scored in the env's
    season; None when the stage has none."""
    spec = (env or {}).get("stages", {}).get(stage)
    return np.asarray(spec["asused_delta"]) if spec and "asused_delta" in spec else None


def load(path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


# ---- fitting --------------------------------------------------------------------------------
def _newton(X, y, off, ridge=RIDGE, iters=100) -> np.ndarray:
    """Logistic regression with an offset, ridge on every coefficient but the intercept."""
    b, P = np.zeros(X.shape[1]), ridge * np.eye(X.shape[1])
    P[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(off + X @ b)))
        step = np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X + P, X.T @ (y - p) - P @ b)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


def _date_folds(dates: pd.Series, k: int = FOLDS) -> np.ndarray:
    """Fold per row: k blocks of consecutive game dates."""
    day = pd.to_datetime(dates).dt.normalize()
    rank = day.rank(method="dense").to_numpy() - 1
    return np.floor(rank * k / (rank.max() + 1)).astype(int)


def decisions(raw: pd.DataFrame) -> pd.DataFrame:
    """2026 swing/take decisions (zone already swapped), as combine.py scores them."""
    import dml_stuff_swing as m

    df = m.prepare(raw, m.SWING_TAKE, min_pitches=1)
    return df.assign(model=m.model_names(df))


def stage_scores(df: pd.DataFrame, stage: str):
    """The saved model, its location-aware first-class probability on every row (NaN outside
    its groups), and per group (rows, group model, stuff log-odds, count/hands cells)."""
    import dml_stuff_swing as m

    model = m.load_model(f"output/{stage}_logit_2325.pkl")
    p, groups = np.full(len(df), np.nan), []
    for name, idx in df.groupby("model", observed=True).indices.items():
        if name not in model.groups:
            continue
        gm = model.groups[name]
        sub = df.iloc[idx].reset_index(drop=True)
        sub["season"] = np.minimum(sub["season"], max(gm.seasons))
        D = gm.indices(sub)
        p[idx] = gm.location_probs(sub, D)[1][:, 0]
        groups.append((idx, gm, gm.stuff_logodds(sub, D), m.cell_rows(sub)))
    return model, p, groups


def _asused_first(model, groups, n: int, rows: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """As-used (league location draws) first-class probability on ``rows``, with delta[count]
    added to the first stage's league shift."""
    import dml_stuff_swing as m

    out = np.full(n, np.nan)
    for idx, gm, t, cells in groups:
        for key, cidx in cells.items():
            sel = cidx[rows[idx[cidx]]]
            if len(sel) and key in gm.draws:
                shift = np.array(model.shift, float)
                shift[0] += delta[key[1] * 3 + key[2]]
                out[idx[sel]] = m.class_probs(t[sel], gm.draws[key], shift)[:, 0]
    return out


def fit_asused_delta(model, groups, df: pd.DataFrame, rows: np.ndarray, change: np.ndarray,
                     max_rows: int = 40_000) -> np.ndarray:  # fmt: skip
    """Per count, the first-stage logit shift that moves the mean as-used probability on these
    rows by the recalibration's mean change to the location-aware one (secant method on up to
    ``max_rows`` rows per count)."""
    rng = np.random.default_rng(0)
    count = df["balls"].to_numpy(int) * 3 + df["strikes"].to_numpy(int)
    delta = np.zeros(N_COUNT)
    for c in range(N_COUNT):
        keep = np.flatnonzero(rows & (count == c))
        keep = rng.choice(keep, min(len(keep), max_rows), replace=False)
        mask = np.zeros(len(df), bool)
        mask[keep] = True
        goal = change[keep].mean()

        def moved(x, c=c, mask=mask, keep=keep):
            d = np.zeros(N_COUNT)
            d[c] = x
            return np.mean(_asused_first(model, groups, len(df), mask, d)[keep])

        base = moved(0.0)
        x0, f0, x1 = 0.0, 0.0, 0.1 if goal > 0 else -0.1
        f1 = moved(x1) - base
        for _ in range(6):
            if abs(f1 - goal) < 1e-5 or f1 == f0:
                break
            x0, f0, x1 = x1, f1, x1 + (goal - f1) * (x1 - x0) / (f1 - f0)
            f1 = moved(x1) - base
        delta[c] = x1
    return delta


def fit_stage(df: pd.DataFrame, stage: str) -> tuple[dict, np.ndarray]:
    """The stage's recalibration and its out-of-fold probabilities (blocks of game dates)."""
    L, x, y = _logit(df["p"].to_numpy()), inputs(df), df["y"].to_numpy()
    spec = {"knots": KNOTS[stage]}
    if stage == "take_outcome":
        spec["L_knots"] = np.quantile(L, L_QUANTILES).round(4).tolist()
    X, off = design(spec, L, x)
    folds, oof = _date_folds(df["game_date"]), np.empty(len(y))
    for f in np.unique(folds):
        te = folds == f
        oof[te] = 1 / (1 + np.exp(-(off[te] + X[te] @ _newton(X[~te], y[~te], off[~te]))))
    spec["coef"] = _newton(X, y, off).tolist()
    return spec, oof


def _report(df: pd.DataFrame, stage: str, oof: np.ndarray) -> None:
    y, p0 = df["y"].to_numpy(), df["p"].to_numpy()

    def ll(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    print(f"\n== {stage}: {len(df):,} rows, observed {y.mean():.4f}, as scored {p0.mean():.4f}, "
          f"recalibrated (out of fold) {oof.mean():.4f}")  # fmt: skip
    print(f"log loss: as scored {ll(p0):.5f}, recalibrated (out of fold) {ll(oof):.5f}")
    x = inputs(df)
    bins = {"z_n": [-9, -0.5, -0.3, -0.1, 0.1, 0.3, 0.7, 0.9, 1.1, 1.3, 1.5, 9],
            "xe": [-9, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, 9]}  # fmt: skip
    for k, b in bins.items():
        d = pd.DataFrame({k: pd.cut(x[k], b), "n": 1, "observed": y, "as scored": p0, "oof": oof})
        g = d.groupby(k, observed=True).agg({"n": "sum", "observed": "mean", "as scored": "mean",
                                             "oof": "mean"})  # fmt: skip
        print(g.round(3).to_string())
    d = pd.DataFrame({"count": [f"{c // 3}-{c % 3}" for c in x["count"]], "n": 1, "observed": y,
                      "as scored": p0, "oof": oof})  # fmt: skip
    g = d.groupby("count").agg({"n": "sum", "observed": "mean", "as scored": "mean", "oof": "mean"})
    print(g.round(3).to_string())


def build_env(raw: pd.DataFrame) -> dict:
    """Reference zones (MLB 2026 batters in zone_ref.parquet) and the 2025 convention."""
    refs = pd.read_parquet(ZONE_REF)
    refs = refs[(refs["level"] == "mlb") & (refs["season"] == SEASON)]
    refs = refs[refs["batter"].isin(raw["batter"].unique())]
    conv = pd.read_parquet(ZONE_CONVENTIONS).set_index("season").loc[CONVENTION]
    cols = ["batter", "sz_top_ref", "sz_bot_ref"]
    ref = [[int(b), round(float(t), 4), round(float(s), 4)]
           for b, t, s in refs[cols].itertuples(index=False)]  # fmt: skip
    zone = {k: float(conv[k]) for k in ("top_a", "top_b", "bot_a", "bot_b")} | {"ref": ref}
    return {"season": SEASON, "zone": zone, "stages": {}}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=f"data/statfast_{SEASON}.parquet")
    ap.add_argument("--out", default=f"output/{FILE}")
    a = ap.parse_args()
    raw = pd.read_parquet(a.data)
    raw = raw[_seasons(raw) == SEASON]
    env = build_env(raw)
    raw = apply_zone(raw, env)
    print(f"{raw['batter'].nunique():,} batters, {len(env['zone']['ref']):,} with a reference zone")
    df = decisions(raw)
    for stage, (codes, eligible) in STAGES.items():
        model, p, groups = stage_scores(df, stage)
        rows = df["call_code"].isin(list(eligible)).to_numpy() & np.isfinite(p)
        sdf = df[rows].assign(p=p[rows], y=df["call_code"][rows].isin(list(codes)).astype(float))
        sdf = sdf.reset_index(drop=True)
        spec, oof = fit_stage(sdf, stage)
        _report(sdf, stage, oof)
        X, off = design(spec, _logit(sdf["p"].to_numpy()), inputs(sdf))
        change = np.full(len(df), np.nan)
        refit = 1 / (1 + np.exp(-(off + X @ np.asarray(spec["coef"]))))
        change[rows] = refit - sdf["p"].to_numpy()
        spec["asused_delta"] = fit_asused_delta(model, groups, df, rows, change).round(5).tolist()
        print("as-used first-stage shift by count (0-0 ... 3-2):", spec["asused_delta"])
        env["stages"][stage] = spec
    Path(a.out).write_text(json.dumps(env, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
