"""Pitch groups (Fastball / Breaking / Offspeed / Other) from the Level 1 pitch classifier.

1. pitch_l1.classify (pitch_l1_v1.npz), arm-adjusted field only: the blend weight on
   the raw HB/IVB field is forced to 0 (cap = 0), so each pitch's family is the argmax of its
   arm-adjusted membership {horz_arm_adj, vert_arm_adj, dspeed}, and the group of a
   pitcher-game-pitch type is its most common family. Level 1's "Breaking Ball" is named
   "Breaking" here, as in the model names.
2. A pitcher-game-pitch type the classifier leaves Unassigned (no pitch in it has an
   arm-adjusted membership, e.g. no hb/ivb) goes by its Statcast pitch type:
       Fastball  FF SI
       Breaking  SL FC SV ST KC CU CS
       Offspeed  CH FS FO
       Other     anything else

The classifier needs statsapi's hb and ivb (statfast columns) and works per pitcher-game, so
pass whole outings. Without hb/ivb every pitch falls back to step 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pitch_l1

GROUPS = ("Fastball", "Breaking", "Offspeed")
OTHER = "Other"
L1_NAMES = {"Fastball": "Fastball", "Breaking Ball": "Breaking", "Offspeed": "Offspeed"}
FALLBACK = {
    "Fastball": frozenset(["FF", "SI"]),
    "Breaking": frozenset(["SL", "FC", "SV", "ST", "KC", "CU", "CS"]),
    "Offspeed": frozenset(["CH", "FS", "FO"]),
}

HERE = Path(__file__).resolve().parent
MODEL_FILE = "pitch_l1_v1.npz"  # models/ in the pitch-modeling repo, output/ in the project
_MODEL: pitch_l1.Model | None = None


def model() -> pitch_l1.Model:
    """The frozen Level 1 model with the raw-movement blend switched off (arm-adjusted only)."""
    global _MODEL
    if _MODEL is None:
        path = next((p for p in (HERE / "models" / MODEL_FILE, HERE / "output" / MODEL_FILE)
                     if p.exists()), pitch_l1.MODEL)  # fmt: skip
        _MODEL = pitch_l1.Model(path)
        _MODEL.params["cap"] = 0.0  # w = min(cap, s * H) = 0: arm-adjusted field only
    return _MODEL


def fallback(pt: pd.Series) -> pd.Series:
    """Group by Statcast pitch type (step 2)."""
    out = pd.Series(OTHER, index=pt.index, dtype="string")
    for name, types in FALLBACK.items():
        out[pt.isin(types).fillna(False).to_numpy(bool)] = name
    return out


def level1(df: pd.DataFrame) -> np.ndarray:
    """Level 1 family per row (Fastball / Breaking / Offspeed / Unassigned), in row order."""
    if not {"hb", "ivb"} <= set(df.columns):
        print("pitch_groups: no hb/ivb columns, grouping by pitch type only", file=sys.stderr)
        return np.full(len(df), "Unassigned", dtype=object)
    d = df.reset_index(drop=True)
    if "release_pos_y" not in d:
        d = d.assign(release_pos_y=50.0)  # statfast's fit plane
    if "game_date" not in d:
        d = d.assign(game_date=d["season"].astype(int).astype(str) + "-07-01")
    out = pitch_l1.classify(d[pitch_l1.REQUIRED], model())
    return out["l1"].map(lambda v: L1_NAMES.get(v, "Unassigned")).to_numpy(object)


def assign(df: pd.DataFrame, pt: str = "pitch_type") -> pd.Series:
    """Pitch group per row of ``df`` (indexed like df): Level 1, then the pitch-type fallback."""
    l1 = pd.Series(level1(df), index=df.index, dtype="string")
    types = df[pt].astype("string")
    return l1.where(l1 != "Unassigned", fallback(types)).astype("string")
