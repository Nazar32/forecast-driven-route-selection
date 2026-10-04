"""
Paper 3: reuse the EXACT Paper-2 feature pipeline without importing torch/catboost.

The functions load_data / engineer_features / temporal_split and the feature-column
constants come from paper2_features.py, a copy of the predictor's code (single source of
truth), so the feature matrix is identical to the one used in Paper 2.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper2_features import (NUMERIC_FEATURE_COLS, engineer_features, get_feature_columns,  # noqa: E402
                             load_data, temporal_split)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "alerts" / "official_data_uk_2026-07.csv.gz"


V1_CACHE = ROOT / "results" / "p3" / "features.parquet"


def fix_rel_rate(df, cutoff):
    """rel_rate_24h = rate_24h / mean alert rate of the oblast BEFORE `cutoff` (v2).
    The Paper-2 code divides by the full-sample mean, which leaks test-period statistics."""
    m = df[df["hour"] < cutoff].groupby("oblast")["alert_occurred"].mean()
    df = df.copy()
    df["rel_rate_24h"] = (df["rate_24h"] / (df["oblast"].map(m).astype(np.float64) + 1e-6)).astype(np.float32)
    return df


def build_frame(cache=V1_CACHE, cutoff="train"):
    """Paper-2 feature frame; v2: rel_rate_24h normalised by pre-`cutoff` means
    (cutoff='train' -> start of the validation block of temporal_split; a Timestamp -> that hour;
    None -> original full-sample normalisation)."""
    cache = Path(cache)
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        raw = load_data(str(DATA))
        df = engineer_features(raw)
        num = [c for c in df.columns if df[c].dtype == np.float64]
        df[num] = df[num].astype(np.float32)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    end = os.environ.get("P3_END")
    if end:                                   # v4: frame restricted to the oblast-declaration period
        df = df[df["hour"] <= pd.Timestamp(end, tz="UTC")].reset_index(drop=True)
    if cutoff is None:
        return df
    if isinstance(cutoff, str) and cutoff == "train":
        hours = df["hour"].sort_values().unique()
        cutoff = hours[int(len(hours) * 0.70)]
    return fix_rel_rate(df, cutoff)
