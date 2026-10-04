"""Feature pipeline of the alert predictor (Melnyk et al., IEEE Access, 2026), copied unchanged from
experiments/proposed_model.py of that work's repository (load_data ... get_feature_columns). Used by p3_features.py
for the unanchored forecast of the paper."""
import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load and expand alert events into hourly active/inactive indicators.

    Each row in the CSV is an alert *event* with a start and end timestamp.
    Correct preprocessing marks every hour that overlaps with an active alert
    as alert_occurred=1, not just the start hour.  Counting only the start
    hour (floor(started_at)) causes severe under-counting because many alerts
    span multiple hours, collapsing multi-hour autocorrelation and making the
    task much harder than it actually is.
    """
    df = pd.read_csv(path, parse_dates=["started_at", "finished_at"])
    oblast_df = df[df["level"] == "oblast"].copy()

    rows = []
    for _, row in oblast_df.iterrows():
        start = row["started_at"].floor("h")
        end   = row["finished_at"].floor("h")
        for h in pd.date_range(start, end, freq="h"):
            rows.append({"oblast": row["oblast"], "hour": h})

    expanded = pd.DataFrame(rows)
    # Deduplicate: multiple overlapping alerts in the same region-hour → still 1
    hourly = (
        expanded.groupby(["oblast", "hour"])
        .size()
        .reset_index(name="n")
        .assign(alert_occurred=1)[["oblast", "hour", "alert_occurred"]]
    )

    all_oblasts = oblast_df["oblast"].unique()
    full_range  = pd.date_range(hourly["hour"].min(), hourly["hour"].max(), freq="h")
    idx = pd.MultiIndex.from_product(
        [all_oblasts, full_range], names=["oblast", "hour"]
    )
    full_df = pd.DataFrame(index=idx).reset_index()
    full_df = full_df.merge(
        hourly[["oblast", "hour", "alert_occurred"]],
        on=["oblast", "hour"],
        how="left",
    )
    full_df["alert_occurred"] = full_df["alert_occurred"].fillna(0).astype(int)
    full_df = full_df.sort_values(["oblast", "hour"]).reset_index(drop=True)
    return full_df


# ──────────────────────────────────────────────────────────────────────────────
# Neighbour graph (shared land borders, administrative geography — simplified)
# ──────────────────────────────────────────────────────────────────────────────

OBLAST_NEIGHBORS = {
    "Івано-Франківська область": [
        "Львівська область", "Тернопільська область", "Закарпатська область",
        "Чернівецька область",
    ],
    "Волинська область": ["Рівненська область", "Львівська область"],
    "Вінницька область": [
        "Житомирська область", "Київська область", "Черкаська область",
        "Хмельницька область", "Тернопільська область", "Кіровоградська область",
        "Одеська область",
    ],
    "Дніпропетровська область": [
        "Запорізька область", "Донецька область", "Харківська область",
        "Полтавська область", "Кіровоградська область", "Миколаївська область",
    ],
    "Донецька область": [
        "Луганська область", "Дніпропетровська область", "Запорізька область",
    ],
    "Житомирська область": [
        "Волинська область", "Рівненська область", "Київська область",
        "Вінницька область", "Чернігівська область",
    ],
    "Закарпатська область": [
        "Львівська область", "Івано-Франківська область",
    ],
    "Запорізька область": [
        "Дніпропетровська область", "Донецька область", "Херсонська область",
    ],
    "Київська область": [
        "м. Київ", "Житомирська область", "Вінницька область", "Черкаська область",
        "Полтавська область", "Чернігівська область",
    ],
    "Кіровоградська область": [
        "Вінницька область", "Черкаська область", "Полтавська область",
        "Дніпропетровська область", "Миколаївська область", "Одеська область",
    ],
    "Луганська область": ["Донецька область", "Харківська область"],
    "Львівська область": [
        "Волинська область", "Рівненська область", "Тернопільська область",
        "Івано-Франківська область", "Закарпатська область",
    ],
    "Миколаївська область": [
        "Одеська область", "Кіровоградська область", "Дніпропетровська область",
        "Херсонська область",
    ],
    "Одеська область": [
        "Миколаївська область", "Кіровоградська область", "Вінницька область",
    ],
    "Полтавська область": [
        "Київська область", "Черкаська область", "Кіровоградська область",
        "Харківська область", "Сумська область",
    ],
    "Рівненська область": [
        "Волинська область", "Житомирська область", "Тернопільська область",
        "Хмельницька область",
    ],
    "Сумська область": ["Чернігівська область", "Полтавська область", "Харківська область"],
    "Тернопільська область": [
        "Рівненська область", "Львівська область", "Івано-Франківська область",
        "Хмельницька область", "Вінницька область",
    ],
    "Харківська область": [
        "Сумська область", "Полтавська область", "Дніпропетровська область",
        "Донецька область", "Луганська область",
    ],
    "Херсонська область": [
        "Миколаївська область", "Дніпропетровська область", "Запорізька область",
    ],
    "Хмельницька область": [
        "Рівненська область", "Тернопільська область", "Вінницька область",
        "Чернівецька область",
    ],
    "Черкаська область": [
        "Київська область", "Полтавська область", "Кіровоградська область",
        "Вінницька область",
    ],
    "Чернівецька область": [
        "Івано-Франківська область", "Тернопільська область", "Хмельницька область",
    ],
    "Чернігівська область": [
        "Київська область", "Сумська область", "Житомирська область",
    ],
    "м. Київ": ["Київська область"],
}


def _add_neighbor_spillover_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (hour, oblast), sum alert_occurred over *adjacent* oblasts only
    (same clock hour).  Adds lag and short rolling views of that signal.
    """
    wide = (
        df.pivot_table(index="hour", columns="oblast", values="alert_occurred",
                       aggfunc="max")
        .fillna(0)
    )
    cols = list(wide.columns)
    neighbor_sum = pd.DataFrame(0.0, index=wide.index, columns=cols, dtype=np.float64)
    neighbor_deg = pd.DataFrame(1.0, index=wide.index, columns=cols, dtype=np.float64)
    for o in cols:
        nbrs = [n for n in OBLAST_NEIGHBORS.get(o, []) if n in wide.columns]
        if nbrs:
            neighbor_sum[o] = wide[nbrs].sum(axis=1).values
            neighbor_deg[o] = float(len(nbrs))
    stacked_sum = neighbor_sum.stack().reset_index()
    stacked_sum.columns = ["hour", "oblast", "neighbor_alerts_now"]
    stacked_deg = neighbor_deg.stack().reset_index()
    stacked_deg.columns = ["hour", "oblast", "neighbor_graph_degree"]
    spill = stacked_sum.merge(stacked_deg, on=["hour", "oblast"])
    spill["neighbor_rate_now"] = (
        spill["neighbor_alerts_now"] / spill["neighbor_graph_degree"].clip(lower=1.0)
    )
    df = df.merge(
        spill[["hour", "oblast", "neighbor_alerts_now", "neighbor_rate_now"]],
        on=["hour", "oblast"],
        how="left",
    )
    df["neighbor_alerts_now"] = df["neighbor_alerts_now"].fillna(0).astype(np.float32)
    df["neighbor_rate_now"] = df["neighbor_rate_now"].fillna(0).astype(np.float32)

    df = df.sort_values(["oblast", "hour"]).reset_index(drop=True)
    df["neighbor_alerts_lag1h"] = (
        df.groupby("oblast")["neighbor_alerts_now"].shift(1).fillna(0).astype(np.float32)
    )
    df["neighbor_alerts_roll3h"] = (
        df.groupby("oblast")["neighbor_alerts_now"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
        .fillna(0)
        .astype(np.float32)
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering  (identical feature set shared with all baseline models)
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Temporal ---
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["month"]       = df["hour"].dt.month
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"]    = df["hour_of_day"].isin(
        list(range(0, 6)) + list(range(22, 24))
    ).astype(int)

    # --- Cyclical encoding ---
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["day_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Current alert status (lag_0h) ---
    # Valid for 1-hour-ahead prediction: at time t we know the current alert
    # status and use it to predict t+1.  This is the single most informative
    # feature due to the multi-hour duration of alerts.
    df["lag_0h"] = df["alert_occurred"].astype(float)

    # --- Lag features (24 h lookback) ---
    for i in range(1, 25):
        df[f"lag_{i}h"] = (
            df.groupby("oblast")["alert_occurred"].shift(i).fillna(0)
        )

    # --- Rolling statistics over multiple windows (including current hour) ---
    # No shift(1): at prediction time t all data up to and including t is known.
    for w in [3, 6, 12, 24, 48, 72]:
        df[f"rate_{w}h"] = (
            df.groupby("oblast")["alert_occurred"]
            .transform(lambda x: x.rolling(w, min_periods=1).mean())
            .fillna(0)
        )
        df[f"count_{w}h"] = (
            df.groupby("oblast")["alert_occurred"]
            .transform(lambda x: x.rolling(w, min_periods=1).sum())
            .fillna(0)
        )
        df[f"std_{w}h"] = (
            df.groupby("oblast")["alert_occurred"]
            .transform(lambda x: x.rolling(w, min_periods=1).std())
            .fillna(0)
        )

    # --- Momentum features ---
    df["momentum_3_12"]  = df["rate_3h"]  - df["rate_12h"]
    df["momentum_6_24"]  = df["rate_6h"]  - df["rate_24h"]
    df["momentum_12_72"] = df["rate_12h"] - df["rate_72h"]

    # --- Volatility / relative rate ---
    global_rate = df.groupby("oblast")["alert_occurred"].transform("mean")
    df["volatility_24h"] = (
        df.groupby("oblast")["alert_occurred"]
        .transform(lambda x: x.rolling(24, min_periods=1).std())
        .fillna(0)
    )
    df["rel_rate_24h"] = df["rate_24h"] / (global_rate + 1e-6)

    # --- Alert state-transition features ---
    # These capture alert momentum far more directly than binary lag indicators.
    # alert_duration:        consecutive hours in current alert (0 when no alert)
    # hours_since_last_alert: hours elapsed since last alert ended (0 when in alert)
    # Both use the "run-length within group" trick: group by change-points, then
    # count position within each constant run.
    def _run_length(s):
        """Position (1-indexed) of each element within its constant run."""
        run_id = (s != s.shift()).cumsum()
        return s.groupby(run_id).cumcount() + 1

    df["alert_duration"] = (
        df.groupby("oblast")["alert_occurred"]
        .transform(lambda s: s * _run_length(s))
        .fillna(0)
    )
    inv = df["alert_occurred"].map({1: 0, 0: 1}).fillna(0)
    df["_inv_alert"] = inv
    df["hours_since_last_alert"] = (
        df.groupby("oblast")["_inv_alert"]
        .transform(lambda s: s * _run_length(s))
        .fillna(0)
    )
    df.drop(columns=["_inv_alert"], inplace=True)

    # --- Cross-region spatial features ---
    # At each hour, count how many OTHER regions currently have an alert.
    # This captures spatial propagation: if many regions are under alert,
    # this region is more likely to be next.  CatBoost/LightGBM cannot
    # learn this from per-region features alone.
    alerts_per_hour = df.groupby("hour")["alert_occurred"].transform("sum")
    df["other_regions_alert"] = alerts_per_hour - df["alert_occurred"]
    n_regions = df["oblast"].nunique()
    df["national_alert_rate"] = df["other_regions_alert"] / max(n_regions - 1, 1)

    # Rolling national alert momentum (was the country-wide alert level rising?)
    df["national_rate_3h"] = (
        df.groupby("oblast")["national_alert_rate"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
        .fillna(0)
    )
    df["national_rate_12h"] = (
        df.groupby("oblast")["national_alert_rate"]
        .transform(lambda x: x.rolling(12, min_periods=1).mean())
        .fillna(0)
    )
    df["national_momentum"] = df["national_rate_3h"] - df["national_rate_12h"]

    # --- Adjacency-weighted neighbour spillover (same hour, bordering oblasts) ---
    df = _add_neighbor_spillover_features(df)

    # --- Oblast identity (full frame before split → stable dummy columns) ---
    oblast_dummies = pd.get_dummies(df["oblast"], prefix="oblast", dtype=np.float32)
    df = pd.concat([df, oblast_dummies], axis=1)

    # --- Target: next-hour alert ---
    df["target"] = df.groupby("oblast")["alert_occurred"].shift(-1)

    return df


BASE_FEATURE_COLS = (
    ["lag_0h"]
    + ["hour_of_day", "day_of_week", "month", "is_weekend", "is_night",
       "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"]
    + [f"lag_{i}h" for i in range(1, 25)]
    + [f"rate_{w}h"  for w in [3, 6, 12, 24, 48, 72]]
    + [f"count_{w}h" for w in [3, 6, 12, 24, 48, 72]]
    + [f"std_{w}h"   for w in [3, 6, 12, 24, 48, 72]]
    + ["momentum_3_12", "momentum_6_24", "momentum_12_72",
       "volatility_24h", "rel_rate_24h"]
    + ["alert_duration", "hours_since_last_alert"]
    + ["other_regions_alert", "national_alert_rate",
       "national_rate_3h", "national_rate_12h", "national_momentum"]
)

NEIGHBOR_FEATURE_COLS = (
    "neighbor_alerts_now",
    "neighbor_rate_now",
    "neighbor_alerts_lag1h",
    "neighbor_alerts_roll3h",
)

# Continuous / count features — StandardScaler; oblast one-hots are passed through.
NUMERIC_FEATURE_COLS = tuple(BASE_FEATURE_COLS) + NEIGHBOR_FEATURE_COLS


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric (scaled) columns + sorted oblast_* dummies present in ``df``."""
    oblast_cols = sorted(c for c in df.columns if c.startswith("oblast_"))
    return list(NUMERIC_FEATURE_COLS) + oblast_cols


# ──────────────────────────────────────────────────────────────────────────────
# Temporal split and array preparation
# ──────────────────────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame):
    """Strict 70/15/15 split on time axis — no shuffling, no data leakage."""
    hours   = df["hour"].sort_values().unique()
    n       = len(hours)
    train_end = hours[int(n * 0.70)]
    val_end   = hours[int(n * 0.85)]
    train = df[df["hour"] <  train_end]
    val   = df[(df["hour"] >= train_end) & (df["hour"] < val_end)]
    test  = df[df["hour"] >= val_end]
    return train, val, test
