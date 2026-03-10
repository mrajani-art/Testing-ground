"""
Recency weighting for ML model training.

Gives heavier weight to recent quotes (last 90 days) using exponential decay,
so the model prioritizes current market pricing while still learning from
historical data.

Usage in model_training.py:
    from src.ml.recency_weights import compute_recency_weights_from_df
    weights = compute_recency_weights_from_df(df, date_column="created_at")
    model.fit(X_train, y_train, sample_weight=w_train)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone


def compute_recency_weights(
    dates: pd.Series,
    recent_days: int = 90,
    recent_weight: float = 3.0,
    decay_half_life: int = 180,
    min_weight: float = 0.2,
    reference_date=None,
) -> np.ndarray:
    """
    Compute sample weights that emphasize recent quotes.

    Strategy:
    - Quotes within `recent_days` get a flat boost of `recent_weight`
    - Older quotes decay exponentially with `decay_half_life`
    - No quote drops below `min_weight`

    Parameters
    ----------
    dates : pd.Series
        Timestamps for each training sample. Can be datetime, date strings,
        or NaT (missing dates get min_weight).
    recent_days : int
        Number of days considered "recent" (default 90).
    recent_weight : float
        Weight multiplier for recent quotes (default 3.0 = 3x base).
    decay_half_life : int
        Days after which an old quote's weight is halved (default 180).
    min_weight : float
        Floor weight so no sample is entirely ignored (default 0.2).
    reference_date : datetime, optional
        "Now" for age calculations. Defaults to utcnow().

    Returns
    -------
    np.ndarray
        Array of weights, same length as `dates`.
    """
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)

    # Ensure tz-aware UTC for consistent comparison
    dates_dt = pd.to_datetime(dates, errors="coerce", utc=True)

    # Age in days (NaT → NaN)
    age_days = (reference_date - dates_dt).dt.total_seconds() / 86400.0

    weights = np.full(len(dates), min_weight, dtype=float)

    for i, age in enumerate(age_days):
        if pd.isna(age):
            # Missing date — floor weight, already set
            continue
        elif age <= recent_days:
            # Recent — full boost
            weights[i] = recent_weight
        else:
            # Exponential decay: weight = recent_weight * 0.5^((age - recent_days) / half_life)
            excess_age = age - recent_days
            decay = 0.5 ** (excess_age / decay_half_life)
            weights[i] = max(recent_weight * decay, min_weight)

    return weights


def compute_recency_weights_from_df(
    df: pd.DataFrame,
    date_column: str = "created_at",
    recent_days: int = 90,
    recent_weight: float = 3.0,
    decay_half_life: int = 180,
    min_weight: float = 0.2,
    reference_date=None,
) -> np.ndarray:
    """
    Convenience wrapper — extracts date column from DataFrame.

    Falls back to uniform weights (all 1.0) if the date column is missing
    or entirely null, so training still works with legacy/demo data.

    Called by model_training.py as:
        weights = compute_recency_weights_from_df(
            df,
            date_column="created_at",
            recent_days=RECENCY_RECENT_DAYS,
            recent_weight=RECENCY_RECENT_WEIGHT,
            decay_half_life=RECENCY_DECAY_HALF_LIFE,
            min_weight=RECENCY_MIN_WEIGHT,
        )
    """
    if date_column not in df.columns:
        return np.ones(len(df), dtype=float)

    dates = df[date_column]

    if dates.isna().all():
        return np.ones(len(df), dtype=float)

    return compute_recency_weights(
        dates,
        recent_days=recent_days,
        recent_weight=recent_weight,
        decay_half_life=decay_half_life,
        min_weight=min_weight,
        reference_date=reference_date,
    )
