"""
scripts/train_models.py
=======================
Pulls training data from Supabase and trains ML models for all vendors.

Usage (from repo root):
    python scripts/train_models.py

What it does:
    1. Loads quotes + quote_prices from Supabase for dazpak, ross, internal
    2. Trains QuoteModelTrainer for each vendor
    3. Saves .joblib files to models/
    4. Prints a performance summary table

Run this whenever you have 20+ new quotes in Supabase beyond the last training set.
Tedpack is trained separately via TEDPACK_ML_V2.ipynb in Colab.

Requirements (in requirements.txt):
    supabase, scikit-learn, joblib, pandas, numpy
"""

import sys
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup (run from repo root) ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supabase import create_client
from config.settings import (
    QUANTITY_TIERS,
    SUBSTRATE_OPTIONS,
    FINISH_OPTIONS,
)
from src.ml.model_training import QuoteModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Supabase credentials (set as env vars or paste directly for local use) ───
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://dernxirzvawjmdxzxefl.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")   # set SUPABASE_KEY env var

if not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_KEY not set. Export it as an environment variable:\n"
        "  export SUPABASE_KEY='your-anon-key-here'"
    )

# ── Vendor training config ────────────────────────────────────────────────────
VENDOR_CONFIG = {
    "dazpak": {
        "use_log_target": False,
        "qty_filter": (10_000, 1_000_000),
        "price_filter": (0.05, 2.00),
    },
    "ross": {
        "use_log_target": False,
        "qty_filter": (1_000, 500_000),
        "price_filter": (0.05, 2.00),
    },
    "internal": {
        "use_log_target": True,   # wide price range benefits from log
        "qty_filter": (500, 200_000),
        "price_filter": (0.03, 5.00),
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_vendor_data(supabase, vendor: str) -> pd.DataFrame:
    """
    Fetch all quotes + price tiers for a vendor from Supabase.
    Returns a flat DataFrame with one row per quote×quantity.
    """
    logger.info(f"Fetching {vendor} quotes from Supabase...")

    # ── Fetch quotes (spec data) ──────────────────────────────────────────
    quotes_resp = (
        supabase.table("quotes")
        .select(
            "id, fl_number, quote_date, width, height, gusset, "
            "substrate, finish, seal_type, gusset_type, "
            "zipper, corner_treatment, embellishment, num_skus, "
            "print_method, created_at"
        )
        .eq("vendor", vendor)
        .execute()
    )
    quotes_df = pd.DataFrame(quotes_resp.data)

    if quotes_df.empty:
        logger.warning(f"  No quotes found for {vendor}")
        return pd.DataFrame()

    logger.info(f"  {len(quotes_df)} quotes loaded")

    # ── Fetch price tiers in chunks (Supabase IN limit = 100) ────────────
    quote_ids = quotes_df["id"].tolist()
    all_prices = []

    for i in range(0, len(quote_ids), 100):
        chunk = quote_ids[i : i + 100]
        resp = (
            supabase.table("quote_prices")
            .select("quote_id, quantity, unit_price")
            .in_("quote_id", chunk)
            .execute()
        )
        all_prices.extend(resp.data)

    prices_df = pd.DataFrame(all_prices)
    logger.info(f"  {len(prices_df)} price tiers loaded")

    # ── Join ──────────────────────────────────────────────────────────────
    df = prices_df.merge(quotes_df, left_on="quote_id", right_on="id", how="inner")

    return df


def clean_vendor_data(df: pd.DataFrame, vendor: str) -> pd.DataFrame:
    """
    Type-cast, rename, and filter a raw vendor DataFrame into a
    clean training set that matches what prepare_features() expects.
    """
    cfg = VENDOR_CONFIG[vendor]

    # ── Rename to match feature engineering expectations ──────────────────
    df = df.rename(columns={
        "unit_price":  "unit_price",
        "num_skus":    "skus_in_quote",
        "fl_number":   "fl_number",
    })

    # ── Numeric casts ─────────────────────────────────────────────────────
    for col in ["width", "height", "gusset", "quantity", "unit_price", "skus_in_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["gusset"] = df["gusset"].fillna(0.0)
    df["skus_in_quote"] = df["skus_in_quote"].fillna(1).astype(int)
    df["quote_date"] = pd.to_datetime(df["quote_date"], errors="coerce")
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    # ── String casts & defaults ────────────────────────────────────────────
    df["substrate"] = df["substrate"].fillna("MET PET").str.strip()
    df["finish"] = df["finish"].fillna("Matte").str.strip()
    df["seal_type"] = df["seal_type"].fillna("Stand Up Pouch").str.strip()
    df["gusset_type"] = df["gusset_type"].fillna("None").str.strip()
    df["zipper"] = df["zipper"].fillna("No Zipper").str.strip()
    df["corner_treatment"] = df["corner_treatment"].fillna("Straight").str.strip()
    df["embellishment"] = df["embellishment"].fillna("None").str.strip()
    df["print_method"] = df["print_method"].fillna(
        "Flexographic" if vendor == "dazpak" else "Digital"
    ).str.strip()

    # ── Quality filters ───────────────────────────────────────────────────
    qty_lo, qty_hi = cfg["qty_filter"]
    price_lo, price_hi = cfg["price_filter"]

    before = len(df)
    df = df[
        df["unit_price"].notna() &
        df["unit_price"].between(price_lo, price_hi) &
        df["quantity"].notna() &
        df["quantity"].between(qty_lo, qty_hi) &
        df["width"].notna() &
        df["height"].notna()
    ].copy().reset_index(drop=True)

    logger.info(
        f"  Cleaned {vendor}: {before} → {len(df)} rows "
        f"(removed {before - len(df)} outliers/nulls)"
    )

    if len(df) < 20:
        logger.warning(
            f"  ⚠ Only {len(df)} training rows for {vendor} — "
            f"model may be unreliable. Need 50+ for good predictions."
        )

    return df


# ═════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def train_all_models():
    """Train Dazpak, Ross, and Internal models and save to models/."""

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info(f"Connected to Supabase: {SUPABASE_URL}")

    results = {}

    for vendor, cfg in VENDOR_CONFIG.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  TRAINING: {vendor.upper()}")
        logger.info(f"{'='*60}")

        # ── Load + clean data ─────────────────────────────────────────────
        raw_df = load_vendor_data(supabase, vendor)
        if raw_df.empty:
            logger.warning(f"  Skipping {vendor} — no data")
            results[vendor] = {"status": "skipped", "reason": "no data"}
            continue

        df = clean_vendor_data(raw_df, vendor)
        if len(df) < 10:
            logger.warning(f"  Skipping {vendor} — insufficient clean data ({len(df)} rows)")
            results[vendor] = {"status": "skipped", "reason": f"only {len(df)} clean rows"}
            continue

        # ── Train ─────────────────────────────────────────────────────────
        try:
            trainer = QuoteModelTrainer(
                vendor=vendor,
                use_log_target=cfg["use_log_target"],
            )
            metrics = trainer.train(df, target_col="unit_price")

            # ── Save ──────────────────────────────────────────────────────
            trainer.save(vendor)
            logger.info(
                f"  ✓ Saved models/{vendor}_point.joblib | "
                f"MAPE={metrics.get('mape', 0)*100:.1f}% | "
                f"R²={metrics.get('r2', 0):.3f} | "
                f"n={metrics.get('n_samples', len(df))}"
            )
            results[vendor] = {"status": "trained", "metrics": metrics}

        except Exception as e:
            logger.error(f"  ✗ Training failed for {vendor}: {e}", exc_info=True)
            results[vendor] = {"status": "error", "reason": str(e)}

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("  TRAINING SUMMARY")
    logger.info(f"{'='*60}")

    for vendor, result in results.items():
        status = result["status"]
        if status == "trained":
            m = result["metrics"]
            mape = m.get("mape", 0) * 100
            r2 = m.get("r2", 0)
            n = m.get("n_samples", "?")
            flag = "✓" if mape < 15 else "⚠"
            logger.info(f"  {flag} {vendor:<10} MAPE={mape:.1f}%  R²={r2:.3f}  n={n}")
        elif status == "skipped":
            logger.info(f"  — {vendor:<10} SKIPPED: {result['reason']}")
        else:
            logger.info(f"  ✗ {vendor:<10} ERROR: {result['reason']}")

    logger.info("\nDone. Commit the new .joblib files in models/ to GitHub.")
    logger.info("Tedpack models are trained separately in TEDPACK_ML_V2.ipynb")


if __name__ == "__main__":
    train_all_models()
