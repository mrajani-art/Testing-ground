"""
Feature engineering for the packaging quote ML models.

Transforms raw quote specifications into numeric feature vectors
suitable for gradient boosting models.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
import joblib

from config.settings import (
    ROSS_CONVERTING_FLAT_RATE,
    ROSS_ZIPPER_COST_MSI,
    ROSS_ZIPPER_WIDTH_IN,
    ROSS_HP200K_CLICK_CMYOVG,
    ROSS_HP200K_CLICK_WHITE,
    ROSS_HP200K_PRIMING_MSI,
    ROSS_HP200K_SETUP_HRS,
    ROSS_HP200K_RATE_PER_HR,
    ROSS_HP200K_SPOILAGE_PCT,
    ROSS_GONDERFLEX_SPOILAGE_TABLE,
)

logger = logging.getLogger(__name__)


# ── Feature Definitions ────────────────────────────────────────────
# These match the actual spreadsheet + PDF data fields.

NUMERIC_FEATURES = [
    "width",           # inches
    "height",          # inches
    "gusset",          # inches (0 if none)
    "print_width",     # calculated: height × 2 + gusset
    "bag_area_sqin",   # calculated: width × height
    "quantity",        # order quantity for this tier
    "log_quantity",    # log10(quantity) — captures volume discount curve
    "inv_quantity",    # 1/quantity — captures fixed-cost amortization
]

CATEGORICAL_FEATURES = [
    "substrate",
    "finish",
    "fill_style",
    "seal_type",
    "gusset_type",
    "zipper",
    "tear_notch",
    "hole_punch",
    "corner_treatment",
    "embellishment",
]

# Ordinal encoding order (from least to most "complex/expensive")
CATEGORY_ORDERS = {
    "substrate": ["CLR_PET", "MET_PET", "WHT_MET_PET", "HB_CLR_PET", "CUSTOM"],
    "finish": ["None", "Matte Laminate", "Gloss Laminate", "Soft Touch Laminate", "Holographic"],
    "fill_style": ["Top", "Bottom"],
    "seal_type": ["Stand Up", "2 Side Seal", "3 Side Seal", "3 Side Top Fill"],
    "gusset_type": ["None", "K Seal", "K Seal & Skirt Seal", "Plow Bottom", "Flat Bottom / Side Gusset"],
    "zipper": ["No Zipper", "Single Profile Non-CR", "Double Profile Non-CR",
               "Standard CR", "CR Zipper", "Presto CR Zipper"],
    "tear_notch": ["None", "Standard", "Double (2)"],
    "hole_punch": ["None", "Standard", "Round (Butterfly)", "Euro Slot", "Sombrero"],
    "corner_treatment": ["Straight", "Rounded"],
    "embellishment": ["None", "Hot Stamp (Gold)", "Hot Stamp (Silver)", "Embossing", "Spot UV"],
}


def normalize_substrate(val: str) -> str:
    """Map raw substrate names to canonical keys."""
    if not isinstance(val, str):
        return "CUSTOM"
    v = val.strip().upper().replace(' ', '_')
    if 'HB' in v or 'HIGH_BARRIER' in v:
        return "HB_CLR_PET"
    if 'WHT' in v or 'WHITE' in v:
        return "WHT_MET_PET"
    if 'CLR' in v or 'CLEAR' in v:
        return "CLR_PET"
    if 'MET' in v:
        return "MET_PET"
    return "CUSTOM"


# ── Ross Cost Structure Helpers ────────────────────────────────────
# These derive features from Ross's known equipment standards.
# They approximate Ross's cost *structure* (not actual cost, since
# material cost and margin are unknown) to give the GBR model
# physically meaningful signals about price drivers.

def _ross_converting_cost(width: float, has_zipper: bool) -> float:
    """
    Estimate Ross per-unit converting cost.
    Flat $0.055/pouch + zipper material if applicable.
    Zipper: $5.258772/MSI at 0.95" width.
    """
    cost = ROSS_CONVERTING_FLAT_RATE
    if has_zipper:
        zipper_msi = width * ROSS_ZIPPER_WIDTH_IN / 1000.0
        cost += ROSS_ZIPPER_COST_MSI * zipper_msi
    return cost


def _ross_gonderflex_spoilage(length_ft: float) -> float:
    """
    Look up Gonderflex spoilage % from the length-based table.
    Returns fraction (e.g. 0.05 for 5%).
    """
    for max_len, spoilage in ROSS_GONDERFLEX_SPOILAGE_TABLE:
        if length_ft <= max_len:
            return spoilage
    # Beyond table: use minimum
    return ROSS_GONDERFLEX_SPOILAGE_TABLE[-1][1]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw quote data into ML-ready features.

    Input df must have columns: width, height, gusset, quantity,
    plus the categorical spec columns.
    """
    df = df.copy()

    # ── Computed numeric features ───────────────────────────────────
    df["gusset"] = pd.to_numeric(df.get("gusset", 0), errors="coerce").fillna(0)
    df["width"] = pd.to_numeric(df["width"], errors="coerce")
    df["height"] = pd.to_numeric(df["height"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    df["print_width"] = df["height"] * 2 + df["gusset"]
    df["bag_area_sqin"] = df["width"] * df["height"]
    df["log_quantity"] = np.log10(df["quantity"].clip(lower=1))
    df["inv_quantity"] = 1.0 / df["quantity"].clip(lower=1)  # Fixed-cost amortization

    # ── Normalize substrate to canonical ────────────────────────────
    if "substrate" in df.columns:
        df["substrate"] = df["substrate"].apply(normalize_substrate)

    # ── Fill missing categoricals with safe defaults ────────────────
    defaults = {
        "substrate": "CUSTOM",
        "finish": "None",
        "fill_style": "Top",
        "seal_type": "Stand Up",
        "gusset_type": "None",
        "zipper": "No Zipper",
        "tear_notch": "None",
        "hole_punch": "None",
        "corner_treatment": "Straight",
        "embellishment": "None",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    # ── Interaction features ────────────────────────────────────────
    # Area × quantity — captures how material cost scales
    df["area_x_logqty"] = df["bag_area_sqin"] * df["log_quantity"]

    # Gusset presence (binary) — simplifies gusset impact
    df["has_gusset"] = (df["gusset"] > 0).astype(int)

    # Zipper complexity score (0-3)
    zipper_score = {"No Zipper": 0, "Single Profile Non-CR": 1,
                    "Double Profile Non-CR": 1.5, "Standard CR": 2,
                    "CR Zipper": 3, "Presto CR Zipper": 3.5}
    df["zipper_score"] = df["zipper"].map(zipper_score).fillna(0)

    # ── Ross cost-structure features ────────────────────────────────
    # These encode known relationships from Ross's equipment standards
    # to give the model physically meaningful signals.

    # Ross stock width: only 26" or 30" — binary, not continuous
    # This prevents print_width from dominating importance
    df["ross_stock_width"] = np.where(df["print_width"] > 26, 30.0, 26.0)

    # Zipper × width interaction: zipper cost scales with bag width
    has_zipper_bool = df["zipper_score"] > 0
    df["zipper_width"] = df["width"] * has_zipper_bool.astype(float)

    # Estimated print area in MSI per unit (for ink/priming cost signals)
    # MSI = (print_width × height) / 1000 — approximate, ignores repeat
    df["print_area_msi"] = (df["print_width"] * df["height"]) / 1000.0

    # Estimated Ross converting cost per unit (flat + zipper material)
    df["ross_converting_cost"] = df.apply(
        lambda r: _ross_converting_cost(r["width"], r["zipper_score"] > 0),
        axis=1,
    )

    return df


def build_preprocessor(vendor: str = "") -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that handles both numeric
    and ordinal encoding of categoricals.

    For Ross, print_width is replaced with ross_stock_width (binary 26/30)
    to prevent the continuous print_width from dominating importance.

    Note: No StandardScaler — GBR is tree-based so splits are
    invariant to scaling. Removing it avoids unnecessary state
    and was shown to improve performance in local testing.
    """
    # Numeric pipeline: passthrough (trees don't need scaling)
    numeric_pipe = "passthrough"

    # Categorical pipeline: ordinal encode (tree models handle ordinals well)
    cat_pipe = OrdinalEncoder(
        categories=[CATEGORY_ORDERS.get(f, ["unknown"]) for f in CATEGORICAL_FEATURES],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    base_numeric = [
        "width", "height", "gusset",
        "bag_area_sqin",
        "quantity", "log_quantity", "inv_quantity",
        "area_x_logqty", "has_gusset", "zipper_score",
        "zipper_width", "print_area_msi",
        "ross_converting_cost",
    ]

    if vendor == "ross":
        # Ross only uses 26" or 30" stock — binary feature instead
        all_numeric = base_numeric + ["ross_stock_width"]
    else:
        # Other vendors use continuous print_width
        all_numeric = ["print_width"] + base_numeric

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, all_numeric),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Get human-readable feature names after transformation."""
    # Extract numeric feature names from the fitted preprocessor
    num_features = preprocessor.transformers_[0][2]  # the column list
    return list(num_features) + CATEGORICAL_FEATURES


def save_preprocessor(preprocessor: ColumnTransformer, path: str):
    """Persist fitted preprocessor."""
    joblib.dump(preprocessor, path)
    logger.info(f"Saved preprocessor to {path}")


def load_preprocessor(path: str) -> ColumnTransformer:
    """Load a fitted preprocessor."""
    return joblib.load(path)
