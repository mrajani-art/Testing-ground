"""
Feature engineering for the Tedpack landed-cost ML model.

Tedpack is different from Dazpak/Ross/Internal — it predicts:
  1. FOB price (from bag specs)
  2. Full landed cost (FOB + freight + tariff + customs + insurance)

The external cost inputs (tariff_rate, container_rate, fx_rate) are
passed at prediction time, not baked into training features for FOB.
"""
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# ── Feature definitions ────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    # Base bag features
    'width', 'height', 'gusset', 'print_width', 'bag_area_sqin',
    'quantity', 'log_quantity', 'inv_quantity',
    # Interaction features
    'area_x_logqty', 'has_gusset', 'zipper_score',
    # Tedpack-specific
    'has_zipper', 'zipper_width', 'print_area_msi', 'estimated_weight_g',
]

CATEGORICAL_FEATURES = [
    'substrate', 'finish', 'fill_style', 'seal_type', 'gusset_type',
    'zipper', 'tear_notch', 'hole_punch', 'corner_treatment', 'embellishment',
]

# External cost features — FOB model does NOT use these
# Only used if training a separate landed model
EXTERNAL_FEATURES = [
    'tariff_rate', 'freight_per_unit', 'fx_cnyusd',
    'customs_per_unit', 'insurance_per_unit',
]

ALL_FEATURES_FOB = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Ordinal encoding order (cheapest → most expensive per category)
ORDINAL_MAPPINGS = {
    'substrate':        ['CLR PET', 'MET PET', 'Kraft', 'WHT MET PET', 'ALOX PET', 'HB CLR PET'],
    'finish':           ['None', 'Gloss', 'Matte', 'Soft Touch'],
    'fill_style':       ['Top Fill', 'Bottom Fill', 'Side Fill'],
    'seal_type':        ['Back Seal', '3-Side Seal', 'Stand Up Pouch', 'Flat Bottom', 'Quad Seal'],
    'gusset_type':      ['None', 'Plow', 'K-Seal', 'Box Bottom'],
    'zipper':           ['None', 'Press to Close', 'Slider', 'Velcro', 'Child Resistant'],
    'tear_notch':       ['None', 'Standard', 'Laser Score'],
    'hole_punch':       ['None', 'Standard', 'Euro Slot'],
    'corner_treatment': ['None', 'Rounded', 'Doyen', 'Butterfly'],
    'embellishment':    ['None', 'Spot UV', 'Emboss', 'Foil Stamp', 'Window'],
}

ZIPPER_SCORES = {
    'None': 0, 'No Zipper': 0,
    'Press to Close': 1.0,
    'Slider': 2.0,
    'Velcro': 2.5,
    'Child Resistant': 3.0,
    'CR Zipper': 3.0,
}


def build_spec_row(spec: dict, qty: int) -> dict:
    """
    Build a single feature row from a bag spec dict + quantity.
    spec keys: width, height, gusset, substrate, finish, fill_style,
               seal_type, gusset_type, zipper, tear_notch, hole_punch,
               corner_treatment, embellishment
    """
    width  = float(spec.get('width', 0))
    height = float(spec.get('height', 0))
    gusset = float(spec.get('gusset', 0))
    zipper_val = spec.get('zipper', 'None')
    zipper_score = ZIPPER_SCORES.get(zipper_val, 0)
    has_z = 1 if zipper_score > 0 else 0
    has_g = 1 if gusset > 0 else 0

    bag_area = width * height
    print_width = height * 2 + gusset
    log_qty = math.log10(max(qty, 1))

    return {
        'width': width,
        'height': height,
        'gusset': gusset,
        'print_width': print_width,
        'bag_area_sqin': bag_area,
        'quantity': qty,
        'log_quantity': log_qty,
        'inv_quantity': 1.0 / max(qty, 1),
        'area_x_logqty': bag_area * log_qty,
        'has_gusset': has_g,
        'zipper_score': zipper_score,
        'has_zipper': has_z,
        'zipper_width': width * has_z,
        'print_area_msi': print_width * height / 1000.0,
        'estimated_weight_g': (
            width * height * 2 * 0.12
            + (gusset * height * 2 * 0.12 if gusset > 0 else 0)
            + (2.5 if has_z else 0)
        ),
        # Categoricals
        'substrate':        spec.get('substrate', 'CLR PET'),
        'finish':           spec.get('finish', 'None'),
        'fill_style':       spec.get('fill_style', 'Top Fill'),
        'seal_type':        spec.get('seal_type', 'Stand Up Pouch'),
        'gusset_type':      spec.get('gusset_type', 'None'),
        'zipper':           zipper_val,
        'tear_notch':       spec.get('tear_notch', 'None'),
        'hole_punch':       spec.get('hole_punch', 'None'),
        'corner_treatment': spec.get('corner_treatment', 'None'),
        'embellishment':    spec.get('embellishment', 'None'),
    }


def encode_features(df: pd.DataFrame, fit: bool = True,
                    encoder: OrdinalEncoder = None):
    """
    Ordinal-encode categorical features.
    Returns (encoded_df, fitted_encoder).
    """
    df_enc = df.copy()
    if fit:
        encoder = OrdinalEncoder(
            categories=[ORDINAL_MAPPINGS[c] for c in CATEGORICAL_FEATURES],
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        )
        df_enc[CATEGORICAL_FEATURES] = encoder.fit_transform(df_enc[CATEGORICAL_FEATURES])
    else:
        df_enc[CATEGORICAL_FEATURES] = encoder.transform(df_enc[CATEGORICAL_FEATURES])
    return df_enc, encoder


def compute_recency_weights(quote_dates, boost_days=90, half_life_days=180,
                            floor_weight=0.2, max_weight=3.0) -> np.ndarray:
    """
    Weight training samples by recency.
    Recent quotes (within boost_days) get max_weight.
    Older quotes decay exponentially, floored at floor_weight.
    """
    dates = pd.to_datetime(quote_dates)
    ref = dates.max()
    age = (ref - dates).dt.days.values
    weights = np.where(
        age <= boost_days,
        max_weight,
        max_weight * np.power(2.0, -age / half_life_days)
    )
    return np.maximum(weights, floor_weight)


def freight_per_unit(qty: int, bag_area_sqin: float, gusset: float,
                     container_rate: float) -> tuple[float, str]:
    """
    Estimate per-unit freight cost based on bag volume and quantity.
    Uses LCL under 12 CBM, FCL_20 under 28 CBM, FCL_40 above.
    """
    pcbm = bag_area_sqin * 2 * 0.00001
    if gusset > 0:
        pcbm *= 1.5
    tcbm = qty * pcbm

    if tcbm <= 12:
        cost = max(tcbm * 118, 150)
        mode = 'LCL'
    elif tcbm <= 28:
        cost = container_rate * 0.65
        mode = 'FCL_20'
    elif tcbm <= 58:
        cost = container_rate
        mode = 'FCL_40'
    else:
        n = math.ceil(tcbm / 58)
        cost = n * container_rate
        mode = f'FCL_40x{n}'

    cost += 485 * max(1, math.ceil(tcbm / 58))
    return cost / qty, mode
