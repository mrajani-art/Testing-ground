"""
Tedpack Quote Predictor.

Separate from the Dazpak/Ross/Internal GBR pipeline because Tedpack
predicts FOB price AND builds a full landed cost breakdown
(freight + tariff + customs + insurance) at prediction time.

Usage:
    predictor = TedpackPredictor()
    predictor.load()
    result = predictor.predict(spec, quantities, tariff_rate=0.35, container_rate=2300)
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.ml.tedpack_features import (
    build_spec_row, encode_features, freight_per_unit, ALL_FEATURES_FOB,
)

logger = logging.getLogger(__name__)

# Default model paths — relative to repo root
MODEL_DIR = Path("models")
TEDPACK_POINT  = MODEL_DIR / "teapack_fob_point.joblib"
TEDPACK_LOWER  = MODEL_DIR / "teapack_fob_lower.joblib"
TEDPACK_UPPER  = MODEL_DIR / "teapack_fob_upper.joblib"
TEDPACK_ENCODER = MODEL_DIR / "teapack_encoder.joblib"


class TedpackPredictor:
    """
    Load trained Tedpack models and generate FOB + landed cost quotes.
    """

    def __init__(self):
        self.model_point = None
        self.model_lower = None
        self.model_upper = None
        self.encoder = None
        self._loaded = False

    def load(self):
        """Load .joblib files from models/ directory."""
        try:
            self.model_point  = joblib.load(TEDPACK_POINT)
            self.model_lower  = joblib.load(TEDPACK_LOWER)
            self.model_upper  = joblib.load(TEDPACK_UPPER)
            self.encoder      = joblib.load(TEDPACK_ENCODER)
            self._loaded = True
            logger.info("Tedpack models loaded successfully")
        except FileNotFoundError as e:
            logger.warning(f"Tedpack model files not found: {e}")
            logger.warning("Run the Tedpack training notebook and save .joblib files to models/")
            self._loaded = False

    @property
    def available(self) -> bool:
        return self._loaded

    def _predict_fob(self, spec: dict, qty: int) -> dict:
        """Predict FOB price for one quantity. Returns point, lower, upper."""
        row = build_spec_row(spec, qty)
        df = pd.DataFrame([row])
        df_enc, _ = encode_features(df, fit=False, encoder=self.encoder)
        X = df_enc[ALL_FEATURES_FOB].values

        fob_log    = float(self.model_point.predict(X)[0])
        lower_log  = float(self.model_lower.predict(X)[0])
        upper_log  = float(self.model_upper.predict(X)[0])

        return {
            'point': 10 ** fob_log,
            'lower': 10 ** lower_log,
            'upper': 10 ** upper_log,
        }

    def predict(self, spec: dict, quantities: list[int],
                tariff_rate: float = 0.35,
                container_rate: float = 2300,
                fx_rate: float = 0.1372) -> dict:
        """
        Generate a full Tedpack quote across quantity tiers.

        Args:
            spec: Bag spec dict (width, height, gusset, substrate, etc.)
            quantities: List of quantities to quote
            tariff_rate: Current tariff as decimal (e.g. 0.35 = 35%)
            container_rate: 40ft container cost in USD (e.g. 2300)
            fx_rate: CNY/USD exchange rate (informational only)

        Returns:
            {
                "vendor": "tedpack",
                "predictions": [{quantity, fob_unit, landed_unit,
                                  fob_lower, fob_upper,
                                  landed_lower, landed_upper,
                                  freight, tariff, customs, insurance,
                                  total_landed, shipping_mode}, ...],
                "cost_stack_pct": {fob, freight, tariff, customs, insurance},
                "warnings": [...],
                "available": True/False,
            }
        """
        if not self._loaded:
            return {
                "vendor": "tedpack",
                "predictions": [],
                "warnings": ["Tedpack model not loaded. Run training notebook and save .joblib files to models/."],
                "available": False,
            }

        warnings = []
        predictions = []

        for qty in quantities:
            fob = self._predict_fob(spec, qty)

            frt, mode = freight_per_unit(
                qty,
                float(spec.get('width', 0)) * float(spec.get('height', 0)),
                float(spec.get('gusset', 0)),
                container_rate,
            )

            tariff_cost  = fob['point'] * tariff_rate
            customs_cost = 320.0 / qty
            ins_cost     = (fob['point'] + frt) * 0.004

            landed = fob['point'] + frt + tariff_cost + customs_cost + ins_cost

            # CI bounds: apply same cost stack with lower/upper FOB
            lo_tariff = fob['lower'] * tariff_rate
            lo_ins    = (fob['lower'] + frt) * 0.004
            landed_lo = fob['lower'] + frt + lo_tariff + customs_cost + lo_ins

            hi_tariff = fob['upper'] * tariff_rate
            hi_ins    = (fob['upper'] + frt) * 0.004
            landed_hi = fob['upper'] + frt + hi_tariff + customs_cost + hi_ins

            predictions.append({
                'quantity':      qty,
                'fob_unit':      round(fob['point'], 5),
                'fob_lower':     round(fob['lower'], 5),
                'fob_upper':     round(fob['upper'], 5),
                'freight':       round(frt, 5),
                'tariff':        round(tariff_cost, 5),
                'customs':       round(customs_cost, 5),
                'insurance':     round(ins_cost, 5),
                'landed_unit':   round(landed, 5),
                'landed_lower':  round(landed_lo, 4),
                'landed_upper':  round(landed_hi, 4),
                'total_landed':  round(landed * qty, 2),
                'shipping_mode': mode,
            })

        # Cost stack breakdown at median quantity (for the UI breakdown bar)
        mid = predictions[len(predictions) // 2]
        total = mid['landed_unit']
        cost_stack_pct = {
            'fob':       round(mid['fob_unit']  / total * 100, 1),
            'freight':   round(mid['freight']   / total * 100, 1),
            'tariff':    round(mid['tariff']    / total * 100, 1),
            'customs':   round(mid['customs']   / total * 100, 1),
            'insurance': round(mid['insurance'] / total * 100, 1),
        }

        if tariff_rate >= 0.30:
            warnings.append(
                f"⚠ Tariff at {tariff_rate*100:.0f}% — verify current rate before quoting. "
                f"Tariff accounts for {cost_stack_pct['tariff']}% of landed cost."
            )

        return {
            'vendor': 'tedpack',
            'print_method': 'flexographic',
            'routing_reason': 'Tedpack: overseas gravure/flexo with landed cost',
            'predictions': predictions,
            'cost_stack_pct': cost_stack_pct,
            'external_rates': {
                'tariff_rate': tariff_rate,
                'container_rate': container_rate,
                'fx_rate': fx_rate,
            },
            'warnings': warnings,
            'available': True,
        }


def generate_penny_step_grid(spec: dict, predictor: TedpackPredictor,
                              tariff_rate: float, container_rate: float,
                              min_qty: int = 5000, max_qty: int = 200000,
                              n_points: int = 600) -> pd.DataFrame:
    """
    Generate the full penny-step curve for Plotly chart.
    Returns a DataFrame with one row per rounded-price change.
    """
    qtys = np.unique(
        np.logspace(np.log10(min_qty), np.log10(max_qty), n_points).astype(int)
    )

    result = predictor.predict(spec, qtys.tolist(), tariff_rate, container_rate)
    if not result['available']:
        return pd.DataFrame()

    rows = result['predictions']
    steps, prev = [], None
    for r in rows:
        p = round(r['landed_unit'], 2)
        if p != prev:
            steps.append({
                'qty':        r['quantity'],
                'landed':     p,
                'fob':        r['fob_unit'],
                'fob_lo':     r['fob_lower'],
                'fob_hi':     r['fob_upper'],
                'freight':    r['freight'],
                'tariff':     r['tariff'],
                'customs':    r['customs'],
                'insurance':  r['insurance'],
                'landed_lo':  r['landed_lower'],
                'landed_hi':  r['landed_upper'],
                'mode':       r['shipping_mode'],
            })
            prev = p
    return pd.DataFrame(steps)
